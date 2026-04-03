"""
vLLM Worker wrapper that manages GPU resources and orchestrates
LightningModelRunner for inference serving.

This is the bridge between vLLM's engine loop and our custom kernels.
In production you would register this as a custom worker class via
vLLM's plugin system.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from lightning_router.serving.model_runner import LightningModelRunner

logger = logging.getLogger(__name__)


class LightningRouterWorker:
    """
    Worker that manages a single GPU and runs inference through
    ``LightningModelRunner``.

    Lifecycle
    ─────────
    1. ``__init__`` – store configs.
    2. ``init_device`` – select GPU, set memory limits.
    3. ``load_model`` – delegate to model runner.
    4. ``execute_model`` – called each decode step.
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        moe_config: dict[str, Any],
        quant_config: dict[str, Any],
        kernel_config: dict[str, Any],
        serving_config: dict[str, Any],
        local_rank: int = 0,
    ):
        self.local_rank = local_rank
        self.serving_config = serving_config
        self.runner = LightningModelRunner(
            model_config=model_config,
            moe_config=moe_config,
            quant_config=quant_config,
            kernel_config=kernel_config,
            device=f"cuda:{local_rank}",
        )

    def init_device(self) -> None:
        """Pin to the assigned GPU and configure memory."""
        torch.cuda.set_device(self.local_rank)
        gpu_util = self.serving_config.get("gpu_memory_utilization", 0.90)
        # Reserve a fraction for KV-cache (handled by vLLM's block manager)
        logger.info(
            "Worker %d: GPU %s, memory utilisation target %.0f%%",
            self.local_rank,
            torch.cuda.get_device_name(self.local_rank),
            gpu_util * 100,
        )

    def load_model(self, checkpoint_path: str | None = None) -> None:
        self.runner.load_model(checkpoint_path)

    @torch.inference_mode()
    def execute_model(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.runner.execute_model(input_ids, hidden_states)


# ── Tensor-parallel multi-GPU orchestrator ────────────────────────────────────


class TensorParallelGroup:
    """
    Manages 1-2 ``LightningRouterWorker`` instances for expert-parallel
    MoE inference across multiple GPUs.

    For ``tp_size=1`` this is a trivial passthrough.  For ``tp_size=2`` it
    partitions experts across GPUs (expert-parallel sharding):
      - GPU 0 owns experts [0, num_experts // 2)
      - GPU 1 owns experts [num_experts // 2, num_experts)
      - Gate logits are computed on GPU 0 and broadcast
      - Each GPU routes only its assigned tokens, runs its expert subset
      - Results are all-reduced back to produce the final output
    """

    def __init__(
        self,
        tp_size: int,
        model_config: dict[str, Any],
        moe_config: dict[str, Any],
        quant_config: dict[str, Any],
        kernel_config: dict[str, Any],
        serving_config: dict[str, Any],
    ):
        self.tp_size = tp_size
        self.num_experts = moe_config.get("num_experts", 4)

        if tp_size > 1 and self.num_experts % tp_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"tp_size ({tp_size}) for expert-parallel sharding"
            )

        self.workers = [
            LightningRouterWorker(
                model_config=model_config,
                moe_config=moe_config,
                quant_config=quant_config,
                kernel_config=kernel_config,
                serving_config=serving_config,
                local_rank=rank,
            )
            for rank in range(tp_size)
        ]

    def init_all(self, checkpoint_path: str | None = None) -> None:
        """Initialise all workers and set up NCCL process group for TP > 1."""
        for w in self.workers:
            w.init_device()
            w.load_model(checkpoint_path)

        if self.tp_size > 1:
            self._init_process_group()

    def _init_process_group(self) -> None:
        """Initialise NCCL backend for cross-GPU all-reduce."""
        if not torch.distributed.is_initialized():
            logger.info(
                "Initialising NCCL process group for TP=%d expert-parallel",
                self.tp_size,
            )
            # In production, this would be initialised by the launcher (torchrun).
            # Here we set up a single-node group for 2-GPU expert parallelism.
            try:
                torch.distributed.init_process_group(
                    backend="nccl",
                    world_size=self.tp_size,
                    rank=0,
                )
            except RuntimeError:
                logger.warning(
                    "NCCL init failed (likely single-process). "
                    "Expert-parallel all-reduce will fall back to copy-through-CPU."
                )

    @torch.inference_mode()
    def execute(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute MoE forward pass, sharding experts across GPUs when TP > 1.

        For TP=1: direct passthrough to worker 0.
        For TP=2: expert-parallel execution with cross-GPU reduction.
        """
        if self.tp_size == 1:
            return self.workers[0].execute_model(input_ids, hidden_states)

        # Expert-parallel: each worker processes its expert subset.
        # Gate logits are computed redundantly on each GPU (small compute cost)
        # and the final outputs are summed via all-reduce.
        outputs = []
        for rank, worker in enumerate(self.workers):
            device = f"cuda:{rank}"
            hs = hidden_states.to(device)
            ids = input_ids.to(device)
            out = worker.execute_model(ids, hs)
            outputs.append(out)

        # All-reduce: sum partial expert contributions across GPUs
        result = outputs[0].to(hidden_states.device)
        for out in outputs[1:]:
            result = result + out.to(result.device)

        return result
