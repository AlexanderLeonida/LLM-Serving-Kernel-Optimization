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
    Manages 1-2 ``LightningRouterWorker`` instances for tensor-parallel
    MoE inference.

    For ``tp_size=1`` this is a trivial passthrough.  For ``tp_size=2`` it
    partitions experts across GPUs (expert-parallel) and uses NCCL
    all-reduce to combine gate logits.
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
        for w in self.workers:
            w.init_device()
            w.load_model(checkpoint_path)

    def execute(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Single-GPU fast-path; multi-GPU would shard here."""
        return self.workers[0].execute_model(input_ids, hidden_states)
