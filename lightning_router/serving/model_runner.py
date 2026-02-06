"""
Custom vLLM model runner that plugs LightningRouter's Triton kernels into the
vLLM inference engine.

Architecture
────────────
vLLM's ``LLMEngine`` delegates per-step execution to a ``ModelRunner``.  We
subclass it to:

1. Replace the default dense FFN with our ``MoELayer`` (Triton routing +
   optional INT4 expert FFN).
2. Register custom attention (GQA) and expert op implementations so that
   ``CUDAGraph`` capture still works.
3. Expose a thin API that the ``LightningRouterWorker`` calls during each
   decode step.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class LightningModelRunner:
    """
    Lightweight model runner that wraps a HuggingFace-style MoE model
    and replaces its FFN layers with LightningRouter's optimised MoE layers.

    In production this would subclass ``vllm.worker.model_runner.ModelRunner``.
    Here we provide the interface skeleton that integrates with vLLM's
    execution loop.
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        moe_config: dict[str, Any],
        quant_config: dict[str, Any],
        kernel_config: dict[str, Any],
        device: torch.device | str = "cuda",
    ):
        self.device = torch.device(device)
        self.model_config = model_config
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.kernel_config = kernel_config
        self.model = None

    # ── Model initialisation ──────────────────────────────────────────

    def load_model(self, checkpoint_path: str | Path | None = None) -> None:
        """
        Instantiate the transformer model and swap dense FFN → MoELayer.

        If ``checkpoint_path`` is provided, load weights (including
        pre-quantised INT4 buffers).  Otherwise initialise randomly (useful
        for benchmarking).
        """
        from lightning_router.config import (
            ModelConfig, MoEConfig, QuantizationConfig, KernelConfig,
        )
        from lightning_router.models.moe_layer import MoELayer

        mcfg = ModelConfig(**self.model_config)
        moecfg = MoEConfig(**self.moe_config)
        qcfg = QuantizationConfig(**self.quant_config)
        kcfg = KernelConfig(**self.kernel_config)

        # For demonstration, create a single MoE block
        self.moe_layer = MoELayer(mcfg, moecfg, qcfg, kcfg).to(self.device)

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.moe_layer.load_state_dict(state, strict=False)
            logger.info("Loaded checkpoint from %s", checkpoint_path)

        self.moe_layer.eval()
        logger.info("Model loaded on %s (quantised=%s)", self.device, qcfg.enabled)

    # ── Execution interface (called by vLLM Worker) ───────────────────

    @torch.inference_mode()
    def execute_model(
        self,
        input_ids: torch.Tensor,        # (batch, seq_len)
        hidden_states: torch.Tensor,     # (batch, seq_len, hidden)
    ) -> torch.Tensor:
        """
        Run one decode step through the MoE FFN layer.

        In a full integration this would run the entire transformer forward
        pass.  Here we focus on the MoE routing + expert execution path.
        """
        output, aux_loss = self.moe_layer(hidden_states)
        return output

    # ── Profiling hooks ───────────────────────────────────────────────

    def enable_profiling(self) -> None:
        """Attach CUDA event timers around the routing kernel."""
        self._prof_start = torch.cuda.Event(enable_timing=True)
        self._prof_end = torch.cuda.Event(enable_timing=True)

    def profile_step(self, hidden_states: torch.Tensor) -> dict[str, float]:
        """Run one step and return timing breakdown."""
        self._prof_start.record()
        output, _ = self.moe_layer(hidden_states)
        self._prof_end.record()
        torch.cuda.synchronize()
        elapsed_ms = self._prof_start.elapsed_time(self._prof_end)
        return {"moe_layer_ms": elapsed_ms}
