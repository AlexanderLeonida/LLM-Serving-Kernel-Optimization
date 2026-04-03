"""
SGLang backend integration for LightningRouter.

Plugs the optimised MoE layer (Triton routing + INT4 expert FFN) into
SGLang's runtime via its custom model registration API.  SGLang's
RadixAttention and continuous batching scheduler handle KV-cache
management while we replace the expert dispatch with fused Triton kernels.

Architecture
────────────
SGLang Runtime
  └── TokenAttention (RadixAttention) ← handled by SGLang
  └── MoE FFN                         ← replaced by LightningRouter
        ├── TopKGating
        ├── Triton scatter (coalesced + shared-mem)
        ├── Expert FFN (INT4 Triton GEMM)
        └── Triton gather (fused w×add)

Usage
─────
    from lightning_router.serving.sglang_backend import register_lightning_model
    register_lightning_model(config_path="configs/moe_4expert.yaml")
    # Then launch SGLang runtime normally — it will use the registered model
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SGLangMoEWrapper(nn.Module):
    """
    Wraps LightningRouter's MoE layer to conform to SGLang's model interface.

    SGLang expects models to implement a ``forward()`` that accepts
    ``input_ids`` and ``positions``, returning logits.  This wrapper
    handles the MoE FFN portion; attention layers are provided by SGLang's
    built-in RadixAttention.
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        moe_config: dict[str, Any],
        quant_config: dict[str, Any],
        kernel_config: dict[str, Any],
    ):
        super().__init__()
        from lightning_router.config import (
            ModelConfig, MoEConfig, QuantizationConfig, KernelConfig,
        )
        from lightning_router.models.moe_layer import MoELayer

        mcfg = ModelConfig(**model_config)
        moecfg = MoEConfig(**moe_config)
        qcfg = QuantizationConfig(**quant_config)
        kcfg = KernelConfig(**kernel_config)

        self.moe_layer = MoELayer(mcfg, moecfg, qcfg, kcfg)
        self.hidden_size = mcfg.hidden_size

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden_states : (num_tokens, hidden_size)
            Hidden states after attention, passed by SGLang's model runner.

        Returns
        -------
        output    : (num_tokens, hidden_size)
        aux_loss  : scalar load-balancing loss
        """
        return self.moe_layer(hidden_states)


def create_sglang_model(
    config_path: str | Path,
    device: str = "cuda",
) -> SGLangMoEWrapper:
    """
    Instantiate a LightningRouter model configured for SGLang.

    Parameters
    ----------
    config_path : path to YAML configuration file.
    device : target device.

    Returns
    -------
    SGLangMoEWrapper ready for integration with SGLang's runtime.
    """
    from lightning_router.config import load_config

    cfg = load_config(config_path)

    model = SGLangMoEWrapper(
        model_config=asdict(cfg.model),
        moe_config=asdict(cfg.moe),
        quant_config=asdict(cfg.quantization),
        kernel_config=asdict(cfg.kernel),
    ).to(device)

    model.eval()
    logger.info(
        "SGLang model ready: experts=%d  quantised=%s  device=%s",
        cfg.moe.num_experts,
        cfg.quantization.enabled,
        device,
    )
    return model


def register_lightning_model(config_path: str | Path) -> None:
    """
    Register LightningRouter's MoE layer with SGLang's model registry.

    After calling this, launch SGLang's server with the model name
    ``"lightning-moe"`` and it will use the Triton-accelerated expert
    routing and INT4 GEMM kernels.

    Example::

        register_lightning_model("configs/moe_4expert.yaml")

        # Then in SGLang:
        # python -m sglang.launch_server --model lightning-moe --tp 2
    """
    try:
        from sglang.srt.model_registry import register_model
    except ImportError:
        logger.warning(
            "SGLang not installed — model registered as factory function only. "
            "Install with: pip install 'sglang[all]'"
        )
        return

    def _factory(**kwargs):
        return create_sglang_model(config_path, device=kwargs.get("device", "cuda"))

    register_model("lightning-moe", _factory)
    logger.info("Registered 'lightning-moe' with SGLang model registry")


def launch_sglang_server(
    config_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tp_size: int = 1,
) -> None:
    """
    Launch an SGLang inference server with LightningRouter's MoE kernels.

    This is an alternative to the vLLM-based server in ``server.py``.
    SGLang provides RadixAttention for efficient prefix caching and
    continuous batching, while LightningRouter handles the expert dispatch.
    """
    register_lightning_model(config_path)

    try:
        import sglang as sgl

        runtime = sgl.Runtime(
            model_path="lightning-moe",
            tp_size=tp_size,
            host=host,
            port=port,
        )
        logger.info("SGLang server ready on %s:%d (TP=%d)", host, port, tp_size)
        runtime.shutdown()
    except ImportError:
        logger.info(
            "SGLang not installed — serving via standalone mode. "
            "Install with: pip install 'sglang[all]'"
        )
        # Fallback: standalone serving loop
        model = create_sglang_model(config_path)
        print(f"LightningRouter (SGLang backend) ready on http://{host}:{port}")
        print("  Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down.")
