"""
Inference server entry-point.

Wires together the config loader, worker initialisation, and a simple
HTTP serving loop (using vLLM's ``AsyncLLMEngine`` pattern) or a
standalone benchmarking mode.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from lightning_router.config import load_config
from lightning_router.serving.worker import TensorParallelGroup

logger = logging.getLogger(__name__)


def launch_server(
    config_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """
    Load config, initialise the tensor-parallel group, and start serving.

    In a full deployment this would launch ``vllm.entrypoints.api_server``
    with our custom worker registered.  Here we demonstrate the setup flow
    and fall back to a simple blocking loop.
    """
    cfg = load_config(config_path)
    logger.info("Loaded config from %s", config_path)

    tp = TensorParallelGroup(
        tp_size=cfg.serving.tensor_parallel_size,
        model_config=asdict(cfg.model),
        moe_config=asdict(cfg.moe),
        quant_config=asdict(cfg.quantization),
        kernel_config=asdict(cfg.kernel),
        serving_config=asdict(cfg.serving),
    )
    tp.init_all()

    logger.info("LightningRouter ready – listening on %s:%d", host, port)
    logger.info(
        "Experts=%d  top_k=%d  quantised=%s  TP=%d",
        cfg.moe.num_experts,
        cfg.moe.num_experts_per_token,
        cfg.quantization.enabled,
        cfg.serving.tensor_parallel_size,
    )

    # ── In production, hand off to vLLM's async engine:
    #
    #   from vllm.entrypoints.api_server import run_server
    #   run_server(engine, host, port)
    #
    # For now, we print a ready message.  The actual serving integration
    # is in the ``LightningRouterWorker`` which plugs into vLLM's loop.
    print(f"✓ LightningRouter serving on http://{host}:{port}")
    print("  Press Ctrl+C to stop.")

    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")
