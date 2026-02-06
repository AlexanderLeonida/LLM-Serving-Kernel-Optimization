"""
Utilities to quantize a full MoE model's expert weights in-place.

This module walks an ``MoELayer`` (or any module tree containing
``ExpertFFN`` instances), quantizes all fp16 weight matrices to INT4,
and replaces the ``nn.Linear`` layers with packed buffer triples
(qweight, scales, zeros) suitable for the Triton quantized_matmul kernel.
"""

from __future__ import annotations

import logging
from typing import Iterator

import torch
import torch.nn as nn

from lightning_router.quantization.pack_weights import quantize_tensor

logger = logging.getLogger(__name__)


def _iter_linear_layers(module: nn.Module, prefix: str = "") -> Iterator[tuple[str, nn.Linear]]:
    """Yield (qualified_name, nn.Linear) for every Linear in the tree."""
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            yield full, child
        else:
            yield from _iter_linear_layers(child, full)


@torch.no_grad()
def quantize_expert_weights(
    model: nn.Module,
    group_size: int = 128,
    scheme: str = "asymmetric",
    target_substrings: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
) -> dict[str, dict]:
    """
    Quantize all expert Linear layers in ``model`` to 4-bit.

    The function:
    1. Iterates over every ``nn.Linear`` whose name contains one of
       ``target_substrings``.
    2. Quantizes ``weight`` (K, N) → packed INT4 buffers.
    3. Registers the buffers on the parent module and deletes the original
       ``nn.Linear``.

    Parameters
    ----------
    model : nn.Module
        The model (or sub-tree) containing experts to quantize.
    group_size : int
        Quantisation group size along the K dimension.
    scheme : str
        ``"asymmetric"`` or ``"symmetric"``.
    target_substrings : tuple[str, ...]
        Only quantize layers whose name contains one of these.

    Returns
    -------
    stats : dict mapping layer name → {"orig_norm", "quant_error"} for QA.
    """
    stats: dict[str, dict] = {}

    for name, linear in list(_iter_linear_layers(model)):
        if not any(sub in name for sub in target_substrings):
            continue

        weight = linear.weight.data  # (out_features, in_features)
        K, N = weight.shape

        # Ensure dims are compatible
        if K % group_size != 0 or N % 8 != 0:
            logger.warning("Skipping %s (K=%d, N=%d) – incompatible dims", name, K, N)
            continue

        qweight, scales, zeros = quantize_tensor(weight.half(), group_size, scheme)

        # Record accuracy stats
        from lightning_router.quantization.pack_weights import dequantize_tensor

        w_recon = dequantize_tensor(qweight, scales, zeros, group_size, N)
        orig_norm = weight.half().norm().item()
        error_norm = (weight.half() - w_recon).norm().item()
        stats[name] = {"orig_norm": orig_norm, "quant_error": error_norm}
        logger.info(
            "Quantized %-40s  error/norm = %.4f",
            name,
            error_norm / (orig_norm + 1e-12),
        )

        # Replace the Linear with buffers on the parent
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr = name

        delattr(parent, attr)
        parent.register_buffer(f"{attr}_qweight", qweight)
        parent.register_buffer(f"{attr}_scales", scales)
        parent.register_buffer(f"{attr}_zeros", zeros)

    return stats
