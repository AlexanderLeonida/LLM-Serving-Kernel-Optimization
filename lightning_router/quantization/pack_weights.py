"""
4-bit weight packing / unpacking utilities.

Converts fp16 weight tensors into the packed INT4 format consumed by the
``quantized_matmul`` Triton kernel, and provides the inverse for validation.

Quantization scheme (asymmetric, per-group)
───────────────────────────────────────────
For each group of ``group_size`` contiguous elements along the K (input) dim:

    scale = (max_val - min_val) / 15
    zero  = round(-min_val / scale)          # stored packed like weights
    q     = clamp(round(w / scale) + zero, 0, 15)

Packing: 8 consecutive INT4 values are packed into a single int32.
"""

from __future__ import annotations

import torch


def quantize_tensor(
    weight: torch.Tensor,
    group_size: int = 128,
    scheme: str = "asymmetric",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a 2-D weight matrix to 4-bit packed format.

    Parameters
    ----------
    weight : (K, N) fp16 or fp32 tensor.
    group_size : elements per quantisation group along K.
    scheme : ``"asymmetric"`` or ``"symmetric"``.

    Returns
    -------
    qweight : (K, N // 8) int32 packed weights.
    scales  : (K // group_size, N) fp16 per-group scales.
    zeros   : (K // group_size, N // 8) int32 packed zero-points.
    """
    K, N = weight.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert N % 8 == 0, f"N={N} must be divisible by 8 for INT4 packing"

    weight = weight.float()
    num_groups = K // group_size

    # Reshape to (num_groups, group_size, N)
    w_grouped = weight.view(num_groups, group_size, N)

    if scheme == "asymmetric":
        w_min = w_grouped.amin(dim=1)  # (num_groups, N)
        w_max = w_grouped.amax(dim=1)
        scales_f = (w_max - w_min) / 15.0
        scales_f = scales_f.clamp(min=1e-8)
        zeros_f = torch.round(-w_min / scales_f).clamp(0, 15)
    else:  # symmetric
        w_absmax = w_grouped.abs().amax(dim=1)
        scales_f = w_absmax / 7.0
        scales_f = scales_f.clamp(min=1e-8)
        zeros_f = torch.full_like(scales_f, 8.0)  # mid-point

    # Quantize
    # Expand scales/zeros back to (num_groups, group_size, N)
    scales_exp = scales_f.unsqueeze(1).expand_as(w_grouped)
    zeros_exp = zeros_f.unsqueeze(1).expand_as(w_grouped)
    q = torch.round(w_grouped / scales_exp + zeros_exp).clamp(0, 15).to(torch.int32)

    # Flatten back to (K, N)
    q = q.view(K, N)

    # Pack: every 8 consecutive columns → 1 int32
    q_packed = _pack_int4(q)                # (K, N // 8)
    z_packed = _pack_int4(zeros_f.int())    # (num_groups, N // 8)

    return q_packed, scales_f.half(), z_packed


def _pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    """Pack the last dim by groups of 8 nibbles → int32."""
    *leading, N = tensor.shape
    assert N % 8 == 0
    tensor = tensor.view(*leading, N // 8, 8).to(torch.int32)
    packed = torch.zeros(*leading, N // 8, dtype=torch.int32, device=tensor.device)
    for i in range(8):
        packed |= (tensor[..., i] & 0xF) << (i * 4)
    return packed


def _unpack_int4(packed: torch.Tensor, N: int) -> torch.Tensor:
    """Unpack int32 → last-dim of 8 nibbles each."""
    *leading, packed_n = packed.shape
    assert packed_n == N // 8
    out = torch.empty(*leading, N, dtype=torch.int32, device=packed.device)
    for i in range(8):
        out[..., i::8] = (packed >> (i * 4)) & 0xF
    # Fix the ordering: unpack to contiguous groups of 8
    result = torch.empty(*leading, N, dtype=torch.int32, device=packed.device)
    for i in range(8):
        result[..., torch.arange(N // 8, device=packed.device) * 8 + i] = (
            (packed >> (i * 4)) & 0xF
        )
    return result


def dequantize_tensor(
    qweight: torch.Tensor,  # (K, N // 8) int32
    scales: torch.Tensor,   # (num_groups, N) fp16
    zeros: torch.Tensor,    # (num_groups, N // 8) int32
    group_size: int,
    N: int,
) -> torch.Tensor:
    """Dequantize packed INT4 weights back to fp16 (for validation)."""
    K = qweight.size(0)
    num_groups = scales.size(0)

    # Unpack weights
    q = _unpack_int4(qweight, N).float()        # (K, N)
    z = _unpack_int4(zeros, N).float()           # (num_groups, N)

    # Expand scales & zeros
    scales_f = scales.float().unsqueeze(1).expand(num_groups, group_size, N)
    zeros_f = z.unsqueeze(1).expand(num_groups, group_size, N)

    q_grouped = q.view(num_groups, group_size, N)
    w = scales_f * (q_grouped - zeros_f)

    return w.view(K, N).half()
