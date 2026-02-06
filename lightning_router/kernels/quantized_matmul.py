"""
4-bit Quantized Matrix-Multiply Triton Kernel
==============================================

Implements a GEMM where the **B** matrix (expert weights) is stored in packed
4-bit format (INT4) with per-group scale+zero-point dequantisation.

Layout
------
* A  – (M, K)  fp16  activations  (tokens × hidden)
* B  – (K, N // 8)  int32  packed weights  (8 × int4 per element)
* scales  – (K // group_size, N)  fp16
* zeros   – (K // group_size, N // 8)  int32  packed like B

The kernel tiles over (M, N, K), dequantises B on-the-fly inside shared
memory, and accumulates in fp32 before down-casting the result to fp16.

Key optimisations
─────────────────
* Coalesced loads of packed B columns via BLOCK_K × BLOCK_N tiles.
* Shared-memory staging for unpacked + dequantised B tiles.
* Software-pipelined loads (``num_stages``) to overlap global loads with
  compute.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_int4(packed: tl.tensor, bit_idx: tl.tensor):
    """Extract a 4-bit value from a packed int32 at *bit_idx* (0..7)."""
    shift = bit_idx * 4
    return ((packed >> shift) & 0xF).to(tl.float16)


@triton.jit
def _quantized_matmul_kernel(
    # Pointers
    A_ptr, B_ptr, scales_ptr, zeros_ptr, C_ptr,
    # Matrix dims
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    # Quant params
    GROUP_SIZE: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Tiled GEMM: C[M,N] = A[M,K] @ dequant(B[K,N])

    B is packed int4 → 8 values per int32.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ── Main K-loop ───────────────────────────────────────────────────
    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)

        # ── Load A tile  (M × K) — coalesced along K ─────────────────
        a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)

        # ── Load + dequantise B tile  (K × N) ────────────────────────
        # B is stored packed: shape (K, N//8) int32
        packed_col = rn[None, :] // 8            # which int32 element
        bit_idx = (rn[None, :] % 8).to(tl.int32) # which nibble inside
        b_ptrs = B_ptr + rk[:, None] * stride_bk + packed_col * stride_bn
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        packed_vals = tl.load(b_ptrs, mask=b_mask, other=0)

        b_raw = _dequant_int4(packed_vals, bit_idx)  # 0-15 range

        # Per-group scale & zero-point
        group_idx = rk[:, None] // GROUP_SIZE
        s_ptrs = scales_ptr + group_idx * stride_scales_g + rn[None, :] * stride_scales_n
        s_mask = (group_idx < (K // GROUP_SIZE)) & (rn[None, :] < N)
        scale = tl.load(s_ptrs, mask=s_mask, other=1.0).to(tl.float16)

        z_packed_col = rn[None, :] // 8
        z_bit_idx = (rn[None, :] % 8).to(tl.int32)
        z_ptrs = zeros_ptr + group_idx * stride_zeros_g + z_packed_col * stride_zeros_n
        z_packed = tl.load(z_ptrs, mask=s_mask, other=0)
        zero = _dequant_int4(z_packed, z_bit_idx)

        # Dequantise: w = scale * (raw - zero)
        b_dequant = (scale * (b_raw - zero)).to(tl.float16)

        # ── Tile multiply-accumulate ──────────────────────────────────
        acc += tl.dot(a, b_dequant).to(tl.float32)

    # ── Store C tile ──────────────────────────────────────────────────
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Python wrapper
# ──────────────────────────────────────────────────────────────────────────────


def quantized_matmul(
    A: torch.Tensor,        # (M, K) fp16
    B_packed: torch.Tensor, # (K, N // 8) int32
    scales: torch.Tensor,   # (num_groups, N) fp16
    zeros: torch.Tensor,    # (num_groups, N // 8) int32
    N: int,                 # unpacked output columns
    group_size: int = 128,
    block_m: int = 128,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    """
    Compute ``C = A @ dequant(B)`` where B is 4-bit packed.

    Parameters
    ----------
    A : (M, K) fp16 activation tensor.
    B_packed : (K, N//8) int32 packed 4-bit weights.
    scales, zeros : per-group quantisation parameters.
    N : true (unpacked) number of output columns.
    group_size : number of elements per quant group along K.

    Returns
    -------
    C : (M, N) fp16 result.
    """
    M, K = A.shape
    assert A.dtype == torch.float16
    assert B_packed.dtype == torch.int32

    C = torch.empty(M, N, dtype=torch.float16, device=A.device)

    grid = (
        triton.cdiv(M, block_m),
        triton.cdiv(N, block_n),
    )

    _quantized_matmul_kernel[grid](
        A, B_packed, scales, zeros, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B_packed.stride(0), B_packed.stride(1),
        C.stride(0), C.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    return C
