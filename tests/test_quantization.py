"""
Unit tests for 4-bit quantization pack/unpack utilities and quantized matmul.
"""

from __future__ import annotations

import pytest
import torch


class TestPackWeights:
    """CPU-safe tests for quantization packing."""

    def test_pack_unpack_roundtrip(self):
        """Packing then unpacking should recover the original INT4 values."""
        from lightning_router.quantization.pack_weights import _pack_int4, _unpack_int4

        N = 64
        original = torch.randint(0, 16, (4, N), dtype=torch.int32)
        packed = _pack_int4(original)
        assert packed.shape == (4, N // 8)

        unpacked = _unpack_int4(packed, N)
        torch.testing.assert_close(unpacked, original, msg="Pack/unpack mismatch")

    def test_quantize_dequantize_error(self):
        """Quantize → dequantize error should be small for typical weight magnitudes."""
        from lightning_router.quantization.pack_weights import quantize_tensor, dequantize_tensor

        K, N = 256, 64
        weight = torch.randn(K, N, dtype=torch.float16) * 0.02  # typical init scale

        qw, scales, zeros = quantize_tensor(weight, group_size=128)
        recon = dequantize_tensor(qw, scales, zeros, group_size=128, N=N)

        error = (weight - recon).float().norm() / weight.float().norm()
        assert error < 0.15, f"Quantisation error too large: {error:.4f}"

    def test_group_size_compatibility(self):
        """Should raise if K is not divisible by group_size."""
        from lightning_router.quantization.pack_weights import quantize_tensor

        with pytest.raises(AssertionError):
            quantize_tensor(torch.randn(100, 64, dtype=torch.float16), group_size=128)


@pytest.mark.gpu
class TestQuantizedMatmul:
    """GPU tests for the Triton quantized GEMM kernel."""

    def test_output_shape(self):
        """Output should be (M, N)."""
        from lightning_router.quantization.pack_weights import quantize_tensor
        from lightning_router.kernels.quantized_matmul import quantized_matmul

        M, K, N = 32, 256, 64
        A = torch.randn(M, K, dtype=torch.float16, device="cuda")
        W = torch.randn(K, N, dtype=torch.float16)
        qw, s, z = quantize_tensor(W, group_size=128)

        C = quantized_matmul(
            A, qw.cuda(), s.cuda(), z.cuda(), N=N, group_size=128,
        )
        assert C.shape == (M, N)

    def test_approximate_correctness(self):
        """Quantized matmul should be close to fp16 matmul (within quant error)."""
        from lightning_router.quantization.pack_weights import quantize_tensor, dequantize_tensor
        from lightning_router.kernels.quantized_matmul import quantized_matmul

        M, K, N = 64, 256, 64
        A = torch.randn(M, K, dtype=torch.float16, device="cuda")
        W = torch.randn(K, N, dtype=torch.float16) * 0.02

        qw, s, z = quantize_tensor(W, group_size=128)
        W_recon = dequantize_tensor(qw, s, z, 128, N).cuda()

        ref = A @ W_recon
        out = quantized_matmul(A, qw.cuda(), s.cuda(), z.cuda(), N=N, group_size=128)

        # Relative error should be modest (quantisation noise + fp16 accumulation)
        rel_err = (out - ref).float().norm() / ref.float().norm()
        assert rel_err < 0.2, f"Quantized matmul error too large: {rel_err:.4f}"
