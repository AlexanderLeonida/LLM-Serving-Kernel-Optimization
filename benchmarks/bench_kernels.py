"""
Benchmark suite using pytest-benchmark.

Run with:  pytest benchmarks/ --benchmark-only -v
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"


class TestRoutingKernelBenchmark:
    """Benchmark the Triton expert-routing scatter/gather kernels."""

    @pytest.mark.parametrize("num_tokens", [1024, 4096, 16384])
    @pytest.mark.parametrize("num_experts", [4])
    def test_routing_throughput(self, benchmark, device, num_tokens, num_experts):
        from lightning_router.kernels.expert_routing import expert_routing_forward

        hidden = 4096
        top_k = 2
        tokens = torch.randn(num_tokens, hidden, dtype=torch.float16, device=device)
        expert_ids = torch.randint(0, num_experts, (num_tokens * top_k,), dtype=torch.int32, device=device)
        gate_weights = torch.rand(num_tokens * top_k, dtype=torch.float32, device=device)
        token_indices = (
            torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1).int()
        )

        def run():
            expert_routing_forward(
                tokens, expert_ids, gate_weights, token_indices, num_experts,
            )
            torch.cuda.synchronize()

        benchmark(run)


class TestQuantizedMatmulBenchmark:
    """Benchmark the Triton 4-bit quantized GEMM."""

    @pytest.mark.parametrize(
        "M,K,N",
        [(1, 4096, 11008), (32, 4096, 11008), (128, 4096, 11008)],
    )
    def test_qmatmul_latency(self, benchmark, device, M, K, N):
        from lightning_router.quantization.pack_weights import quantize_tensor
        from lightning_router.kernels.quantized_matmul import quantized_matmul

        A = torch.randn(M, K, dtype=torch.float16, device=device)
        W = torch.randn(K, N, dtype=torch.float16)
        qw, s, z = quantize_tensor(W, group_size=128)
        qw, s, z = qw.to(device), s.to(device), z.to(device)

        def run():
            quantized_matmul(A, qw, s, z, N=N, group_size=128)
            torch.cuda.synchronize()

        benchmark(run)


class TestMoELayerBenchmark:
    """End-to-end MoE layer benchmark (gating + routing + experts + gather)."""

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_moe_layer_throughput(self, benchmark, device, batch_size):
        from lightning_router.profiling.benchmark_kernels import get_moe_layer_benchmark_fn

        fn, args = get_moe_layer_benchmark_fn(
            batch_size=batch_size, seq_len=512, hidden_size=4096,
            intermediate_size=11008, num_experts=4, device=device,
        )

        def run():
            fn(*args)
            torch.cuda.synchronize()

        benchmark(run)
