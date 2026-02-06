"""
Self-contained benchmark functions for individual kernels.

These are used both by the profiling runner (nsight_runner.py) and by
pytest-benchmark (tests/benchmarks/).
"""

from __future__ import annotations

from typing import Callable, Any

import torch


def get_routing_benchmark_fn(
    batch_size: int = 32,
    seq_len: int = 512,
    hidden_size: int = 4096,
    num_experts: int = 4,
    top_k: int = 2,
    device: str = "cuda",
) -> tuple[Callable, tuple]:
    """
    Return ``(fn, args)`` that runs the Triton expert-routing forward pass.

    Usage::

        fn, args = get_routing_benchmark_fn()
        fn(*args)  # one invocation
    """
    from lightning_router.kernels.expert_routing import expert_routing_forward

    num_tokens = batch_size * seq_len
    tokens = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
    expert_ids = torch.randint(0, num_experts, (num_tokens * top_k,), device=device, dtype=torch.int32)
    gate_weights = torch.rand(num_tokens * top_k, device=device, dtype=torch.float32)
    gate_weights = gate_weights / gate_weights.view(num_tokens, top_k).sum(dim=1, keepdim=True).repeat(1, top_k).view(-1)
    token_indices = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
        .int()
    )

    def fn():
        return expert_routing_forward(
            tokens, expert_ids, gate_weights, token_indices,
            num_experts=num_experts,
        )

    return fn, ()


def get_quantized_matmul_benchmark_fn(
    M: int = 4096,
    K: int = 4096,
    N: int = 11008,
    group_size: int = 128,
    device: str = "cuda",
) -> tuple[Callable, tuple]:
    """
    Return ``(fn, args)`` that runs the Triton 4-bit quantized matmul.
    """
    from lightning_router.quantization.pack_weights import quantize_tensor
    from lightning_router.kernels.quantized_matmul import quantized_matmul

    A = torch.randn(M, K, dtype=torch.float16, device=device)
    W_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    B_packed, scales, zeros = quantize_tensor(W_fp16, group_size=group_size)
    B_packed = B_packed.to(device)
    scales = scales.to(device)
    zeros = zeros.to(device)

    def fn():
        return quantized_matmul(A, B_packed, scales, zeros, N=N, group_size=group_size)

    return fn, ()


def get_moe_layer_benchmark_fn(
    batch_size: int = 32,
    seq_len: int = 512,
    hidden_size: int = 4096,
    intermediate_size: int = 11008,
    num_experts: int = 4,
    top_k: int = 2,
    quantized: bool = False,
    device: str = "cuda",
) -> tuple[Callable, tuple]:
    """Return ``(fn, args)`` for an end-to-end MoE layer forward pass."""
    from lightning_router.config import ModelConfig, MoEConfig, QuantizationConfig, KernelConfig
    from lightning_router.models.moe_layer import MoELayer

    mcfg = ModelConfig(hidden_size=hidden_size, intermediate_size=intermediate_size)
    moecfg = MoEConfig(num_experts=num_experts, num_experts_per_token=top_k)
    qcfg = QuantizationConfig(enabled=quantized)
    kcfg = KernelConfig()

    layer = MoELayer(mcfg, moecfg, qcfg, kcfg).to(device).half().eval()
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=device)

    def fn():
        with torch.inference_mode():
            return layer(x)

    return fn, ()
