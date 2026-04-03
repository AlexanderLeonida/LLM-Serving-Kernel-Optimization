"""
Baseline comparison: LightningRouter vs naive PyTorch MoE routing.

Implements the naive baseline (three separate global-memory passes) and
compares against the Triton fused implementation across varying batch sizes,
sequence lengths, and expert counts.

This produces the data for Table 1 / Figure 2 in a NeurIPS-style paper.

Usage:
    python benchmarks/baseline_comparison.py --output-dir results/baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: Naive PyTorch routing (3 separate passes)
# ─────────────────────────────────────────────────────────────────────────────


def naive_pytorch_routing(
    tokens: torch.Tensor,       # (num_tokens, hidden)
    expert_ids: torch.Tensor,   # (num_slots,) int
    gate_weights: torch.Tensor, # (num_slots,) float
    token_indices: torch.Tensor,# (num_slots,) int
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Naive 3-pass PyTorch routing baseline.

    Pass 1: torch.bincount for histogram
    Pass 2: torch.index_select for scatter (separate per expert)
    Pass 3: index_add_ for gather
    """
    num_tokens, hidden = tokens.shape

    # Pass 1: Histogram
    counts = torch.bincount(expert_ids.long(), minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=tokens.device)
    offsets[1:] = counts.cumsum(0)

    # Pass 2: Scatter — build expert-contiguous layout
    total = int(offsets[-1].item())
    permuted = torch.empty(total, hidden, dtype=tokens.dtype, device=tokens.device)
    routing_w = torch.empty(total, dtype=torch.float32, device=tokens.device)
    src_idx = torch.empty(total, dtype=torch.int64, device=tokens.device)

    write_pos = torch.zeros(num_experts, dtype=torch.int64, device=tokens.device)
    for slot in range(expert_ids.numel()):
        eid = expert_ids[slot].item()
        pos = int(offsets[eid].item() + write_pos[eid].item())
        permuted[pos] = tokens[token_indices[slot].item()]
        routing_w[pos] = gate_weights[slot]
        src_idx[pos] = token_indices[slot]
        write_pos[eid] += 1

    # Pass 3: Gather (identity expert — just scatter back)
    output = torch.zeros(num_tokens, hidden, dtype=tokens.dtype, device=tokens.device)
    for i in range(total):
        output[src_idx[i].item()] += routing_w[i] * permuted[i]

    return permuted, output


def vectorized_pytorch_routing(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_weights: torch.Tensor,
    token_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Vectorized PyTorch baseline using torch.index_select (still 3 global-memory passes).
    """
    num_tokens, hidden = tokens.shape
    output = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=tokens.device)

    for eid in range(num_experts):
        mask = expert_ids == eid
        if not mask.any():
            continue
        slots = mask.nonzero(as_tuple=True)[0]
        toks = token_indices[slots]
        weights = gate_weights[slots].unsqueeze(1)
        expert_in = tokens[toks]  # index_select (non-contiguous)
        # Identity expert — just weight and scatter back
        output.index_add_(0, toks, (expert_in.float() * weights))

    return output.to(tokens.dtype)


@dataclass
class ComparisonResult:
    method: str
    num_tokens: int
    hidden_size: int
    num_experts: int
    mean_ms: float
    median_ms: float
    std_ms: float
    throughput_tok_per_s: float


def _bench_method(
    name: str,
    fn,
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_weights: torch.Tensor,
    token_indices: torch.Tensor,
    num_experts: int,
    warmup: int,
    repeat: int,
) -> ComparisonResult:
    num_tokens = tokens.size(0)

    for _ in range(warmup):
        fn(tokens, expert_ids, gate_weights, token_indices, num_experts)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        fn(tokens, expert_ids, gate_weights, token_indices, num_experts)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    import statistics

    mean_ms = statistics.mean(times)
    return ComparisonResult(
        method=name,
        num_tokens=num_tokens,
        hidden_size=tokens.size(1),
        num_experts=num_experts,
        mean_ms=mean_ms,
        median_ms=statistics.median(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        throughput_tok_per_s=num_tokens / (mean_ms / 1000.0),
    )


def run_comparison(
    output_dir: str = "results/baseline",
    hidden_size: int = 4096,
    num_experts: int = 4,
    top_k: int = 2,
    warmup: int = 10,
    repeat: int = 50,
    device: str = "cuda",
) -> list[ComparisonResult]:
    if not torch.cuda.is_available():
        print("CUDA not available.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    from lightning_router.kernels.expert_routing import expert_routing_forward

    def triton_routing(tokens, expert_ids, gate_weights, token_indices, num_experts):
        return expert_routing_forward(
            tokens, expert_ids, gate_weights, token_indices, num_experts,
        )

    token_counts = [256, 1024, 4096, 8192, 16384]
    all_results: list[ComparisonResult] = []

    print(f"\n{'=' * 78}")
    print("  BASELINE COMPARISON — Routing Kernel Performance")
    print(f"  Hidden={hidden_size}  Experts={num_experts}  Top-K={top_k}")
    print(f"{'=' * 78}\n")

    for num_tokens in token_counts:
        tokens = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
        expert_ids = torch.randint(0, num_experts, (num_tokens * top_k,),
                                   dtype=torch.int32, device=device)
        gate_weights = torch.rand(num_tokens * top_k, dtype=torch.float32, device=device)
        gate_weights = gate_weights / gate_weights.view(num_tokens, top_k).sum(
            -1, keepdim=True
        ).repeat(1, top_k).view(-1)
        token_indices = (
            torch.arange(num_tokens, device=device)
            .unsqueeze(1).expand(-1, top_k).reshape(-1).int()
        )

        print(f"  num_tokens = {num_tokens:>6}")

        # Vectorized PyTorch baseline
        r_pytorch = _bench_method(
            "PyTorch (vectorized)", vectorized_pytorch_routing,
            tokens, expert_ids, gate_weights, token_indices, num_experts,
            warmup, repeat,
        )
        all_results.append(r_pytorch)
        print(f"    PyTorch:    {r_pytorch.mean_ms:>8.3f} ms  "
              f"({r_pytorch.throughput_tok_per_s:>12,.0f} tok/s)")

        # Triton fused
        r_triton = _bench_method(
            "Triton (fused)", triton_routing,
            tokens, expert_ids, gate_weights, token_indices, num_experts,
            warmup, repeat,
        )
        all_results.append(r_triton)
        speedup = r_pytorch.mean_ms / r_triton.mean_ms
        print(f"    Triton:     {r_triton.mean_ms:>8.3f} ms  "
              f"({r_triton.throughput_tok_per_s:>12,.0f} tok/s)  "
              f"[{speedup:.2f}x speedup]")
        print()

    out_path = Path(output_dir) / "baseline_comparison.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"  Results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline comparison benchmark")
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()
    run_comparison(
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        warmup=args.warmup,
        repeat=args.repeat,
    )
