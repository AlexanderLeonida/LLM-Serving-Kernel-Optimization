"""
Ablation study: isolate the contribution of each optimisation.

Measures throughput and latency under four configurations:
  1. **Baseline**       – PyTorch routing  +  FP16 experts
  2. **+Triton routing** – Triton scatter/gather  +  FP16 experts
  3. **+INT4 quant**     – PyTorch routing  +  INT4 quantized experts
  4. **Full system**     – Triton routing  +  INT4 quantized experts

Output: JSON results + console table suitable for inclusion in a paper.

Usage:
    python benchmarks/ablation_study.py --output-dir results/ablation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import torch


@dataclass
class AblationResult:
    config_name: str
    triton_routing: bool
    quantized: bool
    num_tokens: int
    num_experts: int
    mean_ms: float
    median_ms: float
    std_ms: float
    throughput_tok_per_s: float
    gpu_memory_mb: float


def _measure_config(
    *,
    config_name: str,
    routing_impl: str,
    quantized: bool,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    warmup: int,
    repeat: int,
    device: str,
) -> AblationResult:
    """Benchmark a single configuration and return results."""
    from lightning_router.config import ModelConfig, MoEConfig, QuantizationConfig, KernelConfig
    from lightning_router.models.moe_layer import MoELayer

    mcfg = ModelConfig(hidden_size=hidden_size, intermediate_size=intermediate_size)
    moecfg = MoEConfig(
        num_experts=num_experts,
        num_experts_per_token=top_k,
        routing_implementation=routing_impl,
    )
    qcfg = QuantizationConfig(enabled=quantized, quantize_experts=quantized)
    kcfg = KernelConfig()

    layer = MoELayer(mcfg, moecfg, qcfg, kcfg).to(device).half().eval()
    x = torch.randn(1, num_tokens, hidden_size, dtype=torch.float16, device=device)

    # Warm up
    with torch.inference_mode():
        for _ in range(warmup):
            layer(x)
    torch.cuda.synchronize()

    # Measure GPU memory
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        layer(x)
    torch.cuda.synchronize()
    gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    with torch.inference_mode():
        for i in range(repeat):
            start_events[i].record()
            layer(x)
            end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    import statistics

    mean_ms = statistics.mean(times)
    median_ms = statistics.median(times)
    std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
    throughput = num_tokens / (mean_ms / 1000.0)

    # Clean up
    del layer, x
    torch.cuda.empty_cache()

    return AblationResult(
        config_name=config_name,
        triton_routing=routing_impl == "triton",
        quantized=quantized,
        num_tokens=num_tokens,
        num_experts=num_experts,
        mean_ms=mean_ms,
        median_ms=median_ms,
        std_ms=std_ms,
        throughput_tok_per_s=throughput,
        gpu_memory_mb=gpu_mem_mb,
    )


ABLATION_CONFIGS = [
    {"config_name": "Baseline (PyTorch + FP16)", "routing_impl": "torch", "quantized": False},
    {"config_name": "+Triton Routing", "routing_impl": "triton", "quantized": False},
    {"config_name": "+INT4 Quantization", "routing_impl": "torch", "quantized": True},
    {"config_name": "Full System (Triton + INT4)", "routing_impl": "triton", "quantized": True},
]


def run_ablation(
    output_dir: str = "results/ablation",
    num_tokens: int = 4096,
    hidden_size: int = 4096,
    intermediate_size: int = 11008,
    num_experts: int = 4,
    top_k: int = 2,
    warmup: int = 10,
    repeat: int = 100,
    device: str = "cuda",
) -> list[AblationResult]:
    """Run the full ablation study and save results."""
    if not torch.cuda.is_available():
        print("CUDA not available — skipping ablation study.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    results: list[AblationResult] = []

    print(f"\n{'=' * 78}")
    print("  ABLATION STUDY — LightningRouter Optimisation Breakdown")
    print(f"  Tokens={num_tokens}  Experts={num_experts}  Top-K={top_k}  "
          f"Hidden={hidden_size}")
    print(f"{'=' * 78}\n")

    for cfg in ABLATION_CONFIGS:
        print(f"  Benchmarking: {cfg['config_name']} ...", end="", flush=True)
        result = _measure_config(
            config_name=cfg["config_name"],
            routing_impl=cfg["routing_impl"],
            quantized=cfg["quantized"],
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            warmup=warmup,
            repeat=repeat,
            device=device,
        )
        results.append(result)
        print(f"  {result.mean_ms:.2f} ms  ({result.throughput_tok_per_s:.0f} tok/s)")

    # Print table
    baseline = results[0]
    print(f"\n{'─' * 78}")
    print(f"  {'Configuration':<35} {'Latency':>10} {'Speedup':>10} "
          f"{'Throughput':>12} {'Memory':>10}")
    print(f"  {'':35} {'(ms)':>10} {'':>10} {'(tok/s)':>12} {'(MB)':>10}")
    print(f"{'─' * 78}")
    for r in results:
        speedup = baseline.mean_ms / r.mean_ms
        print(f"  {r.config_name:<35} {r.mean_ms:>10.2f} {speedup:>9.2f}x "
              f"{r.throughput_tok_per_s:>12,.0f} {r.gpu_memory_mb:>10.1f}")
    print(f"{'─' * 78}\n")

    # Save JSON
    out_path = Path(output_dir) / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"  Results saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightningRouter ablation study")
    parser.add_argument("--output-dir", default="results/ablation")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    run_ablation(
        output_dir=args.output_dir,
        num_tokens=args.num_tokens,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        warmup=args.warmup,
        repeat=args.repeat,
    )
