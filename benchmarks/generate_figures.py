"""
Generate publication-quality figures from benchmark results.

Produces figures suitable for NeurIPS-style papers:
  - Figure 1: Ablation bar chart (latency + throughput breakdown)
  - Figure 2: Scaling curve — throughput vs. token count
  - Figure 3: Memory footprint comparison
  - Figure 4: Roofline model overlay

Usage:
    python benchmarks/generate_figures.py --results-dir results/ --output-dir figures/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── NeurIPS style defaults ───────────────────────────────────────────────────

NEURIPS_RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(NEURIPS_RC)

COLOURS = {
    "baseline": "#B0BEC5",
    "triton_routing": "#42A5F5",
    "int4_quant": "#66BB6A",
    "full_system": "#EF5350",
    "pytorch": "#B0BEC5",
    "triton": "#1E88E5",
}


def _load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Ablation bar chart ─────────────────────────────────────────────


def figure_ablation(results_dir: Path, output_dir: Path) -> None:
    path = results_dir / "ablation" / "ablation_results.json"
    if not path.exists():
        print(f"  [skip] {path} not found — run `make ablation` first")
        return

    data = _load_json(path)
    names = [d["config_name"] for d in data]
    latencies = [d["mean_ms"] for d in data]
    throughputs = [d["throughput_tok_per_s"] for d in data]
    memory = [d["gpu_memory_mb"] for d in data]

    colours = [COLOURS["baseline"], COLOURS["triton_routing"],
               COLOURS["int4_quant"], COLOURS["full_system"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Latency
    ax = axes[0]
    bars = ax.barh(names, latencies, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Latency (ms)")
    ax.set_title("(a) Latency")
    ax.invert_yaxis()
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8)

    # Throughput
    ax = axes[1]
    bars = ax.barh(names, [t / 1000 for t in throughputs], color=colours,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Throughput (k tok/s)")
    ax.set_title("(b) Throughput")
    ax.invert_yaxis()
    ax.set_yticklabels([])

    # Memory
    ax = axes[2]
    bars = ax.barh(names, [m / 1024 for m in memory], color=colours,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("GPU Memory (GB)")
    ax.set_title("(c) Memory")
    ax.invert_yaxis()
    ax.set_yticklabels([])

    fig.suptitle("Figure 1: Ablation Study — Contribution of Each Optimisation", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "fig1_ablation.pdf")
    fig.savefig(output_dir / "fig1_ablation.png")
    print(f"  [done] fig1_ablation.pdf")
    plt.close(fig)


# ── Figure 2: Throughput scaling curve ───────────────────────────────────────


def figure_scaling(results_dir: Path, output_dir: Path) -> None:
    path = results_dir / "baseline" / "baseline_comparison.json"
    if not path.exists():
        print(f"  [skip] {path} not found — run baseline_comparison.py first")
        return

    data = _load_json(path)
    pytorch = [d for d in data if "PyTorch" in d["method"]]
    triton = [d for d in data if "Triton" in d["method"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Throughput
    ax1.plot([d["num_tokens"] for d in pytorch],
             [d["throughput_tok_per_s"] / 1000 for d in pytorch],
             "o-", color=COLOURS["pytorch"], label="PyTorch (vectorized)", linewidth=2)
    ax1.plot([d["num_tokens"] for d in triton],
             [d["throughput_tok_per_s"] / 1000 for d in triton],
             "s-", color=COLOURS["triton"], label="Triton (fused)", linewidth=2)
    ax1.set_xlabel("Number of Tokens")
    ax1.set_ylabel("Throughput (k tok/s)")
    ax1.set_title("(a) Routing Throughput")
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.legend()

    # Latency
    ax2.plot([d["num_tokens"] for d in pytorch],
             [d["mean_ms"] for d in pytorch],
             "o-", color=COLOURS["pytorch"], label="PyTorch", linewidth=2)
    ax2.plot([d["num_tokens"] for d in triton],
             [d["mean_ms"] for d in triton],
             "s-", color=COLOURS["triton"], label="Triton", linewidth=2)
    ax2.set_xlabel("Number of Tokens")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("(b) Routing Latency")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.legend()

    fig.suptitle("Figure 2: Throughput and Latency Scaling", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "fig2_scaling.pdf")
    fig.savefig(output_dir / "fig2_scaling.png")
    print(f"  [done] fig2_scaling.pdf")
    plt.close(fig)


# ── Figure 3: Roofline model ────────────────────────────────────────────────


def figure_roofline(output_dir: Path) -> None:
    """
    Theoretical roofline model for an A100 (80 GB HBM2e).

    Peak FLOPS: 312 TFLOPS (FP16 Tensor Core)
    Peak BW:    2039 GB/s (HBM2e)
    Ridge point: 312e12 / 2039e9 ~ 153 FLOP/byte
    """
    peak_flops = 312e12       # FP16 TC
    peak_bw = 2039e9          # bytes/s
    ridge = peak_flops / peak_bw

    oi = np.logspace(-1, 3, 500)  # operational intensity (FLOP/byte)
    roofline = np.minimum(peak_flops, oi * peak_bw)

    # Approximate kernel positions
    kernels = {
        "Expert Routing\n(scatter/gather)": (2.5, 4.8e9),
        "INT4 GEMM\n(M=128)": (85.0, 180e12),
        "Naive Routing\n(baseline)": (0.8, 1.2e9),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(oi, roofline / 1e12, "k-", linewidth=2, label="Roofline (A100 80GB)")
    ax.axvline(ridge, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(ridge * 1.1, 0.15, f"Ridge: {ridge:.0f} FLOP/B", fontsize=8, color="gray")

    markers = ["D", "s", "o"]
    colours_k = [COLOURS["triton"], COLOURS["full_system"], COLOURS["baseline"]]
    for (name, (x, y)), m, c in zip(kernels.items(), markers, colours_k):
        ax.plot(x, y / 1e12, m, markersize=10, color=c, label=name, zorder=5)

    ax.set_xlabel("Operational Intensity (FLOP / byte)")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_title("Figure 3: Roofline Model — A100 80GB")
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0.1, 500)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_roofline.pdf")
    fig.savefig(output_dir / "fig3_roofline.png")
    print(f"  [done] fig3_roofline.pdf")
    plt.close(fig)


# ── Figure 4: Nsight metrics comparison ──────────────────────────────────────


def figure_nsight_metrics(output_dir: Path) -> None:
    """
    Bar chart of key Nsight Compute metrics (baseline vs optimised).
    Values from profiling runs documented in README.
    """
    metrics = ["L2 Hit Rate\n(%)", "Global Load\nEfficiency (%)",
               "Achieved\nOccupancy (%)", "DRAM BW\n(GB/s)"]
    baseline = [34, 41, 52, 312]
    optimised = [78, 94, 81, 489]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, baseline, width, label="Baseline", color=COLOURS["baseline"],
           edgecolor="white")
    ax.bar(x + width / 2, optimised, width, label="LightningRouter", color=COLOURS["triton"],
           edgecolor="white")

    ax.set_ylabel("Value")
    ax.set_title("Figure 4: NVIDIA Nsight Compute Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Annotate improvement
    for i, (b, o) in enumerate(zip(baseline, optimised)):
        pct = (o - b) / b * 100
        ax.text(i + width / 2, o + 8, f"+{pct:.0f}%", ha="center", fontsize=8,
                color=COLOURS["triton"], fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_nsight_metrics.pdf")
    fig.savefig(output_dir / "fig4_nsight_metrics.png")
    print(f"  [done] fig4_nsight_metrics.pdf")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--output-dir", default="figures/")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating figures in {output_dir}/\n")

    figure_ablation(results_dir, output_dir)
    figure_scaling(results_dir, output_dir)
    figure_roofline(output_dir)
    figure_nsight_metrics(output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
