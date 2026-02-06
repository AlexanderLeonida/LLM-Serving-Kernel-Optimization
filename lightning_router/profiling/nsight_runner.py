"""
NVIDIA Nsight Systems / Nsight Compute profiling runner.

Provides helpers to:
1. Wrap kernel launches with NVTX ranges so they show up as named regions
   in Nsight traces.
2. Drive ``nsys`` / ``ncu`` from Python for automated profiling sweeps.
3. Collect and summarise key metrics (kernel duration, memory throughput,
   occupancy, L2 hit rate).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# NVTX annotation helpers
# ──────────────────────────────────────────────────────────────────────────────


def nvtx_range(name: str):
    """Context manager that pushes/pops an NVTX range (visible in Nsight)."""
    import contextlib

    @contextlib.contextmanager
    def _range():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    return _range()


# ──────────────────────────────────────────────────────────────────────────────
# Kernel-level timer
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class KernelTimingResult:
    kernel_name: str
    warmup_iters: int
    profile_iters: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    all_ms: list[float] = field(default_factory=list)


def time_kernel(
    fn,
    *args,
    warmup: int = 10,
    repeat: int = 100,
    label: str = "kernel",
    **kwargs,
) -> KernelTimingResult:
    """
    Precisely time a GPU kernel using CUDA events.

    Parameters
    ----------
    fn : callable – the kernel wrapper to time.
    warmup : number of warm-up iterations (not timed).
    repeat : number of timed iterations.
    label : human-readable name for reporting.

    Returns
    -------
    KernelTimingResult with ms-level statistics.
    """
    # Warm up
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        fn(*args, **kwargs)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    import statistics

    return KernelTimingResult(
        kernel_name=label,
        warmup_iters=warmup,
        profile_iters=repeat,
        mean_ms=statistics.mean(times),
        median_ms=statistics.median(times),
        min_ms=min(times),
        max_ms=max(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        all_ms=times,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Nsight CLI drivers
# ──────────────────────────────────────────────────────────────────────────────


def run_nsys_profile(
    script_path: str,
    output_dir: str = "profiling_results",
    extra_args: list[str] | None = None,
) -> Path:
    """
    Launch ``nsys profile`` on a Python script and return the report path.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_name = Path(output_dir) / f"nsys_{int(time.time())}"

    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx,osrt",
        "--force-overwrite=true",
        f"--output={report_name}",
        "python", script_path,
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return report_name.with_suffix(".nsys-rep")


def run_ncu_profile(
    script_path: str,
    kernel_name: str = "expert_routing",
    output_dir: str = "profiling_results",
) -> Path:
    """
    Launch ``ncu`` (Nsight Compute) on a specific kernel for detailed metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    report = Path(output_dir) / f"ncu_{kernel_name}_{int(time.time())}.ncu-rep"

    cmd = [
        "ncu",
        "--set=full",
        f"--kernel-name={kernel_name}",
        "--launch-count=5",
        f"--export={report}",
        "python", script_path,
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point used by CLI / Makefile
# ──────────────────────────────────────────────────────────────────────────────


def run_profiling(kernel: str = "expert_routing", output_dir: str = "profiling_results") -> None:
    """Profile a kernel using CUDA events and print a summary."""
    import statistics

    from lightning_router.profiling.benchmark_kernels import get_routing_benchmark_fn

    logger.info("Profiling kernel: %s", kernel)
    fn, args = get_routing_benchmark_fn(batch_size=32, num_experts=4)
    result = time_kernel(fn, *args, warmup=10, repeat=100, label=kernel)

    print(f"\n{'─' * 60}")
    print(f"  Kernel: {result.kernel_name}")
    print(f"  Mean:   {result.mean_ms:.3f} ms")
    print(f"  Median: {result.median_ms:.3f} ms")
    print(f"  Min:    {result.min_ms:.3f} ms")
    print(f"  Max:    {result.max_ms:.3f} ms")
    print(f"  Std:    {result.std_ms:.3f} ms")
    print(f"{'─' * 60}\n")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{kernel}_timing.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "kernel": result.kernel_name,
                "mean_ms": result.mean_ms,
                "median_ms": result.median_ms,
                "min_ms": result.min_ms,
                "max_ms": result.max_ms,
                "std_ms": result.std_ms,
            },
            f,
            indent=2,
        )
    print(f"Saved to {out_path}")
