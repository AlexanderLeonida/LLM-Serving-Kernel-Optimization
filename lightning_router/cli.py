"""CLI entry-point for LightningRouter."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lightning-router",
        description="LightningRouter – MoE LLM inference with custom GPU kernels",
    )
    sub = parser.add_subparsers(dest="command")

    # ── serve ──────────────────────────────────────────────────────────
    serve_p = sub.add_parser("serve", help="Launch vLLM-backed inference server")
    serve_p.add_argument("--config", default="configs/moe_4expert.yaml")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)

    # ── bench ──────────────────────────────────────────────────────────
    bench_p = sub.add_parser("bench", help="Run kernel benchmarks")
    bench_p.add_argument("--kernel", choices=["routing", "quantized_matmul", "all"], default="all")
    bench_p.add_argument("--batch-size", type=int, default=32)
    bench_p.add_argument("--num-experts", type=int, default=4)

    # ── profile ────────────────────────────────────────────────────────
    prof_p = sub.add_parser("profile", help="Run Nsight profiling")
    prof_p.add_argument("--kernel", default="expert_routing")
    prof_p.add_argument("--output-dir", default="profiling_results/")

    args = parser.parse_args(argv)

    if args.command == "serve":
        from lightning_router.serving.server import launch_server

        launch_server(args.config, args.host, args.port)
    elif args.command == "bench":
        from lightning_router.benchmarks.kernel_bench import run_benchmarks

        run_benchmarks(args.kernel, args.batch_size, args.num_experts)
    elif args.command == "profile":
        from lightning_router.profiling.nsight_runner import run_profiling

        run_profiling(args.kernel, args.output_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
