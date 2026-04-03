# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LightningRouter is a GPU kernel optimization system for Mixture-of-Experts (MoE) LLM inference. It replaces the expert-routing bottleneck with fused Triton GPU kernels, 4-bit INT4 quantized GEMM, and integrates with both vLLM and SGLang for serving on 1-2 GPUs.

## Common Commands

```bash
# Install (editable, all extras)
pip install -e ".[dev,serving,profile]"

# Tests
pytest tests/ -v -m "not gpu"          # CPU-only tests
pytest tests/ -v -m gpu                # GPU tests (require CUDA + Triton)
pytest tests/test_quantization.py::TestPackWeights::test_pack_unpack_roundtrip -v  # single test

# Benchmarks
pytest benchmarks/ -v --benchmark-only  # pytest-benchmark suite

# Lint
ruff check lightning_router/ tests/ benchmarks/
mypy lightning_router/

# CLI
lightning-router bench --kernel all
lightning-router profile --kernel expert_routing
lightning-router serve --config configs/moe_4expert.yaml
```

## Architecture

The system is a four-stage pipeline per MoE forward pass:

1. **Gating** (`models/gating.py`) — TopKGating projects hidden states to expert logits, selects top-k experts per token, produces flat `expert_ids`, `gate_weights`, `token_indices` tensors, and an auxiliary load-balancing loss.

2. **Scatter** (`kernels/expert_routing.py`) — Triton kernels permute tokens into expert-contiguous layout. Three phases: histogram (shared-memory atomics), prefix-sum (CSR offsets), scatter (coalesced 128-byte writes). Output: `permuted_tokens`, `expert_offsets`, `routing_weights`, `source_indices`.

3. **Expert FFN** (`models/experts.py`) — SwiGLU FFN per expert (`down(SiLU(gate(x)) * up(x))`). In quantized mode, weight matrices are packed INT4 (`int32` with 8 nibbles each) and the forward pass uses `kernels/quantized_matmul.py` which dequantises on-the-fly with per-group scale+zero.

4. **Gather** (`kernels/expert_routing.py`) — Triton kernel scatters expert outputs back to original token positions, fusing multiply-by-gate-weight and atomic accumulation in one pass.

`models/moe_layer.py` orchestrates all four stages and has a pure-PyTorch fallback path (routing_implementation="torch") for CPU/validation.

### Quantization data flow

`quantization/pack_weights.py` converts `(K, N)` fp16 weights → packed `(K, N//8)` int32 + per-group scales/zeros. The Triton GEMM kernel (`quantized_matmul.py`) loads packed int32, extracts 4-bit values with bit-shifts, applies `scale * (raw - zero)`, and accumulates in fp32.

### Serving layer

`serving/server.py` dispatches to vLLM (`worker.py` → `model_runner.py`) or SGLang (`sglang_backend.py`) based on `config.serving.engine`. `TensorParallelGroup` in `worker.py` handles expert-parallel sharding across 2 GPUs with NCCL all-reduce.

### Configuration

All configs are YAML-based dataclasses in `config.py`. Six sections: `ModelConfig`, `MoEConfig`, `QuantizationConfig`, `KernelConfig`, `ServingConfig`, `ProfilingConfig`. Loaded via `load_config()` which maps YAML dicts to typed dataclasses.

## Key Conventions

- Triton kernels use `tl.constexpr` for tile sizes (BLOCK_M=128, BLOCK_N=64, BLOCK_K=32).
- INT4 packing: 8 consecutive 4-bit values per int32, extracted via `(packed >> (i * 4)) & 0xF`.
- Tests marked `@pytest.mark.gpu` require CUDA; unmarked tests run on CPU.
- Benchmark scripts in `benchmarks/` output JSON to `results/` and figures to `figures/`.
- Ruff config: line-length 100, rules E/F/W/I/N/UP, target Python 3.10.
