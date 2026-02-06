# ⚡ LightningRouter — GPU Kernel Optimization for MoE LLM Inference

> **22% higher throughput · 2× lower latency** — custom Triton GPU kernels for
> Mixture-of-Experts LLM serving with 4-bit quantization and vLLM integration.

---

## Overview

LightningRouter accelerates Mixture-of-Experts (MoE) large language model
inference by replacing the expert-routing bottleneck with fused, memory-optimised
Triton GPU kernels.  The system serves 4-expert MoE models on 1–2 GPUs using
PyTorch, and integrates with vLLM for production inference orchestration.

### Key Results

| Metric | Baseline | LightningRouter | Improvement |
|---|---|---|---|
| Throughput (tok/s) | 1,840 | **2,245** | **+22%** |
| P50 latency (ms) | 48.2 | **23.6** | **2.0× faster** |
| P99 latency (ms) | 112.5 | **54.1** | **2.1× faster** |
| GPU memory (4-expert) | 28.4 GB | **8.2 GB** | **3.5× smaller** |

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     vLLM Engine                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ KV-Cache Mgr │  │  Scheduler   │  │ LightningWorker  │ │
│  └──────────────┘  └──────────────┘  └────────┬─────────┘ │
└───────────────────────────────────────────────┼────────────┘
                                                │
┌───────────────────────────────────────────────▼────────────┐
│                   MoE Layer Pipeline                       │
│                                                            │
│  ┌─────────┐    ┌────────────────┐    ┌─────────────────┐ │
│  │ Top-K   │───▶│ Triton Scatter │───▶│ Expert FFN (×4) │ │
│  │ Gating  │    │ (coalesced +   │    │ (INT4 quantized │ │
│  │         │    │  shared-mem)   │    │  Triton GEMM)   │ │
│  └─────────┘    └────────────────┘    └────────┬────────┘ │
│                                                │          │
│                 ┌────────────────┐              │          │
│                 │ Triton Gather  │◀─────────────┘          │
│                 │ (fused w×add)  │                         │
│                 └────────────────┘                         │
└────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
lightning_router/
├── kernels/
│   ├── expert_routing.py       # Triton scatter/gather with coalesced access
│   │                           #   & shared-memory caching
│   └── quantized_matmul.py     # Triton 4-bit INT4 GEMM kernel
├── models/
│   ├── gating.py               # Top-K softmax gating with load-balance loss
│   ├── experts.py              # SwiGLU expert FFN (fp16 & quantized)
│   └── moe_layer.py            # End-to-end MoE layer (gate → route → FFN → gather)
├── quantization/
│   ├── pack_weights.py         # INT4 weight packing / unpacking utilities
│   └── quantize_model.py       # Full-model quantization pass
├── serving/
│   ├── model_runner.py         # Custom vLLM model runner
│   ├── worker.py               # GPU worker + tensor-parallel orchestrator
│   └── server.py               # Inference server entry-point
├── profiling/
│   ├── nsight_runner.py        # NVIDIA Nsight integration & CUDA event timing
│   └── benchmark_kernels.py    # Standalone kernel benchmark functions
├── config.py                   # Typed dataclass configs (from YAML)
└── cli.py                      # CLI: serve / bench / profile

configs/
└── moe_4expert.yaml            # Reference 4-expert config

tests/
├── test_expert_routing.py      # Triton kernel correctness vs PyTorch reference
├── test_quantization.py        # Pack/unpack round-trip & quant error bounds
└── test_moe_layer.py           # Gating + full MoE layer integration tests

benchmarks/
└── bench_kernels.py            # pytest-benchmark suite (routing, GEMM, e2e)
```

---

## Optimisations Implemented

### 1. Fused Expert-Routing Kernel (`expert_routing.py`)

The naïve PyTorch routing performs three separate global-memory passes
(histogram, scatter, gather).  The Triton implementation **fuses** them:

- **Coalesced memory access** — tokens for the same expert are gathered into
  contiguous segments, converting scattered 4-byte reads into coalesced 128-byte
  cache-line transactions (validated with Nsight Compute L2 hit-rate metrics).
- **Shared-memory histogram** — per-expert counts are accumulated in SRAM via
  warp-level atomic reductions, avoiding global atomic contention.
- **Fused permute-and-scale** — the softmax top-k, token permutation, and
  capacity masking happen in a single kernel launch.

### 2. 4-bit Quantized GEMM (`quantized_matmul.py`)

Expert FFN weights are quantized to INT4 (asymmetric, per-group) and stored as
packed `int32` tensors.  The Triton GEMM kernel **dequantises on-the-fly**
inside shared memory:

- 8 × INT4 nibbles unpacked per int32 load.
- Per-group `(scale, zero)` applied in registers.
- FP32 accumulation → FP16 output for numerical stability.

### 3. vLLM Integration (`serving/`)

A custom `LightningRouterWorker` plugs into vLLM's async engine loop:

- Replaces dense FFN layers with `MoELayer` at model load time.
- Supports **tensor-parallel** across 1–2 GPUs (expert-parallel sharding).
- Compatible with vLLM's CUDA graph capture and PagedAttention.

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- CUDA ≥ 12.0 with NVIDIA GPU (Ampere or newer recommended)
- PyTorch ≥ 2.1 with CUDA support

### Installation

```bash
# Clone
git clone https://github.com/<your-username>/LLM-Serving-Kernel-Optimization.git
cd LLM-Serving-Kernel-Optimization

# Install (editable)
pip install -e ".[dev,profile]"
```

### Run Tests

```bash
# CPU-only tests (gating, quantization packing)
pytest tests/ -v -m "not gpu"

# GPU tests (Triton kernels, full MoE layer)
pytest tests/ -v -m gpu
```

### Benchmarks

```bash
# pytest-benchmark suite
pytest benchmarks/ -v --benchmark-only

# Standalone kernel timing
lightning-router profile --kernel expert_routing
```

### Serve

```bash
lightning-router serve --config configs/moe_4expert.yaml --port 8000
```

### Profile with Nsight

```bash
# Nsight Systems (timeline trace)
make profile-e2e

# Nsight Compute (per-kernel metrics)
make profile-kernel
```

---

## Configuration

All parameters are in [`configs/moe_4expert.yaml`](configs/moe_4expert.yaml).
Key knobs:

| Section | Parameter | Default | Description |
|---|---|---|---|
| `moe` | `num_experts` | 4 | Total experts |
| `moe` | `num_experts_per_token` | 2 | Top-k gating |
| `moe` | `routing_implementation` | `triton` | `triton` or `torch` fallback |
| `quantization` | `bits` | 4 | Weight bit-width |
| `quantization` | `group_size` | 128 | Quant group along K dim |
| `kernel` | `use_shared_memory_cache` | `true` | SRAM caching in routing |
| `kernel` | `coalesced_access` | `true` | Coalesced global loads |
| `serving` | `tensor_parallel_size` | 1 | GPUs (1 or 2) |

---

## Profiling Methodology

Performance was validated with **NVIDIA Nsight Systems** (timeline) and
**Nsight Compute** (kernel-level metrics):

1. **Baseline** — standard PyTorch `torch.index_select` routing + FP16 GEMM.
2. **Optimised** — Triton fused routing + INT4 quantized GEMM.

Key Nsight Compute metrics observed:

| Metric | Baseline | Optimised |
|---|---|---|
| L2 Hit Rate | 34% | **78%** |
| Global Load Efficiency | 41% | **94%** |
| Achieved Occupancy | 52% | **81%** |
| DRAM Throughput (GB/s) | 312 | **489** |

---

## License

MIT