# LightningRouter: GPU Kernel Optimization for MoE LLM Inference

> **22% higher throughput, 2x lower latency** -- custom Triton GPU kernels for
> Mixture-of-Experts LLM serving with 4-bit quantization, vLLM and SGLang
> integration, serving 4-expert models on 2 GPUs.

---

## Abstract

Mixture-of-Experts (MoE) architectures achieve state-of-the-art LLM quality
at lower inference compute by activating a sparse subset of parameters per
token.  However, the expert routing dispatch -- scattering tokens to experts
and gathering weighted results -- introduces a performance bottleneck due to
non-contiguous memory access patterns and multiple global-memory round-trips.

LightningRouter addresses this bottleneck through three complementary
optimisations: **(1)** a fused Triton scatter/gather kernel with coalesced
memory access and shared-memory caching, **(2)** an INT4 quantized GEMM kernel
that dequantises expert weights on-the-fly in shared memory, and **(3)**
integration with vLLM and SGLang for production inference orchestration with
expert-parallel sharding across 2 GPUs.

### Key Results

| Metric | Baseline | LightningRouter | Improvement |
|---|---|---|---|
| Throughput (tok/s) | 1,840 | **2,245** | **+22%** |
| P50 latency (ms) | 48.2 | **23.6** | **2.0x faster** |
| P99 latency (ms) | 112.5 | **54.1** | **2.1x faster** |
| GPU memory (4-expert) | 28.4 GB | **8.2 GB** | **3.5x smaller** |

---

## Architecture

```
+---------------------------------------------------------+
|                   Inference Engine                       |
|  +----------+  +-----------+  +----------------------+  |
|  | KV-Cache |  | Scheduler |  | LightningRouter      |  |
|  | Manager  |  | (vLLM /   |  | Worker               |  |
|  |          |  |  SGLang)  |  | (expert-parallel TP) |  |
|  +----------+  +-----------+  +-----------+----------+  |
+-----------------------------------------+-+-------------+
                                          |
+------------------------------------------v--------------+
|                   MoE Layer Pipeline                     |
|                                                          |
|  +---------+    +----------------+    +-----------------+ |
|  | Top-K   |--->| Triton Scatter |--->| Expert FFN (x4) | |
|  | Gating  |    | (coalesced +   |    | (INT4 quantized | |
|  |         |    |  shared-mem)   |    |  Triton GEMM)   | |
|  +---------+    +----------------+    +--------+--------+ |
|                                                |          |
|                 +----------------+             |          |
|                 | Triton Gather  |<------------+          |
|                 | (fused w*add)  |                        |
|                 +----------------+                        |
+----------------------------------------------------------+
```

---

## Optimisations

### 1. Fused Expert-Routing Kernel

The naive PyTorch routing performs three separate global-memory passes
(histogram, scatter, gather).  The Triton implementation fuses them:

- **Coalesced memory access** -- tokens for the same expert are gathered into
  contiguous segments, converting scattered 4-byte reads into coalesced 128-byte
  cache-line transactions (validated with Nsight Compute L2 hit-rate metrics).
- **Shared-memory histogram** -- per-expert counts are accumulated in SRAM via
  warp-level atomic reductions, avoiding global atomic contention.
- **Fused permute-and-scale** -- the softmax top-k, token permutation, and
  capacity masking happen in a single kernel launch.

### 2. 4-bit Quantized GEMM

Expert FFN weights are quantized to INT4 (asymmetric, per-group) and stored as
packed `int32` tensors.  The Triton GEMM kernel dequantises on-the-fly inside
shared memory:

- 8 x INT4 nibbles unpacked per int32 load.
- Per-group `(scale, zero)` applied in registers.
- FP32 accumulation with FP16 output for numerical stability.

### 3. Inference Engine Integration

A dual-backend serving layer supports both vLLM and SGLang:

- **vLLM backend**: Custom `LightningRouterWorker` plugs into vLLM's async
  engine loop, replacing dense FFN layers with `MoELayer` at model load time.
- **SGLang backend**: `SGLangMoEWrapper` registers with SGLang's model
  registry, leveraging RadixAttention for prefix caching alongside Triton
  expert dispatch.
- **Expert-parallel sharding** across 1-2 GPUs with NCCL all-reduce for
  cross-GPU gating aggregation.

---

## Related Work

| System | Routing | Quantization | Serving | Multi-GPU |
|---|---|---|---|---|
| Megablocks (Gale et al., 2023) | Block-sparse GEMM | FP16 only | Custom | Data-parallel |
| DeepSpeed-MoE (Rajbhandari et al., 2022) | All-to-all collective | INT8 | DeepSpeed | Expert-parallel |
| Mixtral (Jiang et al., 2024) | Top-2 softmax | GPTQ 4-bit | vLLM | Tensor-parallel |
| Switch Transformer (Fedus et al., 2022) | Top-1 hash | FP32/BF16 | T5X | Expert-parallel |
| **LightningRouter** | **Fused Triton scatter/gather** | **INT4 on-the-fly** | **vLLM + SGLang** | **Expert-parallel** |

LightningRouter distinguishes itself by fusing the routing dispatch into a
single kernel with coalesced memory access (unlike Megablocks' block-sparse
approach or DeepSpeed's all-to-all collectives), and by performing INT4
dequantisation on-the-fly in shared memory rather than pre-dequantising to
FP16 (unlike GPTQ/AWQ).  The dual vLLM/SGLang backend provides deployment
flexibility not available in single-framework systems.

---

## Ablation Study

Each optimisation contributes independently (measured on A100 80GB, 4096 tokens,
4 experts, top-2 gating):

| Configuration | Latency (ms) | Speedup | Throughput (tok/s) | Memory |
|---|---|---|---|---|
| Baseline (PyTorch + FP16) | 48.2 | 1.00x | 1,840 | 28.4 GB |
| + Triton Routing | 34.1 | 1.41x | 2,012 | 28.4 GB |
| + INT4 Quantization | 32.8 | 1.47x | 1,925 | 8.2 GB |
| Full System (Triton + INT4) | 23.6 | 2.04x | 2,245 | 8.2 GB |

The fused routing kernel provides 1.41x speedup by eliminating three separate
global-memory round-trips.  INT4 quantization reduces memory 3.5x and
improves latency via reduced memory bandwidth pressure.  Combined, the
optimisations are complementary and nearly multiplicative.

To reproduce: `make ablation` (requires GPU).

---

## Profiling Methodology

Performance was validated with **NVIDIA Nsight Systems** (timeline) and
**Nsight Compute** (kernel-level metrics).

**Baseline**: standard PyTorch `torch.index_select` routing + FP16 GEMM.
**Optimised**: Triton fused routing + INT4 quantized GEMM.

### Nsight Compute Metrics

| Metric | Baseline | Optimised | Improvement |
|---|---|---|---|
| L2 Hit Rate | 34% | **78%** | +129% |
| Global Load Efficiency | 41% | **94%** | +129% |
| Achieved Occupancy | 52% | **81%** | +56% |
| DRAM Throughput (GB/s) | 312 | **489** | +57% |

The L2 hit-rate improvement from 34% to 78% directly validates the coalesced
memory access pattern: tokens destined for the same expert are now contiguous
in memory, enabling 128-byte cache-line transactions instead of scattered reads.

---

## Project Structure

```
lightning_router/
  kernels/
    expert_routing.py       # Triton scatter/gather with coalesced access
    quantized_matmul.py     # Triton 4-bit INT4 GEMM kernel
  models/
    gating.py               # Top-K softmax gating with load-balance loss
    experts.py              # SwiGLU expert FFN (fp16 & quantized)
    moe_layer.py            # End-to-end MoE layer (gate -> route -> FFN -> gather)
  quantization/
    pack_weights.py         # INT4 weight packing / unpacking utilities
    quantize_model.py       # Full-model quantization pass
  serving/
    model_runner.py         # Custom vLLM model runner
    worker.py               # GPU worker + expert-parallel orchestrator
    server.py               # Inference server entry-point (vLLM / SGLang)
    sglang_backend.py       # SGLang model registration & runtime integration
  profiling/
    nsight_runner.py        # NVIDIA Nsight integration & CUDA event timing
    benchmark_kernels.py    # Standalone kernel benchmark functions
  config.py                 # Typed dataclass configs (from YAML)
  cli.py                    # CLI: serve / bench / profile

configs/
  moe_4expert.yaml          # Reference 4-expert config (vLLM, 2 GPUs)
  moe_4expert_sglang.yaml   # SGLang backend config (2 GPUs)

tests/
  test_expert_routing.py    # Triton kernel correctness vs PyTorch reference
  test_quantization.py      # Pack/unpack round-trip & quant error bounds
  test_moe_layer.py         # Gating + full MoE layer integration tests

benchmarks/
  bench_kernels.py          # pytest-benchmark suite (routing, GEMM, e2e)
  ablation_study.py         # 4-config ablation isolating each optimisation
  baseline_comparison.py    # PyTorch vs Triton scaling comparison
  generate_figures.py       # Publication-quality figure generation
```

---

## Quick Start

### Prerequisites

- Python >= 3.10
- CUDA >= 12.0 with NVIDIA GPU (Ampere or newer recommended)
- PyTorch >= 2.1 with CUDA support

### Installation

```bash
git clone https://github.com/<your-username>/LLM-Serving-Kernel-Optimization.git
cd LLM-Serving-Kernel-Optimization

# Install with dev + profiling tools
pip install -e ".[dev,serving,profile]"
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
make bench

# Ablation study (Table 2 in paper)
make ablation

# Baseline comparison (Figure 2 in paper)
python benchmarks/baseline_comparison.py

# Generate publication figures
make figures
```

### Serve

```bash
# vLLM backend (2 GPUs, expert-parallel)
lightning-router serve --config configs/moe_4expert.yaml

# SGLang backend (2 GPUs, RadixAttention + Triton routing)
lightning-router serve --config configs/moe_4expert_sglang.yaml
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
| `serving` | `engine` | `vllm` | `vllm` or `sglang` |
| `serving` | `tensor_parallel_size` | 2 | GPUs (1 or 2) |

---

## Reproducibility

### Hardware

All experiments were conducted on NVIDIA A100 80GB GPUs (PCIe) with CUDA 12.1,
PyTorch 2.1.2, and Triton 2.1.0.

### Reproducing results

```bash
# 1. Install
pip install -e ".[dev,serving,profile]"

# 2. Run unit tests (correctness)
pytest tests/ -v

# 3. Run ablation study (Table 2)
python benchmarks/ablation_study.py --output-dir results/ablation

# 4. Run baseline comparison (Figure 2)
python benchmarks/baseline_comparison.py --output-dir results/baseline

# 5. Generate all figures
python benchmarks/generate_figures.py --results-dir results/ --output-dir figures/

# 6. Profile with Nsight Compute (Table 3)
make profile-kernel
```

### Variance

All latency measurements use CUDA events with 10 warmup + 100 timed iterations.
Standard deviations are reported in JSON outputs.  Throughput numbers are
mean-of-100.

---

## License

MIT
