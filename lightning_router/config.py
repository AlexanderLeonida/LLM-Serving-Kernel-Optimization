"""Shared configuration dataclasses parsed from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


# ─── Sub-configs ──────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    name: str = "lightning-moe-4e"
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    num_hidden_layers: int = 32
    vocab_size: int = 32000
    max_position_embeddings: int = 4096


@dataclass
class MoEConfig:
    num_experts: int = 4
    num_experts_per_token: int = 2
    gating_type: str = "top_k_softmax"
    expert_capacity_factor: float = 1.25
    load_balance_loss_weight: float = 0.01
    routing_implementation: Literal["triton", "torch"] = "triton"


@dataclass
class QuantizationConfig:
    enabled: bool = True
    bits: int = 4
    group_size: int = 128
    scheme: Literal["symmetric", "asymmetric"] = "asymmetric"
    quantize_experts: bool = True
    quantize_attention: bool = True


@dataclass
class KernelConfig:
    block_size_m: int = 128
    block_size_n: int = 64
    block_size_k: int = 32
    num_warps: int = 4
    num_stages: int = 3
    use_shared_memory_cache: bool = True
    coalesced_access: bool = True


@dataclass
class ServingConfig:
    engine: str = "vllm"
    tensor_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.90
    swap_space_gb: int = 4
    enforce_eager: bool = False


@dataclass
class ProfilingConfig:
    enabled: bool = False
    nsight_output_dir: str = "profiling_results/"
    trace_expert_routing: bool = True
    trace_kernel_execution: bool = True
    warmup_iterations: int = 5
    profile_iterations: int = 20


# ─── Top-level config ────────────────────────────────────────────────────────


@dataclass
class LightningRouterConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)


def _dict_to_dataclass(cls, d: dict):
    """Recursively map dict → dataclass, ignoring unknown keys."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in field_names})


def load_config(path: str | Path) -> LightningRouterConfig:
    """Load a YAML config and return a typed ``LightningRouterConfig``."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    return LightningRouterConfig(
        model=_dict_to_dataclass(ModelConfig, raw.get("model", {})),
        moe=_dict_to_dataclass(MoEConfig, raw.get("moe", {})),
        quantization=_dict_to_dataclass(QuantizationConfig, raw.get("quantization", {})),
        kernel=_dict_to_dataclass(KernelConfig, raw.get("kernel", {})),
        serving=_dict_to_dataclass(ServingConfig, raw.get("serving", {})),
        profiling=_dict_to_dataclass(ProfilingConfig, raw.get("profiling", {})),
    )
