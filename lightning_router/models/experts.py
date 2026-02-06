"""
Expert Feed-Forward Network (FFN) with optional 4-bit quantized weights.

Each expert is a standard SwiGLU FFN:
    out = W_down · (SiLU(W_gate · x) ⊙ (W_up · x))

When quantization is enabled, W_gate, W_up, and W_down are stored as packed
INT4 tensors with per-group scales/zeros, and the forward pass uses the
custom Triton ``quantized_matmul`` kernel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning_router.kernels.quantized_matmul import quantized_matmul


class ExpertFFN(nn.Module):
    """Single SwiGLU expert, optionally quantized to 4-bit."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quantized: bool = False,
        group_size: int = 128,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.quantized = quantized
        self.group_size = group_size

        if not quantized:
            # Full-precision weights
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        else:
            # Packed INT4 weight buffers (not nn.Parameter – frozen at inference)
            self._register_quant_buffers("gate", hidden_size, intermediate_size)
            self._register_quant_buffers("up", hidden_size, intermediate_size)
            self._register_quant_buffers("down", intermediate_size, hidden_size)

    # ── Quantized buffer helpers ──────────────────────────────────────

    def _register_quant_buffers(self, name: str, K: int, N: int) -> None:
        num_groups = K // self.group_size
        self.register_buffer(f"{name}_qweight", torch.zeros(K, N // 8, dtype=torch.int32))
        self.register_buffer(f"{name}_scales", torch.zeros(num_groups, N, dtype=torch.float16))
        self.register_buffer(f"{name}_zeros", torch.zeros(num_groups, N // 8, dtype=torch.int32))
        # Store unpacked N for the kernel wrapper
        setattr(self, f"{name}_N", N)

    def _qmatmul(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Run quantized matmul for buffer set ``name``."""
        return quantized_matmul(
            A=x.half(),
            B_packed=getattr(self, f"{name}_qweight"),
            scales=getattr(self, f"{name}_scales"),
            zeros=getattr(self, f"{name}_zeros"),
            N=getattr(self, f"{name}_N"),
            group_size=self.group_size,
        )

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU FFN:  down(SiLU(gate(x)) * up(x))

        Parameters
        ----------
        x : (num_tokens, hidden_size)

        Returns
        -------
        (num_tokens, hidden_size)
        """
        if not self.quantized:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        else:
            gate_out = self._qmatmul(x, "gate")
            up_out = self._qmatmul(x, "up")
            hidden = F.silu(gate_out) * up_out
            return self._qmatmul(hidden, "down")


class ExpertGroup(nn.Module):
    """Collection of *N* independent expert FFN modules."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        quantized: bool = False,
        group_size: int = 128,
    ):
        super().__init__()
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_size, intermediate_size, quantized, group_size)
            for _ in range(num_experts)
        ])

    def forward(
        self,
        permuted_tokens: torch.Tensor,   # (total_capacity, hidden)
        expert_offsets: torch.Tensor,     # (num_experts + 1,) CSR
    ) -> torch.Tensor:
        """Run each expert on its contiguous token slice."""
        outputs = torch.empty_like(permuted_tokens)
        for i, expert in enumerate(self.experts):
            start = int(expert_offsets[i].item())
            end = int(expert_offsets[i + 1].item())
            if end > start:
                outputs[start:end] = expert(permuted_tokens[start:end])
        return outputs
