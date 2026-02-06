"""
Full MoE layer: gating → routing → expert FFN → gather.

This module wires together the gating network, the Triton-accelerated expert
routing kernels, and the (optionally quantized) expert FFNs into a single
drop-in ``nn.Module`` that replaces the dense FFN in each transformer block.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lightning_router.config import MoEConfig, QuantizationConfig, ModelConfig, KernelConfig
from lightning_router.kernels.expert_routing import expert_routing_forward, expert_routing_gather
from lightning_router.models.gating import TopKGating
from lightning_router.models.experts import ExpertGroup


class MoELayer(nn.Module):
    """
    Mixture-of-Experts transformer FFN replacement.

    Pipeline per forward call
    ─────────────────────────
    1. **Gating** – TopKGating computes expert assignments & gate weights.
    2. **Scatter** – Triton ``expert_routing_forward`` permutes tokens into
       expert-contiguous layout with coalesced memory access.
    3. **Expert FFN** – Each expert processes its contiguous slice (optionally
       using 4-bit quantized Triton matmul).
    4. **Gather** – Triton ``expert_routing_gather`` scatters results back,
       weighted by gate scores.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        moe_cfg: MoEConfig,
        quant_cfg: QuantizationConfig,
        kernel_cfg: KernelConfig,
    ):
        super().__init__()
        self.moe_cfg = moe_cfg
        self.kernel_cfg = kernel_cfg
        self.use_triton_routing = moe_cfg.routing_implementation == "triton"

        # Sub-modules
        self.gate = TopKGating(
            hidden_size=model_cfg.hidden_size,
            num_experts=moe_cfg.num_experts,
            top_k=moe_cfg.num_experts_per_token,
            load_balance_weight=moe_cfg.load_balance_loss_weight,
        )
        self.experts = ExpertGroup(
            num_experts=moe_cfg.num_experts,
            hidden_size=model_cfg.hidden_size,
            intermediate_size=model_cfg.intermediate_size,
            quantized=quant_cfg.enabled and quant_cfg.quantize_experts,
            group_size=quant_cfg.group_size,
        )

    # ── Fallback: pure-PyTorch routing (for validation / CPU) ─────────

    def _torch_routing(
        self,
        tokens: torch.Tensor,
        expert_ids: torch.Tensor,
        gate_weights: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Reference routing implementation in plain PyTorch."""
        num_tokens, hidden = tokens.shape
        output = torch.zeros_like(tokens)

        for i in range(self.moe_cfg.num_experts):
            mask = expert_ids == i
            if not mask.any():
                continue
            slot_indices = mask.nonzero(as_tuple=True)[0]
            toks = token_indices[slot_indices]
            weights = gate_weights[slot_indices].unsqueeze(1)
            expert_in = tokens[toks]
            expert_out = self.experts.experts[i](expert_in)
            output.index_add_(0, toks, expert_out * weights)

        return output

    # ── Forward ───────────────────────────────────────────────────────

    def forward(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden_states : (batch, seq_len, hidden) or (num_tokens, hidden)

        Returns
        -------
        output    : same shape as ``hidden_states``
        aux_loss  : scalar load-balancing loss
        """
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, orig_shape[-1])
        num_tokens = hidden_states.size(0)

        # 1. Gating
        expert_ids, gate_weights, token_indices, aux_loss = self.gate(hidden_states)

        if self.use_triton_routing and hidden_states.is_cuda:
            # 2. Triton scatter
            permuted, offsets, routing_w, src_idx = expert_routing_forward(
                tokens=hidden_states,
                expert_ids=expert_ids,
                gate_weights=gate_weights,
                token_indices=token_indices,
                num_experts=self.moe_cfg.num_experts,
                capacity_factor=self.moe_cfg.expert_capacity_factor,
            )

            # 3. Expert FFN (contiguous slices)
            expert_out = self.experts(permuted, offsets)

            # 4. Triton gather
            output = expert_routing_gather(expert_out, routing_w, src_idx, num_tokens)
        else:
            output = self._torch_routing(
                hidden_states, expert_ids, gate_weights, token_indices,
            )

        if len(orig_shape) == 3:
            output = output.view(orig_shape)

        return output, aux_loss
