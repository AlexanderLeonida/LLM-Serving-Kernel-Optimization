"""
Top-K Softmax gating network for Mixture-of-Experts routing.

Produces expert assignments, gate weights, and an auxiliary load-balancing loss
that encourages uniform expert utilisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKGating(nn.Module):
    """
    Top-K softmax router.

    For each token, projects the hidden state to ``num_experts`` logits,
    takes the softmax, and keeps only the top-``k`` experts.

    Parameters
    ----------
    hidden_size : int
        Dimension of token representations.
    num_experts : int
        Total number of experts.
    top_k : int
        How many experts each token is routed to.
    load_balance_weight : float
        Coefficient for the auxiliary load-balancing loss.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        self.gate_proj = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden_states : (batch_size, seq_len, hidden_size) or (num_tokens, hidden_size)

        Returns
        -------
        expert_ids     : (num_tokens * top_k,) int64 – flat expert assignments
        gate_weights   : (num_tokens * top_k,) float32 – normalised gate scores
        token_indices  : (num_tokens * top_k,) int64 – source token for each slot
        aux_loss       : scalar – load-balancing loss
        """
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, orig_shape[-1])
        num_tokens = hidden_states.size(0)

        # ── Gate logits → softmax ─────────────────────────────────────
        logits = self.gate_proj(hidden_states)       # (num_tokens, num_experts)
        probs = F.softmax(logits, dim=-1)            # (num_tokens, num_experts)

        # ── Top-k selection ───────────────────────────────────────────
        topk_weights, topk_ids = torch.topk(probs, self.top_k, dim=-1)
        # Re-normalise so weights sum to 1 per token
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # ── Flatten to slot-level tensors ─────────────────────────────
        expert_ids = topk_ids.reshape(-1)            # (num_tokens * top_k,)
        gate_weights = topk_weights.reshape(-1).float()
        token_indices = (
            torch.arange(num_tokens, device=hidden_states.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )

        # ── Auxiliary load-balancing loss (Switch-Transformer style) ──
        # f_i = fraction of tokens routed to expert i
        # P_i = mean gate probability for expert i
        # loss = num_experts * sum(f_i * P_i)
        tokens_per_expert = torch.zeros(
            self.num_experts, device=hidden_states.device, dtype=torch.float32,
        )
        tokens_per_expert.scatter_add_(
            0, expert_ids.long(), torch.ones_like(gate_weights),
        )
        f = tokens_per_expert / (num_tokens * self.top_k)
        P = probs.mean(dim=0)
        aux_loss = self.load_balance_weight * self.num_experts * (f * P).sum()

        return expert_ids.int(), gate_weights, token_indices.int(), aux_loss
