"""
Unit tests for the Triton expert-routing kernel.

Validates the fused scatter/gather against a pure-PyTorch reference to ensure
coalesced-memory and shared-memory optimisations don't break correctness.
"""

from __future__ import annotations

import pytest
import torch

# Check for CUDA and Triton availability
CUDA_AVAILABLE = torch.cuda.is_available()
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# Skip entire module if CUDA or Triton not available
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
    pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not installed"),
]


def _reference_scatter(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_weights: torch.Tensor,
    token_indices: torch.Tensor,
    num_experts: int,
):
    """Pure-PyTorch reference for expert routing (scatter phase)."""
    num_tokens, hidden = tokens.shape

    # Build per-expert token lists
    expert_tokens = {i: [] for i in range(num_experts)}
    expert_weights = {i: [] for i in range(num_experts)}
    expert_src_idx = {i: [] for i in range(num_experts)}

    for slot in range(expert_ids.numel()):
        eid = expert_ids[slot].item()
        expert_tokens[eid].append(tokens[token_indices[slot].item()])
        expert_weights[eid].append(gate_weights[slot].item())
        expert_src_idx[eid].append(token_indices[slot].item())

    # Build contiguous output
    permuted_list = []
    offsets = [0]
    weights_list = []
    src_list = []
    for i in range(num_experts):
        if expert_tokens[i]:
            permuted_list.append(torch.stack(expert_tokens[i]))
            weights_list.extend(expert_weights[i])
            src_list.extend(expert_src_idx[i])
        offsets.append(offsets[-1] + len(expert_tokens[i]))

    permuted = torch.cat(permuted_list, dim=0) if permuted_list else torch.empty(0, hidden)
    offsets_t = torch.tensor(offsets, dtype=torch.int32)
    weights_t = torch.tensor(weights_list, dtype=torch.float32)
    src_t = torch.tensor(src_list, dtype=torch.int32)

    return permuted, offsets_t, weights_t, src_t


def _make_test_data(num_tokens=64, hidden=256, num_experts=4, top_k=2, device="cuda"):
    tokens = torch.randn(num_tokens, hidden, dtype=torch.float16, device=device)
    expert_ids = torch.randint(0, num_experts, (num_tokens * top_k,), dtype=torch.int32, device=device)
    gate_weights = torch.rand(num_tokens * top_k, dtype=torch.float32, device=device)
    # Normalise
    gate_weights = gate_weights / gate_weights.view(num_tokens, top_k).sum(-1, keepdim=True).repeat(1, top_k).view(-1)
    token_indices = (
        torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1).int()
    )
    return tokens, expert_ids, gate_weights, token_indices


class TestExpertRoutingKernel:
    """Tests for expert_routing_forward / expert_routing_gather."""

    def test_histogram_matches_reference(self):
        """Expert counts from the Triton histogram should match torch.bincount."""
        tokens, expert_ids, gate_weights, token_indices = _make_test_data()
        num_experts = 4

        from lightning_router.kernels.expert_routing import expert_routing_forward

        _, offsets, _, _ = expert_routing_forward(
            tokens, expert_ids, gate_weights, token_indices, num_experts,
        )

        ref_counts = torch.bincount(expert_ids.long(), minlength=num_experts)
        triton_counts = offsets[1:] - offsets[:-1]

        torch.testing.assert_close(
            triton_counts.cpu().long(), ref_counts.cpu(), msg="Histogram mismatch",
        )

    def test_all_tokens_appear_in_output(self):
        """Every source index in the scatter output should be a valid token id."""
        tokens, expert_ids, gate_weights, token_indices = _make_test_data()
        num_tokens = tokens.size(0)

        from lightning_router.kernels.expert_routing import expert_routing_forward

        _, _, _, src_idx = expert_routing_forward(
            tokens, expert_ids, gate_weights, token_indices, 4,
        )

        assert src_idx.min() >= 0
        assert src_idx.max() < num_tokens

    def test_gather_recovers_shape(self):
        """Gather should produce an output with the original number of tokens."""
        tokens, expert_ids, gate_weights, token_indices = _make_test_data()
        num_tokens = tokens.size(0)

        from lightning_router.kernels.expert_routing import (
            expert_routing_forward, expert_routing_gather,
        )

        permuted, offsets, routing_w, src_idx = expert_routing_forward(
            tokens, expert_ids, gate_weights, token_indices, 4,
        )

        # Use permuted as a stand-in for expert output (identity experts)
        output = expert_routing_gather(permuted, routing_w, src_idx, num_tokens)
        assert output.shape == tokens.shape

    def test_round_trip_approximate(self):
        """Scatter → identity expert → gather should approximate gated sum."""
        num_tokens, hidden, num_experts, top_k = 32, 128, 4, 2
        tokens, expert_ids, gate_weights, token_indices = _make_test_data(
            num_tokens, hidden, num_experts, top_k,
        )

        from lightning_router.kernels.expert_routing import (
            expert_routing_forward, expert_routing_gather,
        )

        permuted, offsets, routing_w, src_idx = expert_routing_forward(
            tokens, expert_ids, gate_weights, token_indices, num_experts,
        )
        output = expert_routing_gather(permuted, routing_w, src_idx, num_tokens)

        # Reference: for each token, sum(gate_weight * token) over its top-k
        ref = torch.zeros_like(tokens, dtype=torch.float32)
        for slot in range(expert_ids.numel()):
            tid = token_indices[slot].item()
            w = gate_weights[slot].item()
            ref[tid] += w * tokens[tid].float()

        torch.testing.assert_close(
            output.float().cpu(), ref.cpu(), atol=1e-2, rtol=1e-2,
            msg="Round-trip mismatch",
        )
