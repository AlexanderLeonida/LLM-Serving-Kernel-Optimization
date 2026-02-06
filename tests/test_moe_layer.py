"""
Tests for the TopKGating module and full MoE layer.
"""

from __future__ import annotations

import pytest
import torch

from lightning_router.models.gating import TopKGating


class TestTopKGating:
    """CPU tests for the gating network."""

    def test_output_shapes(self):
        hidden, num_experts, top_k = 128, 4, 2
        gate = TopKGating(hidden, num_experts, top_k)
        x = torch.randn(8, 16, hidden)  # (batch, seq, hidden)

        expert_ids, weights, token_idx, loss = gate(x)

        num_tokens = 8 * 16
        assert expert_ids.shape == (num_tokens * top_k,)
        assert weights.shape == (num_tokens * top_k,)
        assert token_idx.shape == (num_tokens * top_k,)
        assert loss.dim() == 0  # scalar

    def test_expert_ids_in_range(self):
        gate = TopKGating(64, 4, 2)
        x = torch.randn(4, 10, 64)
        expert_ids, *_ = gate(x)

        assert expert_ids.min() >= 0
        assert expert_ids.max() < 4

    def test_weights_sum_to_one(self):
        gate = TopKGating(64, 4, 2)
        x = torch.randn(4, 10, 64)
        _, weights, token_idx, _ = gate(x)

        num_tokens = 4 * 10
        weight_sums = torch.zeros(num_tokens)
        weight_sums.scatter_add_(0, token_idx.long(), weights)

        torch.testing.assert_close(
            weight_sums, torch.ones(num_tokens), atol=1e-5, rtol=1e-5,
        )

    def test_aux_loss_positive(self):
        gate = TopKGating(64, 4, 2, load_balance_weight=0.01)
        x = torch.randn(4, 10, 64)
        *_, loss = gate(x)
        assert loss.item() > 0


@pytest.mark.gpu
class TestMoELayer:
    """GPU tests for the full MoE layer."""

    def test_forward_shape(self):
        from lightning_router.config import ModelConfig, MoEConfig, QuantizationConfig, KernelConfig
        from lightning_router.models.moe_layer import MoELayer

        mcfg = ModelConfig(hidden_size=128, intermediate_size=256)
        moecfg = MoEConfig(num_experts=4, num_experts_per_token=2)
        qcfg = QuantizationConfig(enabled=False)
        kcfg = KernelConfig()

        layer = MoELayer(mcfg, moecfg, qcfg, kcfg).cuda().half()
        x = torch.randn(2, 8, 128, dtype=torch.float16, device="cuda")

        out, loss = layer(x)
        assert out.shape == x.shape
        assert loss.dim() == 0

    def test_torch_fallback_matches_shape(self):
        """PyTorch fallback should produce the same output shape."""
        from lightning_router.config import ModelConfig, MoEConfig, QuantizationConfig, KernelConfig
        from lightning_router.models.moe_layer import MoELayer

        mcfg = ModelConfig(hidden_size=64, intermediate_size=128)
        moecfg = MoEConfig(
            num_experts=4, num_experts_per_token=2, routing_implementation="torch",
        )
        qcfg = QuantizationConfig(enabled=False)
        kcfg = KernelConfig()

        layer = MoELayer(mcfg, moecfg, qcfg, kcfg).cuda().half()
        x = torch.randn(2, 4, 64, dtype=torch.float16, device="cuda")

        out, loss = layer(x)
        assert out.shape == x.shape
