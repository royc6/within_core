"""Norm, per-layer projection and feedforward against Transformers."""

import copy

import pytest
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from tiny import channels_first
from within_core.layers import FeedForwardAndPLE, PerLayerProjection, RMSNorm


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.no_grad()
def test_rms_norm_matches_transformers(reference: Gemma4ForCausalLM, dtype: torch.dtype) -> None:
    source = copy.deepcopy(reference.model.layers[0].input_layernorm).to(dtype)
    torch.manual_seed(1)
    x = torch.randn(1, 8, 32, dtype=dtype)
    actual = RMSNorm.from_module(source)(channels_first(x)).squeeze(2).transpose(1, 2)
    tolerance = 2e-6 if dtype == torch.float32 else 6e-3
    torch.testing.assert_close(actual, source(x), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("amplitude", [0.001, 1.0, 1000.0])
@torch.no_grad()
def test_fp16_norm_stays_finite_and_close(reference: Gemma4ForCausalLM, amplitude: float) -> None:
    source = reference.model.layers[0].input_layernorm
    torch.manual_seed(2)
    x = (torch.randn(1, 8, 32) * amplitude).half()
    actual = RMSNorm.from_module(source).half()(channels_first(x)).squeeze(2).transpose(1, 2)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), source(x.float()), atol=6e-3, rtol=6e-3)


@torch.no_grad()
def test_norm_applies_learned_scale_before_fp16_rounding() -> None:
    """These values round differently if the scale is applied after an FP16 cast."""
    torch.manual_seed(4)
    weight = torch.linspace(0.25, 16, 32).half()
    x = torch.randn(1, 32, 1, 8).half()
    expected = x.float() * torch.rsqrt(x.float().pow(2).mean(1, keepdim=True) + 1e-6)
    expected = (expected * weight.float()[None, :, None, None]).half()
    torch.testing.assert_close(RMSNorm(weight, 1e-6)(x), expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("length", [1, 8])
@torch.no_grad()
def test_per_layer_projection_matches_transformers(
    reference: Gemma4ForCausalLM, dtype: torch.dtype, length: int
) -> None:
    model = reference.model
    torch.manual_seed(5)
    tokens = torch.randint(0, 64, (1, length))
    embedded = model.embed_tokens(tokens)
    identity = model.get_per_layer_inputs(tokens, embedded)
    expected = model.project_per_layer_inputs(embedded, identity)
    target = PerLayerProjection(
        model.per_layer_model_projection.weight.to(dtype),
        model.per_layer_projection_norm.weight.to(dtype),
        eps=reference.config.rms_norm_eps,
        layers=reference.config.num_hidden_layers,
        projection_scale=model.per_layer_model_projection_scale,
        input_scale=model.per_layer_input_scale,
    )
    actual = target(channels_first(embedded).to(dtype), identity.permute(0, 3, 2, 1).to(dtype))
    tolerance = 2e-5 if dtype == torch.float32 else 6e-3
    actual = actual.permute(0, 3, 2, 1).float()
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("length", [1, 8])
@torch.no_grad()
def test_feedforward_and_ple_match_transformers(reference: Gemma4ForCausalLM, length: int) -> None:
    layer = reference.model.layers[0]
    torch.manual_seed(6)
    hidden = torch.randn(1, length, 32)
    ple = torch.randn(1, length, 8)
    expected = hidden + layer.post_feedforward_layernorm(
        layer.mlp(layer.pre_feedforward_layernorm(hidden))
    )
    gate = layer.act_fn(layer.per_layer_input_gate(expected))
    expected = expected + layer.post_per_layer_input_norm(layer.per_layer_projection(gate * ple))
    expected = expected * layer.layer_scalar
    actual = FeedForwardAndPLE(layer)(channels_first(hidden), channels_first(ple))
    # GELU in sigmoid form with FP16 coefficients differs from tanh GELU by about 1e-4.
    torch.testing.assert_close(actual.squeeze(2).transpose(1, 2), expected, atol=1e-3, rtol=1e-3)


@torch.no_grad()
def test_trained_activation_range_stays_finite_in_fp16(reference: Gemma4ForCausalLM) -> None:
    """Trained layers produce gated products far larger than random weights do."""
    target = FeedForwardAndPLE(reference.model.layers[0]).half()
    for projection, value in ((target.gate, 28), (target.up, 36)):
        projection.weight.zero_()
        projection.weight[:, 0, 0, 0] = value
    hidden = torch.ones(1, 32, 1, 4, dtype=torch.float16)
    ple = torch.ones(1, 8, 1, 4, dtype=torch.float16)
    assert torch.isfinite(target(hidden, ple)).all()
