"""Grouped attention with host-held KV against Transformers' attention."""

import pytest
import torch
from torch import Tensor
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from tiny import WINDOW, channels_first
from within_core.attention import Attention


class _AppendingCache:
    """Transformers cache stand-in: returns the old K/V followed by the new ones."""

    def __init__(self, key: Tensor, value: Tensor) -> None:
        self.key, self.value = key, value

    def update(self, key: Tensor, value: Tensor, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Return the cached K/V followed by ``key`` and ``value``."""
        return torch.cat((self.key, key), dim=2), torch.cat((self.value, value), dim=2)


# Layers 0 and 5 own KV (sliding, full); 40 and 41 reuse a donor's KV (sliding, full).
@pytest.mark.parametrize("layer_index", [0, 5, 40, 41])
@pytest.mark.parametrize(("step", "length"), [(0, 4), (7, 1), (19, 3)])
@torch.no_grad()
def test_attention_matches_transformers(
    reference: Gemma4ForCausalLM, layer_index: int, step: int, length: int
) -> None:
    model = reference.model
    source = model.layers[layer_index].self_attn
    torch.manual_seed(8)
    hidden = torch.randn(1, length, 32)
    positions = torch.arange(step, step + length)[None]
    cos, sin = model.rotary_emb(hidden, positions, source.layer_type)
    keys = torch.arange(step + length)[None, :]
    allowed = keys <= positions.T
    if source.is_sliding:
        allowed &= keys > positions.T - WINDOW
    mask = torch.where(allowed, 0.0, -1e9)[None, None]
    # A shared layer's cache already holds the donor's keys for this step.
    cached = step + length if source.is_kv_shared_layer else step
    key = torch.randn(1, 2, cached, source.head_dim)  # Transformers layout [B, H, K, D]
    value = torch.randn_like(key)
    shared = {source.layer_type: (key, value)} if source.is_kv_shared_layer else {}
    expected, _ = source(
        hidden, (cos, sin), mask, shared, past_key_values=_AppendingCache(key, value)
    )
    actual, fresh_key, fresh_value = Attention(source)(
        channels_first(hidden),
        channels_first(cos),
        channels_first(sin),
        key.transpose(-1, -2),
        value.transpose(-1, -2),
        mask,
    )
    torch.testing.assert_close(actual.squeeze(2).transpose(1, 2), expected, atol=2e-6, rtol=2e-5)
    if source.is_kv_shared_layer:
        assert fresh_key is None and fresh_value is None
    else:
        assert fresh_key.shape == fresh_value.shape == (1, 2, source.head_dim, length)
