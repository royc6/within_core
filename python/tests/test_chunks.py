"""Chunk contract, KV layout, and chunked execution against Transformers."""

import itertools

import pytest
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from tiny import Host, build_chunks
from within_core.checkpoint import pinned_config
from within_core.chunks import ChunkLayout, HeadLayout, LayerChunk, donor_layers
from within_core.export import BOUNDARIES, plan

# The Neural Engine formulation (layer-norm RMSNorm, sigmoid GELU with FP16 coefficients)
# differs from Transformers by rounding only; wiring errors show up as O(1) differences.
ATOL, RTOL = 5e-4, 5e-4


def test_e4b_plan_topology() -> None:
    layouts = plan(pinned_config())
    assert len(layouts) == 8
    layers, head = layouts[:7], layouts[7]
    assert [(c.start, c.end) for c in layers] == list(itertools.pairwise(BOUNDARIES))
    assert [c.has_prefill for c in layouts] == [True] * 4 + [False] * 4
    assert [c.owned for c in layers[:4]] == [tuple(range(s, s + 6)) for s in (0, 6, 12, 18)]
    assert [c.prefill_outputs[0] for c in layers[:4]] == ["hidden"] * 3 + ["fresh_key_18"]
    donors = {"sliding_attention": 22, "full_attention": 23}
    assert donor_layers(layers[0].layer_types, 24) == donors
    donor_inputs = "key_22 value_22 fresh_key_22 fresh_value_22 key_23 value_23 fresh_key_23"
    for consumer in layers[4:]:
        assert consumer.owned == ()
        assert consumer.external_donors == (22, 23)
        assert consumer.step_outputs == ("hidden",)
        assert consumer.input_names[-8:] == (*donor_inputs.split(), "fresh_value_23")
    assert layers[0].input_names[:2] == ("embedded", "token_identity")
    assert layers[1].input_names[:2] == ("hidden", "embedded")
    assert isinstance(head, HeadLayout)
    assert head.step_outputs == tuple(f"logits_{i:02d}" for i in range(16))
    assert head.contract("chunk-07.aimodel", 16)["head"] == {"parts": 16, "rows": 16384}


def test_e4b_cache_shapes_use_rings_for_sliding_layers() -> None:
    first = plan(pinned_config())[0]
    shapes = first.shapes(16)
    assert shapes["key_0"] == [1, 2, 256, 512]
    assert shapes["value_5"] == [1, 2, 512, 16384]
    assert shapes["local_mask"] == [1, 1, 16, 512 + 16]
    assert shapes["global_mask"] == [1, 1, 16, 16384 + 16]
    assert shapes["token_identity"] == [1, 256, 6, 16]
    assert list(first.contract("chunk-00.aimodel", 16)["functions"]) == ["step_1", "prefill_16"]


def test_layout_rejects_invalid_ranges() -> None:
    config = pinned_config()
    with pytest.raises(ValueError):
        ChunkLayout.from_config(config, 6, 6, 16384)
    with pytest.raises(ValueError):
        ChunkLayout.from_config(config, 36, 43, 16384)
    with pytest.raises(ValueError):
        ChunkLayout.from_config(config, 0, 6, 512)


def test_layer_chunk_rejects_blocks_of_another_range(reference: Gemma4ForCausalLM) -> None:
    chunk = build_chunks(reference, BOUNDARIES, 4)[0]
    layout = ChunkLayout.from_config(reference.config, 24, 30, 16)
    with pytest.raises(ValueError, match="sharing"):
        LayerChunk(layout, list(chunk.blocks), chunk.ple)


@pytest.mark.parametrize("boundaries", [BOUNDARIES, (0, 10, 21, 28, 42)])
def test_chunks_driven_by_host_match_transformers(
    reference: Gemma4ForCausalLM, boundaries: tuple[int, ...]
) -> None:
    """Headless prefill of 4 + 4 + 1 tokens, then four decode steps, with ring caches.

    13 positions through a window of 5 rewrite ring slots more than twice. The second split
    puts the donors (22, 23) in a chunk that also holds consumers.
    """
    host = Host(reference, build_chunks(reference, boundaries, head_parts=4))
    torch.manual_seed(3)
    tokens = torch.randint(3, 64, (1, 13))
    position = 0
    for length, prefill in [(4, True), (4, True), (1, True)] + [(1, False)] * 4:
        logits = host.call(tokens[:, position : position + length], position, prefill)
        if prefill:
            assert logits is None
        else:
            with torch.no_grad():
                expected = reference(input_ids=tokens[:, : position + length], use_cache=False)
            actual = logits.squeeze(2).transpose(1, 2)
            torch.testing.assert_close(actual, expected.logits[:, -length:], atol=ATOL, rtol=RTOL)
        position += length
