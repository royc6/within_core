"""Mixed-precision compression of the projections and the bounded vocabulary head."""

import pytest
import torch
from torch import nn
from torch.nn.utils.parametrize import is_parametrized
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from within_core.chunks import DecoderBlock
from within_core.layers import pointwise
from within_core.quantize import (
    quantize_conv,
    quantize_head,
    quantize_projections,
    require_compressed,
)


@torch.no_grad()
def test_block_projections_take_their_schemes(reference: Gemma4ForCausalLM) -> None:
    block = DecoderBlock(reference.model.layers[0]).half()
    with pytest.raises(ValueError, match="escaped compression"):
        require_compressed(block)
    assert quantize_projections(block) == 9
    require_compressed(block)
    assert quantize_projections(block) == 0
    for name, conv in block.named_modules():
        if not isinstance(conv, nn.Conv2d):
            continue
        assert is_parametrized(conv)
        palette = any(hasattr(m, "lut") for m in conv.modules())
        assert palette == (name in ("feedforward.gate", "feedforward.up", "feedforward.down"))


@torch.no_grad()
def test_bounded_head_equals_whole_head_quantization() -> None:
    torch.manual_seed(9)
    dense = torch.randn(64, 16, dtype=torch.float16)
    parts = quantize_head(lambda start, end: dense[start:end].clone(), 64, 16, 4, block_rows=24)
    _, whole = quantize_conv(pointwise(dense), "whole", "w4")
    data = torch.cat([part.parametrizations.weight[0].quantized_data for part in parts])
    scale = torch.cat([part.parametrizations.weight[0].scale for part in parts])
    assert parts[0].parametrizations.weight[0].input_dtype == torch.int4
    assert torch.equal(data, whole.quantized_data)
    assert torch.equal(scale, whole.scale)
    x = torch.randn(1, 16, 1, 3, dtype=torch.float16)
    logits = torch.cat([part(x) for part in parts], dim=1)
    dequantized = (whole.quantized_data.float() * whole.scale.float()).half()
    expected = nn.functional.conv2d(x, dequantized)
    torch.testing.assert_close(logits, expected, rtol=1e-3, atol=1e-3)


def test_head_parts_must_divide_the_vocabulary() -> None:
    with pytest.raises(ValueError, match="divide"):
        quantize_head(lambda start, end: torch.zeros(end - start, 16).half(), 64, 16, 5)
