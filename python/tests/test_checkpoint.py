"""Pinned, offline checkpoint reads."""

import json
from pathlib import Path

import pytest
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from tiny import write_checkpoint
from within_core.checkpoint import LocalCheckpoint, load_reference


@pytest.fixture(scope="module")
def tiny_checkpoint(
    tmp_path_factory: pytest.TempPathFactory, reference: Gemma4ForCausalLM
) -> tuple[Path, dict]:
    """A tiny checkpoint folder and its pinned description."""
    folder = tmp_path_factory.mktemp("tiny") / "checkpoint"
    return folder, write_checkpoint(folder, reference)


def test_pinned_reference_describes_e4b() -> None:
    reference = load_reference()
    assert reference["source"] == "google/gemma-4-e4b-it"
    assert set(reference["files"]) == {
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    }
    assert reference["text_config"]["num_hidden_layers"] == 42


def test_reads_are_cast_and_bounded(
    tiny_checkpoint: tuple[Path, dict], reference: Gemma4ForCausalLM
) -> None:
    checkpoint = LocalCheckpoint(*tiny_checkpoint)
    rows = checkpoint.row_range("embed_tokens.weight", 8, 12)
    expected = reference.model.embed_tokens.weight[8:12].detach().bfloat16().half()
    assert rows.dtype == torch.float16
    assert torch.equal(rows, expected)
    owner, consumer = checkpoint.layer(23), checkpoint.layer(24)
    assert owner.self_attn.k_proj.weight.dtype == torch.float16
    assert not hasattr(consumer.self_attn, "k_proj")
    with pytest.raises(ValueError, match="outside"):
        checkpoint.layer(42)


def test_a_changed_file_is_refused_before_use(tiny_checkpoint: tuple[Path, dict]) -> None:
    folder, pinned = tiny_checkpoint
    checkpoint = LocalCheckpoint(folder, pinned)
    with pytest.raises(ValueError, match="not a pinned"):
        checkpoint.path("README.md")
    changed = json.loads(json.dumps(pinned))
    changed["files"]["model.safetensors"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not the pinned file"):
        LocalCheckpoint(folder, changed).tensor("norm.weight")
    changed["text_config"]["hidden_size"] = 64
    with pytest.raises(ValueError, match="text configuration"):
        LocalCheckpoint(folder, changed)
