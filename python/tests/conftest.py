"""Pytest fixtures shared by the tests."""

import pytest
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from tiny import tiny_config


@pytest.fixture(scope="session")
def reference() -> Gemma4ForCausalLM:
    """A tiny 42-layer Transformers model with random weights."""
    torch.manual_seed(219)
    return Gemma4ForCausalLM(tiny_config()).eval()
