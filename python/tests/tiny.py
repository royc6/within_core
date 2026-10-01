"""Test helpers: toy-width Gemma 4 models with E4B's topology, and a host loop.

The tests compare the Neural Engine formulation with the Transformers reference on tiny
models, author tiny chunks, and run the command-line steps on a tiny checkpoint. Nothing is
downloaded and no Core AI program is executed.
"""

import hashlib
import itertools
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from within_core.chunks import (
    SLIDING,
    ChunkLayout,
    DecoderBlock,
    HeadChunk,
    HeadLayout,
    LayerChunk,
)
from within_core.layers import PerLayerProjection, RMSNorm, pointwise

E4B_LAYER_TYPES = ["full_attention" if i % 6 == 5 else SLIDING for i in range(42)]
WINDOW = 5
CAPACITY = 16
TOKENIZER_STUBS = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def tiny_config(layers: int = 42) -> Gemma4TextConfig:
    """Return a toy-width configuration; 42 layers give E4B's exact layer topology."""
    e4b = layers == 42
    config = Gemma4TextConfig(
        vocab_size=64,
        vocab_size_per_layer_input=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=16,
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=18 if e4b else 2,
        layer_types=E4B_LAYER_TYPES if e4b else [SLIDING, "full_attention"] * (layers // 2),
        max_position_embeddings=64,
        sliding_window=WINDOW,
        final_logit_softcapping=30.0,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
    )
    config._attn_implementation = "eager"
    return config


def channels_first(x: Tensor) -> Tensor:
    """``[B, S, C]`` to ``[B, C, 1, S]``."""
    return x.transpose(1, 2).unsqueeze(2)


def build_chunks(
    reference: Gemma4ForCausalLM, boundaries: tuple[int, ...], head_parts: int
) -> list[LayerChunk | HeadChunk]:
    """Split a Transformers model into chunks without compression."""
    config, model = reference.config, reference.model
    ple_dim = config.hidden_size_per_layer_input
    chunks: list[LayerChunk | HeadChunk] = []
    for start, end in itertools.pairwise(boundaries):
        layout = ChunkLayout.from_config(config, start, end, CAPACITY)
        ple = PerLayerProjection(
            model.per_layer_model_projection.weight[start * ple_dim : end * ple_dim],
            model.per_layer_projection_norm.weight,
            eps=config.rms_norm_eps,
            layers=end - start,
            projection_scale=model.per_layer_model_projection_scale,
            input_scale=model.per_layer_input_scale,
        )
        blocks = [DecoderBlock(model.layers[i]) for i in range(start, end)]
        chunks.append(LayerChunk(layout, blocks, ple).eval())
    rows = config.vocab_size // head_parts
    layout = HeadLayout(config.num_hidden_layers, head_parts, rows, config.hidden_size)
    parts = [pointwise(weight) for weight in reference.lm_head.weight.split(rows)]
    norm = RMSNorm.from_module(model.norm)
    chunks.append(HeadChunk(layout, norm, parts, config.final_logit_softcapping).eval())
    return chunks


def ring_mask(position: int, length: int, window: int) -> Tensor:
    """``[1, 1, L, window + L]`` mask over a ring cache written at ``p % window``."""
    mask = torch.full((length, window + length), -1e4)
    for query in range(length):
        current = position + query
        for slot in range(window):
            if position:
                held = position - 1 - ((position - 1 - slot) % window)
                if held >= 0 and current - held < window:
                    mask[query, slot] = 0
        for fresh in range(length):
            if position + fresh <= current and current - (position + fresh) < window:
                mask[query, window + fresh] = 0
    return mask[None, None]


def full_mask(position: int, length: int, capacity: int) -> Tensor:
    """``[1, 1, L, capacity + L]`` mask over a cache written at ``p``."""
    mask = torch.full((length, capacity + length), -1e4)
    for query in range(length):
        mask[query, :position] = 0
        mask[query, capacity : capacity + query + 1] = 0
    return mask[None, None]


class Host:
    """The app's host loop in miniature: caches, masks, RoPE and the chunk sequence."""

    def __init__(self, reference: Gemma4ForCausalLM, chunks: list[LayerChunk | HeadChunk]) -> None:
        self.reference = reference
        self.chunks = chunks
        layout = chunks[0].layout
        self.caches = {}
        # Dirty unwritten slots expose missing masking.
        torch.manual_seed(5)
        for layer in range(layout.first_shared):
            local = layout.layer_types[layer] == SLIDING
            dimension = layout.local_head_dim if local else layout.global_head_dim
            slots = WINDOW if local else CAPACITY
            for kind in ("key", "value"):
                self.caches[f"{kind}_{layer}"] = torch.randn(1, layout.kv_heads, dimension, slots)

    def inputs(self, tokens: Tensor, position: int) -> dict[str, Tensor]:
        """Return the shared per-call inputs for ``tokens`` at ``position``."""
        model = self.reference.model
        length = tokens.shape[1]
        embedded = model.embed_tokens(tokens)
        identity = model.get_per_layer_inputs(tokens, embedded)
        positions = torch.arange(position, position + length)[None]
        local = model.rotary_emb(embedded, positions, "sliding_attention")
        global_ = model.rotary_emb(embedded, positions, "full_attention")
        values = {
            "embedded": channels_first(embedded),
            "identity": identity.permute(0, 3, 2, 1),
            "local_cos": channels_first(local[0]),
            "local_sin": channels_first(local[1]),
            "global_cos": channels_first(global_[0]),
            "global_sin": channels_first(global_[1]),
            "local_mask": ring_mask(position, length, WINDOW),
            "global_mask": full_mask(position, length, CAPACITY),
        }
        return values | self.caches

    @torch.no_grad()
    def call(self, tokens: Tensor, position: int, prefill: bool) -> Tensor | None:
        """Run one prefill block or decode step; return the logits of a decode step."""
        values = self.inputs(tokens, position)
        fresh: dict[str, Tensor] = {}
        logits = None
        for chunk in self.chunks:
            layout = chunk.layout
            if prefill and not layout.has_prefill:
                break
            if isinstance(chunk, LayerChunk):
                values["token_identity"] = values["identity"][:, :, layout.start : layout.end]
            chunk.prefill_mode = prefill
            outputs = chunk(tuple(values[name] for name in layout.input_names))
            chunk.prefill_mode = False
            names = layout.prefill_outputs if prefill else layout.step_outputs
            produced = dict(zip(names, outputs, strict=True))
            if isinstance(chunk, HeadChunk):
                logits = torch.cat([produced[name] for name in layout.step_outputs], dim=1)
                continue
            if "hidden" in produced:
                values["hidden"] = produced.pop("hidden")
            fresh |= produced
            values |= produced
        self.write(fresh, position, tokens.shape[1])
        return logits

    def write(self, fresh: dict[str, Tensor], position: int, length: int) -> None:
        """Write fresh K/V into the caches: at ``p`` (full) or ``p % window`` (sliding)."""
        for name, value in fresh.items():
            cache = self.caches[name.removeprefix("fresh_")]
            for offset in range(length):
                slot = (position + offset) % cache.shape[-1]
                cache[..., slot] = value[..., offset]


def write_checkpoint(folder: Path, model: Gemma4ForCausalLM) -> dict:
    """Write ``model`` as a checkpoint folder; return its pinned reference description."""
    folder.mkdir(parents=True)
    state = {
        "model.language_model." + name.removeprefix("model."): value.to(torch.bfloat16)
        for name, value in model.state_dict().items()
        if name.startswith("model.")
    }
    save_file(state, folder / "model.safetensors")
    text_config = json.loads(json.dumps(model.config.to_dict()))
    (folder / "config.json").write_text(json.dumps({"text_config": text_config}))
    for name in TOKENIZER_STUBS:
        (folder / name).write_text(f"{name}\n")
    files = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(folder.iterdir())
    }
    return {"source": "tiny", "revision": "0", "files": files, "text_config": text_config}
