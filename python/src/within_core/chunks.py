"""Chunks: the units the decoder is split into, one Core AI asset each.

Pipeline position: :mod:`within_core.export` builds one chunk at a time from the checkpoint
and authors it. A *layer chunk* runs a contiguous range of decoder layers; the *head chunk*
runs the final norm and the vocabulary head as separate row slices. Between chunks the
host passes ``hidden`` ``[1, H, 1, S]``.

KV is host-held. A chunk takes, per KV-owning layer ``n``, the read-only caches
``key_n``/``value_n`` ``[1, kv_heads, D, slots]`` and returns ``fresh_key_n``/
``fresh_value_n`` ``[1, kv_heads, D, S]``, which the host writes into its caches after the
call. Full-attention caches have one slot per context position. Sliding-attention caches
are rings of the attention window: attention is order-free over keys and RoPE is applied
before caching, so the host writes position ``p`` to slot ``p % window``. One additive mask
per attention type, ``[1, 1, S, slots + S]``, selects the valid cache slots and the causal
part of the fresh keys. Layers from ``first_shared`` on own no KV: each reads the cache of
its type's donor (the last owner of that type) plus the donor's fresh K/V from the same
step. A chunk holding such consumers but not their donor takes the donor's cache and fresh
K/V as inputs.

Every chunk has a ``step_1`` entrypoint (decode one token). Chunks that own KV also have a
``prefill_<N>`` entrypoint (N prompt tokens). Prefill is headless: it returns fresh K/V,
plus ``hidden`` when a later chunk also owns KV, and the chunks after the last KV owner
(including the head) do not run.

:class:`ChunkLayout` and :class:`HeadLayout` hold a chunk's interface (I/O names, shapes and
contract entry) and need no weights; :class:`LayerChunk` and :class:`HeadChunk` are the
modules that implement them.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import Gemma4TextConfig

from within_core.attention import Attention
from within_core.layers import FeedForwardAndPLE, PerLayerProjection, RMSNorm

SLIDING = "sliding_attention"
FULL = "full_attention"
PREFIX = {SLIDING: "local", FULL: "global"}


def donor_layers(layer_types: Sequence[str], first_shared: int) -> dict[str, int]:
    """Return the KV donor of each attention type: its last owner before sharing starts.

    Args:
        layer_types: Attention type of every decoder layer.
        first_shared: Index of the first layer that owns no KV.

    Returns:
        Map from attention type to donor layer index.
    """
    owners = layer_types[:first_shared]
    return {
        kind: max(i for i in range(first_shared) if layer_types[i] == kind)
        for kind in (SLIDING, FULL)
        if kind in owners
    }


class DecoderBlock(nn.Module):
    """One Gemma 4 decoder layer: attention, then the feedforward and PLE path."""

    def __init__(self, reference: nn.Module) -> None:
        """Initialize from a Transformers ``Gemma4TextDecoderLayer``.

        Args:
            reference: The source layer.
        """
        super().__init__()
        self.input_norm = RMSNorm.from_module(reference.input_layernorm)
        self.attention = Attention(reference.self_attn)
        self.post_norm = RMSNorm.from_module(reference.post_attention_layernorm)
        self.feedforward = FeedForwardAndPLE(reference)
        self.layer_type = reference.self_attn.layer_type
        self.stores_shared = reference.self_attn.store_full_length_kv

    def forward(
        self,
        hidden: Tensor,
        ple: Tensor,
        cos: Tensor,
        sin: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Return the new hidden state and the layer's fresh K/V (``None`` when shared)."""
        attended, fresh_key, fresh_value = self.attention(
            self.input_norm(hidden), cos, sin, key, value, mask
        )
        hidden = self.feedforward(hidden + self.post_norm(attended), ple)
        return hidden, fresh_key, fresh_value


@dataclass(frozen=True)
class ChunkLayout:
    """Interface of a layer chunk over decoder layers ``[start, end)``.

    Attributes:
        start: First decoder layer of the chunk.
        end: One past the last decoder layer of the chunk.
        layer_types: Attention type of every decoder layer of the model.
        first_shared: Index of the first layer that owns no KV.
        capacity: Slots of a full-attention cache (the context limit).
        window: Slots of a sliding-attention cache (the attention window).
        hidden_size: Width of ``hidden`` and ``embedded``.
        ple_size: Per-layer embedding width.
        kv_heads: KV heads of every layer.
        local_head_dim: Head dimension of sliding-attention layers.
        global_head_dim: Head dimension of full-attention layers.
    """

    start: int
    end: int
    layer_types: tuple[str, ...]
    first_shared: int
    capacity: int
    window: int
    hidden_size: int
    ple_size: int
    kv_heads: int
    local_head_dim: int
    global_head_dim: int

    def __post_init__(self) -> None:
        """Validate the layer range and the cache sizes."""
        layers = len(self.layer_types)
        if not 0 <= self.start < self.end <= layers or not 0 < self.first_shared <= layers:
            raise ValueError("Chunk range lies outside the decoder")
        if not 0 < self.window < self.capacity:
            raise ValueError("The sliding window must be smaller than the context capacity")
        if set(self.layer_types) - set(PREFIX):
            raise ValueError("Unsupported attention type")

    @classmethod
    def from_config(
        cls, config: Gemma4TextConfig, start: int, end: int, capacity: int
    ) -> "ChunkLayout":
        """Return the layout of layers ``[start, end)`` of a ``Gemma4TextConfig``.

        Args:
            config: The text configuration of the model.
            start: First decoder layer of the chunk.
            end: One past the last decoder layer of the chunk.
            capacity: Slots of a full-attention cache.

        Returns:
            The chunk's layout.
        """
        return cls(
            start=start,
            end=end,
            layer_types=tuple(config.layer_types),
            first_shared=config.num_hidden_layers - config.num_kv_shared_layers,
            capacity=capacity,
            window=config.sliding_window,
            hidden_size=config.hidden_size,
            ple_size=config.hidden_size_per_layer_input,
            kv_heads=config.num_key_value_heads,
            local_head_dim=config.head_dim,
            global_head_dim=config.global_head_dim,
        )

    @property
    def donors(self) -> dict[str, int]:
        """KV donor layer of each attention type."""
        return donor_layers(self.layer_types, self.first_shared)

    @property
    def owned(self) -> tuple[int, ...]:
        """Layers of the chunk that own KV."""
        return tuple(range(self.start, min(self.end, self.first_shared)))

    @property
    def external_donors(self) -> tuple[int, ...]:
        """Donors outside the chunk whose cache and fresh K/V the chunk's consumers read."""
        shared = range(max(self.start, self.first_shared), self.end)
        consumers = {self.layer_types[i] for i in shared}
        donors = self.donors
        return tuple(sorted(donors[k] for k in consumers if not self.start <= donors[k] < self.end))

    @property
    def input_names(self) -> tuple[str, ...]:
        """Input names, in the order of the authored graph's arguments."""
        kinds = set(self.layer_types[self.start : self.end])
        names = ["embedded"] if self.start == 0 else ["hidden", "embedded"]
        names.append("token_identity")
        for suffixes in (("cos", "sin"), ("mask",)):
            names += [f"{PREFIX[k]}_{s}" for k in (SLIDING, FULL) if k in kinds for s in suffixes]
        names += [f"{k}_{n}" for n in self.owned for k in ("key", "value")]
        names += [
            f"{p}{k}_{n}"
            for n in self.external_donors
            for p in ("", "fresh_")
            for k in ("key", "value")
        ]
        return tuple(names)

    @property
    def fresh_outputs(self) -> tuple[str, ...]:
        """Fresh K/V outputs of the chunk's owners."""
        return tuple(f"fresh_{k}_{n}" for n in self.owned for k in ("key", "value"))

    @property
    def has_prefill(self) -> bool:
        """Whether the chunk has a prefill entrypoint (it owns KV)."""
        return bool(self.owned)

    @property
    def prefill_outputs(self) -> tuple[str, ...]:
        """Prefill outputs: fresh K/V, after ``hidden`` if a later chunk owns KV."""
        return (("hidden",) if self.end < self.first_shared else ()) + self.fresh_outputs

    @property
    def step_outputs(self) -> tuple[str, ...]:
        """Decode outputs: ``hidden`` and fresh K/V."""
        return ("hidden",) + self.fresh_outputs

    def entrypoints(self, prefill_length: int) -> list[tuple[str, bool, int]]:
        """Return ``(name, is_prefill, length)`` for each entrypoint."""
        prefill = [(f"prefill_{prefill_length}", True, prefill_length)] if self.has_prefill else []
        return [("step_1", False, 1), *prefill]

    def shapes(self, length: int) -> dict[str, list[int]]:
        """Return the input shapes of a call with ``length`` query positions."""
        shape = {
            "embedded": [1, self.hidden_size, 1, length],
            "hidden": [1, self.hidden_size, 1, length],
            "token_identity": [1, self.ple_size, self.end - self.start, length],
            "local_cos": [1, self.local_head_dim, 1, length],
            "local_sin": [1, self.local_head_dim, 1, length],
            "global_cos": [1, self.global_head_dim, 1, length],
            "global_sin": [1, self.global_head_dim, 1, length],
            "local_mask": [1, 1, length, self.window + length],
            "global_mask": [1, 1, length, self.capacity + length],
        }
        for layer in (*self.owned, *self.external_donors):
            local = self.layer_types[layer] == SLIDING
            dimension = self.local_head_dim if local else self.global_head_dim
            slots = self.window if local else self.capacity
            for k in ("key", "value"):
                shape[f"{k}_{layer}"] = [1, self.kv_heads, dimension, slots]
                shape[f"fresh_{k}_{layer}"] = [1, self.kv_heads, dimension, length]
        return {name: shape[name] for name in self.input_names}

    def contract(self, file: str, prefill_length: int) -> dict[str, object]:
        """Return the chunk's entry in ``chunk-contract.json``."""
        functions = {
            name: {
                "length": length,
                "inputs": list(self.input_names),
                "outputs": list(self.prefill_outputs if prefill else self.step_outputs),
            }
            for name, prefill, length in self.entrypoints(prefill_length)
        }
        return {
            "file": file,
            "start": self.start,
            "end": self.end,
            "functions": functions,
            "owned": list(self.owned),
            "externalDonors": list(self.external_donors),
            "inputShapes": self.shapes(1),
        }


@dataclass(frozen=True)
class HeadLayout:
    """Interface of the head chunk: final norm and vocabulary head in ``parts`` row slices.

    The host concatenates ``logits_00``, ``logits_01``, ... in name order. Separate slice
    outputs run much faster on the Neural Engine than one vocabulary-wide output.

    Attributes:
        layers: Number of decoder layers (the chunk starts and ends after the last one).
        parts: Number of row slices of the head.
        rows: Vocabulary rows per slice.
        hidden_size: Width of ``hidden``.
    """

    layers: int
    parts: int
    rows: int
    hidden_size: int

    input_names = ("hidden",)
    prefill_outputs = ()
    has_prefill = False

    @property
    def step_outputs(self) -> tuple[str, ...]:
        """One logits output per row slice."""
        return tuple(f"logits_{i:02d}" for i in range(self.parts))

    def entrypoints(self, prefill_length: int) -> list[tuple[str, bool, int]]:
        """Return the only entrypoint, ``step_1``: prefill never reaches the head."""
        return [("step_1", False, 1)]

    def shapes(self, length: int) -> dict[str, list[int]]:
        """Return the input shapes of a call with ``length`` query positions."""
        return {"hidden": [1, self.hidden_size, 1, length]}

    def contract(self, file: str, prefill_length: int) -> dict[str, object]:
        """Return the chunk's entry in ``chunk-contract.json``."""
        return {
            "file": file,
            "start": self.layers,
            "end": self.layers,
            "owned": [],
            "externalDonors": [],
            "head": {"parts": self.parts, "rows": self.rows},
            "functions": {
                "step_1": {
                    "length": 1,
                    "inputs": list(self.input_names),
                    "outputs": list(self.step_outputs),
                }
            },
            "inputShapes": self.shapes(1),
        }


Layout = ChunkLayout | HeadLayout


class LayerChunk(nn.Module):
    """Decoder layers ``[start, end)`` with host-held KV (see the module docstring).

    ``prefill_mode`` selects which entrypoint the next trace authors; it is an authoring
    switch, not a runtime branch.
    """

    def __init__(
        self, layout: ChunkLayout, blocks: Sequence[DecoderBlock], ple: PerLayerProjection
    ) -> None:
        """Initialize from the chunk's decoder blocks and per-layer projection.

        Args:
            layout: The chunk's interface.
            blocks: One block per layer of the range.
            ple: The per-layer projection restricted to the range.

        Raises:
            ValueError: If the blocks disagree with the layout.
        """
        super().__init__()
        if len(blocks) != layout.end - layout.start:
            raise ValueError("One block per layer of the chunk is required")
        donors = layout.donors.values()
        for index, block in enumerate(blocks, start=layout.start):
            attention = block.attention
            full = block.layer_type == FULL
            dimension = layout.global_head_dim if full else layout.local_head_dim
            if block.layer_type != layout.layer_types[index]:
                raise ValueError(f"Layer {index} type disagrees with the layout")
            if attention.shared != (index >= layout.first_shared):
                raise ValueError(f"Layer {index} sharing disagrees with the layout")
            if index in donors and not block.stores_shared:
                raise ValueError(f"Layer {index} should be a KV donor")
            if (attention.kv_heads, attention.head_dim) != (layout.kv_heads, dimension):
                raise ValueError(f"Layer {index} KV geometry disagrees with the layout")
        self.layout = layout
        self.blocks = nn.ModuleList(blocks)
        self.ple = ple
        self.prefill_mode = False

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Run the chunk.

        Args:
            inputs: Tensors in ``layout.input_names`` order.

        Returns:
            Tensors in ``layout.prefill_outputs`` order in prefill mode, otherwise in
            ``layout.step_outputs`` order.
        """
        layout = self.layout
        values = dict(zip(layout.input_names, inputs, strict=True))
        embedded = values["embedded"]
        ple = self.ple(embedded, values["token_identity"])
        hidden = embedded if layout.start == 0 else values["hidden"]
        # Consumers attend over their donor's cache followed by the donor's fresh K/V,
        # exactly as the donor itself does.
        shared = {
            layer: (
                torch.cat((values[f"key_{layer}"], values[f"fresh_key_{layer}"]), dim=-1),
                torch.cat((values[f"value_{layer}"], values[f"fresh_value_{layer}"]), dim=-1),
            )
            for layer in layout.external_donors
        }
        donors = layout.donors
        fresh: list[Tensor] = []
        for offset, block in enumerate(self.blocks):
            index = layout.start + offset
            prefix = PREFIX[block.layer_type]
            cos, sin = values[f"{prefix}_cos"], values[f"{prefix}_sin"]
            mask = values[f"{prefix}_mask"]
            if block.attention.shared:
                key, value = shared[donors[block.layer_type]]
            else:
                key, value = values[f"key_{index}"], values[f"value_{index}"]
            ple_input = ple[:, :, offset : offset + 1]
            hidden, new_key, new_value = block(hidden, ple_input, cos, sin, key, value, mask)
            if not block.attention.shared:
                fresh.extend((new_key, new_value))
                if block.stores_shared:
                    shared[index] = (
                        torch.cat((key, new_key), dim=-1),
                        torch.cat((value, new_value), dim=-1),
                    )
        if self.prefill_mode and "hidden" not in layout.prefill_outputs:
            return tuple(fresh)
        return (hidden, *fresh)


class HeadChunk(nn.Module):
    """Final norm, vocabulary head in row slices, and Gemma's logit softcap."""

    def __init__(
        self, layout: HeadLayout, norm: RMSNorm, parts: Sequence[nn.Module], softcap: float
    ) -> None:
        """Initialize from the final norm and the head's row slices.

        Args:
            layout: The chunk's interface.
            norm: The model's final norm.
            parts: One 1x1 projection per row slice, in vocabulary order.
            softcap: Final logit softcap.

        Raises:
            ValueError: If the slices disagree with the layout.
        """
        super().__init__()
        if len(parts) != layout.parts:
            raise ValueError("One projection per head slice is required")
        self.layout = layout
        self.norm = norm
        self.parts = nn.ModuleList(parts)
        self.softcap = softcap
        self.prefill_mode = False

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Return the softcapped logits of each row slice for ``hidden``."""
        normalized = self.norm(inputs[0])
        logits = [part(normalized) for part in self.parts]
        return tuple(torch.tanh(x / self.softcap) * self.softcap for x in logits)
