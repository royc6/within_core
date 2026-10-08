"""Build, quantize and author the chunks of the package as Core AI assets.

Pipeline position: after :mod:`within_core.checkpoint`, before :mod:`within_core.package`.
:func:`export_chunk` builds one chunk from the checkpoint, compresses it, traces each of its
entrypoints with ``torch.export`` and serializes the authored Core AI program; run it once
per chunk, each in its own process, so that only one chunk's weights are ever in memory.
:func:`finalize` then writes ``chunk-contract.json`` (every chunk's I/O names and shapes) and
``gemma-runtime.json`` (what the host needs to drive them).

The model is never executed here: the export only authors and serializes graphs, and the
Core AI runtime compiles them on the device.

The plan: the 42 decoder layers in 7 chunks of 6, a balance between what the Neural Engine
compiler accepts per asset, memory, and per-call overhead; then the vocabulary head as an
8th chunk of its own (together with the last layers it is too large to compile), split into
16 row slices; host-held KV with a 16384-token context and 512-slot sliding-attention rings;
prefill in blocks of 16 tokens. The lite packages (release 1.1) use the same plan with a
4096- or 32000-token context, prefill blocks of 128 tokens, and Google's QAT codes
(:mod:`within_core.qat`) in place of our quantization.
"""

import functools
import gc
import itertools
import json
import logging
from pathlib import Path

import torch
from coreai.authoring.asset import AIProgram
from coreai_models.export.mlir_ops import register_custom_torch_lowering
from coreai_torch import TorchConverter, get_decomp_table
from torch import Tensor, nn
from torch.export import ExportedProgram
from torch.nn.utils.parametrize import is_parametrized
from transformers import Gemma4TextConfig

from within_core.checkpoint import LocalCheckpoint
from within_core.chunks import ChunkLayout, DecoderBlock, HeadChunk, HeadLayout, LayerChunk, Layout
from within_core.layers import PerLayerProjection, RMSNorm
from within_core.qat import QATCheckpoint
from within_core.quantize import (
    LITE_WEIGHT_FORMAT,
    WEIGHT_FORMAT,
    quantize_head,
    quantize_head_qat,
    quantize_projections,
    quantize_projections_qat,
    require_compressed,
)
from within_core.tables import EMBEDDINGS, PER_LAYER_EMBEDDINGS

Checkpoint = LocalCheckpoint | QATCheckpoint

BOUNDARIES = (0, 6, 12, 18, 24, 30, 36, 42)
HEAD_PARTS = 16
PREFILL_LENGTH = 16
MAX_PREFILL_LENGTH = 256
CONTEXT = 16384
CONTRACT = "gemma4-chunked-hostkv-v1"
CHUNK_CONTRACT = "chunk-contract.json"
RUNTIME_CONFIG = "gemma-runtime.json"

logger = logging.getLogger(__name__)


def chunk_file(index: int) -> str:
    """Return the asset name of chunk ``index``."""
    return f"chunk-{index:02d}.aimodel"


def record_file(index: int) -> str:
    """Return the name of the record that marks chunk ``index`` as exported."""
    return f"chunk-{index:02d}.json"


def plan(config: Gemma4TextConfig, context: int = CONTEXT) -> list[Layout]:
    """Return the layout of every chunk: the layer chunks, then the head chunk.

    Args:
        config: The text configuration of the model.
        context: Context limit (slots of a full-attention cache).

    Returns:
        One layout per chunk, in execution order.

    Raises:
        ValueError: If the boundaries do not cover the decoder.
    """
    if BOUNDARIES[-1] != config.num_hidden_layers:
        raise ValueError("The chunk boundaries do not cover the decoder")
    layers: list[Layout] = [
        ChunkLayout.from_config(config, start, end, context)
        for start, end in itertools.pairwise(BOUNDARIES)
    ]
    head = HeadLayout(
        layers=config.num_hidden_layers,
        parts=HEAD_PARTS,
        rows=config.vocab_size // HEAD_PARTS,
        hidden_size=config.hidden_size,
    )
    return [*layers, head]


def weight_format(checkpoint: Checkpoint) -> str:
    """Return the ``weightFormat`` of a package converted from ``checkpoint``."""
    return LITE_WEIGHT_FORMAT if isinstance(checkpoint, QATCheckpoint) else WEIGHT_FORMAT


def runtime_config(
    config: Gemma4TextConfig, context: int = CONTEXT, prefill_length: int = PREFILL_LENGTH
) -> dict[str, object]:
    """Return the contents of ``gemma-runtime.json``.

    It gives the host the model geometry, the RoPE parameters it computes cosines and sines
    from, the cache sizes, the prefill block length and the FP16 tables it reads
    embeddings from (:mod:`within_core.package` replaces the table entries for a lite
    package, whose tables are 2-bit).
    """
    rope = config.rope_parameters
    return {
        "schemaVersion": 1,
        "hiddenSize": config.hidden_size,
        "vocabularySize": config.vocab_size,
        "layerTypes": list(config.layer_types),
        "sharedLayerCount": config.num_kv_shared_layers,
        "perLayerInputSize": config.hidden_size_per_layer_input,
        "keyValueHeads": config.num_key_value_heads,
        "localHeadDimension": config.head_dim,
        "globalHeadDimension": config.global_head_dim,
        "localTheta": rope["sliding_attention"]["rope_theta"],
        "globalTheta": rope["full_attention"]["rope_theta"],
        "globalRotaryFraction": rope["full_attention"]["partial_rotary_factor"],
        "slidingWindow": config.sliding_window,
        "contextLimit": context,
        "prefillLength": prefill_length,
        "embeddings": {
            "file": EMBEDDINGS,
            "format": "float16",
            "rows": config.vocab_size,
            "columns": config.hidden_size,
        },
        "perLayerEmbeddings": {
            "file": PER_LAYER_EMBEDDINGS,
            "format": "float16",
            "rows": config.vocab_size,
            "columns": config.num_hidden_layers * config.hidden_size_per_layer_input,
        },
    }


def build_layer_chunk(checkpoint: Checkpoint, layout: ChunkLayout) -> LayerChunk:
    """Build a compressed layer chunk, reading only its own layers from the checkpoint.

    Each layer is compressed as soon as it is built, so only one layer is dense at a time.
    From a QAT checkpoint the projections take Google's codes and clamps, and the per-layer
    model projection stays FP16 as Google left it.
    """
    config = checkpoint.config
    lite = isinstance(checkpoint, QATCheckpoint)
    ple_dim = config.hidden_size_per_layer_input
    ple = PerLayerProjection(
        checkpoint.row_range(
            "per_layer_model_projection.weight", layout.start * ple_dim, layout.end * ple_dim
        ),
        checkpoint.tensor("per_layer_projection_norm.weight"),
        eps=config.rms_norm_eps,
        layers=layout.end - layout.start,
        projection_scale=config.hidden_size**-0.5,
        input_scale=2**-0.5,
    )
    if not lite:
        quantize_projections(ple)
    blocks = []
    for index in range(layout.start, layout.end):
        block = DecoderBlock(checkpoint.layer(index))
        if lite:
            quantize_projections_qat(
                block,
                lambda module, i=index: checkpoint.linear(f"layers.{i}.{module}"),
                lambda module, i=index: checkpoint.activation_scales(f"layers.{i}.{module}"),
            )
        else:
            quantize_projections(block)
        blocks.append(block)
        gc.collect()
    return LayerChunk(layout, blocks, ple).eval()


def build_head_chunk(checkpoint: Checkpoint, layout: HeadLayout) -> HeadChunk:
    """Build the head chunk; the head is read and compressed one row block at a time.

    The BF16 checkpoint's head is tied to the embedding and quantized here; the QAT
    checkpoint ships its own 2-bit head, written as palettes.
    """
    config = checkpoint.config
    norm = RMSNorm(checkpoint.tensor("norm.weight"), config.rms_norm_eps)
    if isinstance(checkpoint, QATCheckpoint):
        read_codes = functools.partial(checkpoint.linear, "lm_head")
        parts = quantize_head_qat(read_codes, config.vocab_size, config.hidden_size, layout.parts)
    else:
        read_rows = functools.partial(checkpoint.row_range, "embed_tokens.weight")
        parts = quantize_head(read_rows, config.vocab_size, config.hidden_size, layout.parts)
    return HeadChunk(layout, norm, parts, config.final_logit_softcapping).eval()


def require_lite_compressed(chunk: LayerChunk | HeadChunk) -> None:
    """Raise ``ValueError`` unless every projection but the per-layer model projection is
    compressed."""
    convs = [(n, c) for n, c in chunk.named_modules() if isinstance(c, nn.Conv2d)]
    dense = [name for name, conv in convs if not is_parametrized(conv) and name != "ple.projection"]
    if dense:
        raise ValueError(f"Projections escaped compression: {dense}")


def example_inputs(layout: Layout, length: int) -> tuple[Tensor, ...]:
    """Return FP16 example inputs for tracing a call of ``length`` positions.

    Values do not shape the graph; cosines of one and sines of zero make a valid rotation.
    """
    return tuple(
        (torch.ones if name.endswith("_cos") else torch.zeros)(*shape, dtype=torch.float16)
        for name, shape in layout.shapes(length).items()
    )


def trace(chunk: LayerChunk | HeadChunk, inputs: tuple[Tensor, ...]) -> ExportedProgram:
    """Trace one entrypoint of ``chunk`` with ``torch.export``.

    Raises:
        ValueError: If the traced program mutates an input or buffer (KV is host-held).
    """
    decompositions = get_decomp_table()
    # Keep layer norm whole: RMSNorm relies on its accumulation-safe implementation.
    decompositions.pop(torch.ops.aten.native_layer_norm.default, None)
    with torch.no_grad():
        program = torch.export.export(chunk, (inputs,)).run_decompositions(decompositions)
    signature = program.graph_signature
    if signature.user_inputs_to_mutate or signature.buffers_to_mutate:
        raise ValueError("A chunk must not mutate its inputs or buffers")
    return program


def prune_unused_constants(program: AIProgram, entrypoints: list[str]) -> dict[str, int]:
    """Remove weight placeholders that no operation of an entrypoint uses.

    ``torch.export`` drops computations that a headless prefill does not need (for
    example the last layer's MLP when only fresh K/V are returned), but the converter still
    emits their weights. Only top-level ``coreai.constant`` operations without users are
    removed; computations and nested constants are left alone.

    Args:
        program: The authored program, edited in place.
        entrypoints: Names of the entrypoints to prune.

    Returns:
        The number of constants removed per entrypoint.

    Raises:
        ValueError: If an entrypoint is missing (checked before any graph is edited).
    """
    graphs = {name: program.get_graph(name) for name in entrypoints}
    missing = [name for name, graph in graphs.items() if graph is None]
    if missing:
        raise ValueError(f"Missing entrypoints: {missing}")
    removed = {}
    for name, graph in graphs.items():
        count = 0
        for op in list(graph.regions[0].blocks[0].operations):
            if (
                op.operation.name == "coreai.constant"
                and len(op.results) == 1
                and not any(op.results[0].uses)
            ):
                op.operation.erase()
                count += 1
        graph.operation.verify()
        removed[name] = count
    return removed


def author(chunk: LayerChunk | HeadChunk, prefill_length: int = PREFILL_LENGTH) -> AIProgram:
    """Author every entrypoint of ``chunk`` into one uncompiled Core AI program.

    All entrypoints share the chunk's weights. The converter runs in release mode, so the
    program records operation IDs rather than source paths.

    Raises:
        ValueError: If an authored entrypoint's I/O names differ from the layout's.
    """
    layout = chunk.layout
    converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
    register_custom_torch_lowering(converter)
    entrypoints = layout.entrypoints(prefill_length)
    try:
        for name, prefill, length in entrypoints:
            chunk.prefill_mode = prefill
            program = trace(chunk, example_inputs(layout, length))
            outputs = layout.prefill_outputs if prefill else layout.step_outputs
            converter.add_exported_program(
                program,
                input_names=list(layout.input_names),
                output_names=list(outputs),
                entrypoint_name=name,
            )
            del program
    finally:
        chunk.prefill_mode = False
    authored = converter.to_coreai()
    prune_unused_constants(authored, [name for name, _, _ in entrypoints])
    for name, prefill, _ in entrypoints:
        graph = authored.get_graph(name)
        inputs = [attributes["coreai.name"].value for attributes in graph.arg_attrs]
        outputs = [attributes["coreai.name"].value for attributes in graph.res_attrs]
        expected = layout.prefill_outputs if prefill else layout.step_outputs
        if inputs != list(layout.input_names) or outputs != list(expected):
            raise ValueError(f"{name} was authored with I/O {inputs} -> {outputs}")
    return authored


def export_chunk(
    checkpoint: Checkpoint,
    index: int,
    output: Path,
    context: int = CONTEXT,
    prefill_length: int = PREFILL_LENGTH,
) -> Path:
    """Export chunk ``index`` of the plan into ``output``.

    Writes ``chunk-<index>.aimodel`` and, once it is complete, the record
    ``chunk-<index>.json`` holding the chunk's contract entry and the weight format.

    Args:
        checkpoint: The verified checkpoint (BF16, or Google's QAT one for a lite package).
        index: Chunk index in the plan (``0`` to ``7``).
        output: Export folder, shared by all chunks of one export.
        context: Context limit.
        prefill_length: Tokens per prefill call (2 to 256).

    Returns:
        The path of the asset.

    Raises:
        FileExistsError: If the chunk was already exported into ``output``.
        ValueError: If ``prefill_length`` is out of range.
    """
    if not 2 <= prefill_length <= MAX_PREFILL_LENGTH:
        raise ValueError(f"prefill_length must be 2 to {MAX_PREFILL_LENGTH}")
    layout = plan(checkpoint.config, context)[index]
    asset, record = output / chunk_file(index), output / record_file(index)
    if asset.exists() or record.exists():
        raise FileExistsError(f"Chunk {index} already exists in {output}")
    output.mkdir(parents=True, exist_ok=True)
    logger.info("chunk %d: building and compressing", index)
    if isinstance(layout, HeadLayout):
        chunk = build_head_chunk(checkpoint, layout)
    else:
        chunk = build_layer_chunk(checkpoint, layout)
    if isinstance(checkpoint, QATCheckpoint):
        require_lite_compressed(chunk)
    else:
        require_compressed(chunk)
    gc.collect()
    logger.info("chunk %d: authoring", index)
    authored = author(chunk, prefill_length)
    del chunk
    gc.collect()
    logger.info("chunk %d: saving %s", index, asset)
    authored.save_asset(asset)
    entry = layout.contract(asset.name, prefill_length)
    _write_json(record, {"contract": entry, "weightFormat": weight_format(checkpoint)})
    return asset


def finalize(
    output: Path,
    config: Gemma4TextConfig,
    context: int = CONTEXT,
    prefill_length: int = PREFILL_LENGTH,
) -> None:
    """Write ``chunk-contract.json`` and ``gemma-runtime.json`` once every chunk is exported.

    Raises:
        FileNotFoundError: If a chunk has not been exported.
        ValueError: If a chunk was exported under a different plan or weight format.
    """
    layouts = plan(config, context)
    entries = []
    formats = set()
    for index, layout in enumerate(layouts):
        record = output / record_file(index)
        if not record.is_file():
            raise FileNotFoundError(f"Chunk {index} has not been exported to {output}")
        entry = layout.contract(chunk_file(index), prefill_length)
        recorded = json.loads(record.read_text())
        if recorded.get("contract") != entry:
            raise ValueError(f"Chunk {index} was exported under a different plan")
        formats.add(recorded.get("weightFormat"))
        entries.append(entry)
    if len(formats) != 1 or None in formats:
        raise ValueError(f"The chunks were exported in different weight formats: {formats}")
    contract = {
        "contract": CONTRACT,
        "contextLimit": context,
        "slidingCacheLength": config.sliding_window,
        "prefillLength": prefill_length,
        "firstSharedLayer": config.num_hidden_layers - config.num_kv_shared_layers,
        "weightFormat": formats.pop(),
        "chunks": entries,
    }
    _write_json(output / CHUNK_CONTRACT, contract)
    _write_json(output / RUNTIME_CONFIG, runtime_config(config, context, prefill_length))


def _write_json(path: Path, value: object) -> None:
    """Write ``value`` as indented JSON to a new file (never replacing one)."""
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
