"""Google's QAT mobile codes, read as stored, and the lite conversion built from them.

A tiny synthetic checkpoint in Google's exact layout (key names, dtypes, packing,
``quantization_config``) with E4B's 42-layer topology. Unpacking and dequantization are
compared bit for bit with Transformers' ``gemma_quant``; the chunks' projections, clamps,
head and tables are checked against the codes; the whole lite conversion runs on it. Not a
quality gate: that is the frozen set on the phone.
"""

import copy
import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers import Gemma4TextConfig
from transformers.integrations import gemma_quant
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

from tiny import E4B_LAYER_TYPES, WINDOW
from within_core import export, package, qat
from within_core.convert import open_checkpoint, parser
from within_core.quantize import LITE_WEIGHT_FORMAT, QAT_SOURCES, ClampedProjection

VOCAB = 64
HIDDEN = 32
PLE = 8
TEXT = {
    "vocab_size": VOCAB,
    "vocab_size_per_layer_input": VOCAB,
    "hidden_size": HIDDEN,
    "intermediate_size": 64,
    "num_hidden_layers": 42,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "global_head_dim": 16,
    "hidden_size_per_layer_input": PLE,
    "num_kv_shared_layers": 18,
    "layer_types": E4B_LAYER_TYPES,
    "max_position_embeddings": 64,
    "sliding_window": WINDOW,
    "final_logit_softcapping": 30.0,
    "tie_word_embeddings": False,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {
            "rope_type": "proportional",
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.25,
        },
    },
}
# Google's quantization_config, verbatim from the checkpoint's config.json.
QUANTIZATION = {
    "module_quant_configs": {
        "^lm_head$": {"num_bits": 2},
        "audio_tower(?!.*lconv1d\\.linear_start)": {"num_bits": 2},
        "audio_tower\\.layers\\.\\d+\\.lconv1d\\.linear_start\\.": {"num_bits": 4},
        "language_model\\.embed_tokens$": {"num_bits": 2},
        "language_model\\.embed_tokens_per_layer$": {"num_bits": 2},
        "language_model\\.layers\\.\\d+\\.mlp\\.": {"num_bits": 4},
        "language_model\\.layers\\.\\d+\\.per_layer_input_gate$": {"num_bits": 8},
        "language_model\\.layers\\.\\d+\\.per_layer_projection$": {"num_bits": 8},
        "language_model\\.layers\\.\\d+\\.self_attn\\.": {"num_bits": 4},
        "vision_tower": {"num_bits": 8},
    },
    "modules_to_not_convert": [
        "model.vision_tower.patch_embedder",
        "model.audio_tower.subsample_conv_projection",
        "model.audio_tower.output_proj",
        "relative_k_proj",
        "model.embed_audio",
        "model.embed_vision",
        "per_layer_model_projection",
    ],
    "num_bits": 4,
    "quant_method": "gemma",
    "quantize_embeddings": True,
}
P = qat.TENSOR_PREFIX
STUBS = ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
Change = Callable[[dict[str, torch.Tensor], dict], None]


def quantized(
    generator: torch.Generator, out: int, columns: int, bits: int, prefix: str, tensors: dict
) -> None:
    """Add Google's layout of one projection: packed codes, F32 scales, activation scales."""
    if bits == 8:
        tensors[prefix + ".weight"] = torch.randint(
            -128, 128, (out, columns), dtype=torch.int8, generator=generator
        )
    else:
        tensors[prefix + ".weight"] = torch.randint(
            0, 256, (out, columns * bits // 8), dtype=torch.uint8, generator=generator
        )
    tensors[prefix + ".weight_scale"] = torch.exp(
        torch.empty(out, 1).uniform_(-7, -3, generator=generator)
    )
    tensors[prefix + ".input_activation_scale"] = torch.rand((), generator=generator)
    tensors[prefix + ".output_activation_scale"] = torch.rand((), generator=generator)


def write_qat_checkpoint(folder: Path, change: Change | None = None) -> dict:
    """Write a tiny QAT checkpoint folder; return its pinned reference description."""
    folder.mkdir(parents=True)
    generator = torch.Generator().manual_seed(1001)
    config = Gemma4TextConfig(**TEXT)
    tensors: dict[str, torch.Tensor] = {}
    quantization = copy.deepcopy(QUANTIZATION)

    def bf16(*shape: int) -> torch.Tensor:
        return (1 + 0.1 * torch.randn(*shape, generator=generator)).bfloat16()

    for index in range(42):
        prefix = f"{P}layers.{index}."
        with torch.device("meta"):
            layer = Gemma4TextDecoderLayer(config, index)
        linears = {
            name: (module.out_features, module.in_features)
            for name, module in layer.named_modules()
            if isinstance(module, nn.Linear)
        }
        if layer.self_attn.is_kv_shared_layer:  # Google's file keeps the unused K/V projections
            width = TEXT[
                "global_head_dim" if E4B_LAYER_TYPES[index] == "full_attention" else "head_dim"
            ]
            linears.update({f"self_attn.{p}_proj": (2 * width, HIDDEN) for p in "kv"})
        for name, (out, columns) in linears.items():
            bits = 8 if name.startswith("per_layer") else 4
            quantized(generator, out, columns, bits, prefix + name, tensors)
        for name, value in layer.state_dict().items():
            if name.removesuffix(".weight") not in linears:
                tensors[prefix + name] = bf16(*value.shape)
        for kind in "kv":
            tensors[f"{prefix}self_attn.{kind}_cache_scale"] = torch.rand((), generator=generator)
    for name, columns, blocks in (
        ("embed_tokens", HIDDEN, 1),
        ("embed_tokens_per_layer", 42 * PLE, 42),
    ):
        tensors[f"{P}{name}.embedding_quantized"] = torch.randint(
            0, 256, (VOCAB, columns // 4), dtype=torch.uint8, generator=generator
        )
        tensors[f"{P}{name}.embedding_scale"] = torch.rand(VOCAB, blocks, generator=generator) / 16
    quantized(generator, VOCAB, HIDDEN, 2, "lm_head", tensors)
    tensors[P + "norm.weight"] = bf16(HIDDEN)
    tensors[P + "per_layer_projection_norm.weight"] = bf16(PLE)
    tensors[P + "per_layer_model_projection.weight"] = bf16(42 * PLE, HIDDEN)
    if change:
        change(tensors, quantization)
    save_file(tensors, folder / "model.safetensors")
    (folder / "config.json").write_text(
        json.dumps({"text_config": TEXT, "quantization_config": quantization})
    )
    for name in STUBS:
        (folder / name).write_text(f"{name}\n")
    files = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(folder.iterdir())
    }
    return {
        "source": "tiny-qat",
        "revision": "0",
        "files": files,
        "quantization_config": quantization,
        "text_config": TEXT,
    }


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> qat.QATCheckpoint:
    """A tiny QAT checkpoint."""
    folder = tmp_path_factory.mktemp("qat") / "checkpoint"
    return qat.QATCheckpoint(folder, write_qat_checkpoint(folder))


def test_unpack_matches_gemma_quant() -> None:
    packed = torch.randint(0, 256, (5, 12), dtype=torch.uint8)
    assert np.array_equal(
        qat.unpack(packed.numpy(), 4, 24), gemma_quant._unpack_int4(packed, 24).numpy()
    )
    assert np.array_equal(
        qat.unpack(packed.numpy(), 2, 48), gemma_quant._unpack_int2(packed, 48).numpy()
    )
    every_byte = np.arange(256, dtype=np.uint8)[None]
    assert set(np.unique(qat.unpack(every_byte, 2, 1024))) == {-2, -1, 0, 1}


@pytest.mark.parametrize(
    ("name", "bits"),
    [
        ("layers.5.self_attn.q_proj", 4),
        ("layers.30.mlp.down_proj", 4),
        ("layers.3.per_layer_input_gate", 8),
        ("layers.3.per_layer_projection", 8),
        ("lm_head", 2),
    ],
)
def test_projections_dequantize_exactly_as_gemma_quant(
    checkpoint: qat.QATCheckpoint, name: str, bits: int
) -> None:
    codes, scale, stored = checkpoint.linear(name)
    assert stored == bits
    google = gemma_quant.QuantizedLinear(codes.shape[1], codes.shape[0], num_bits=bits)
    with (
        safe_open(checkpoint.path("model.safetensors"), framework="pt") as tensors,
        torch.no_grad(),
    ):
        google.weight.copy_(tensors.get_tensor(checkpoint.key(name) + ".weight"))
        google.weight_scale.copy_(scale)
    reference = google._dequantize_weights()
    assert reference.dtype == torch.float32
    assert torch.equal(codes.float() * scale, reference)
    if name != "lm_head":
        index, _, module = name.removeprefix("layers.").partition(".")
        layer = checkpoint.layer(int(index), torch.float32)
        assert torch.equal(layer.get_submodule(module).weight, reference)


def test_shared_layers_drop_the_unused_projections(checkpoint: qat.QATCheckpoint) -> None:
    shared = checkpoint.layer(30)
    assert shared.self_attn.is_kv_shared_layer and not hasattr(shared.self_attn, "k_proj")
    assert len(checkpoint.activation_scales("layers.0.mlp.up_proj")) == 2


def expected_weight(checkpoint: qat.QATCheckpoint, name: str) -> torch.Tensor:
    """FP32 code times the FP16 scale, rounded once to FP16: what the palette must give."""
    codes, scale, _ = checkpoint.linear(name)
    return (codes.float() * scale.half().float()).half()


@pytest.mark.parametrize(("start", "end"), [(0, 6), (24, 30)])
def test_lite_chunk_projections_are_googles_codes(
    checkpoint: qat.QATCheckpoint, start: int, end: int
) -> None:
    layout = export.plan(checkpoint.config, 16)[start // 6]
    chunk = export.build_layer_chunk(checkpoint, layout)
    export.require_lite_compressed(chunk)
    assert not nn.utils.parametrize.is_parametrized(chunk.ple.projection)  # FP16, as Google left it
    clamped = [(n, m) for n, m in chunk.named_modules() if isinstance(m, ClampedProjection)]
    assert len(clamped) == (end - start) * (9 if start < 24 else 7)  # shared-KV blocks have no k, v
    for name, projection in clamped:
        block, module = name.removeprefix("blocks.").split(".", 1)
        source = f"layers.{start + int(block)}.{QAT_SOURCES[module]}"
        codes, scale, bits = checkpoint.linear(source)
        weight = projection.conv.parametrizations.weight[0]
        if bits == 4:
            assert weight.indices.dtype == torch.uint8
            assert torch.equal(weight.indices[:, :, 0, 0].to(torch.int8) - 8, codes)
            assert torch.equal(
                weight.lut[:, 0, 0, 0, :, 0],
                torch.arange(-8, 8).half().expand(len(codes) // 16, 16),
            )
        else:
            assert bits == 8 and weight.input_dtype == torch.int8
            assert torch.equal(weight.quantized_data[:, :, 0, 0], codes)
        assert torch.equal(weight.scale[:, :, 0, 0], scale.half())
        assert torch.equal(projection.conv.weight[:, :, 0, 0], expected_weight(checkpoint, source))
        # The clamps: each side to [-128 s, 127 s] of its calibrated scale, without rounding.
        in_scale, out_scale = checkpoint.activation_scales(source)
        x = (torch.randn(2, codes.shape[1], 1, 1) * 400 * max(in_scale, 1e-3)).half()
        bounded = x.clamp(-128 * in_scale, 127 * in_scale) if in_scale > 0 else x
        expected = projection.conv(bounded)
        expected = expected.clamp(-128 * out_scale, 127 * out_scale) if out_scale > 0 else expected
        assert torch.equal(projection(x), expected)


def test_projections_reading_one_tensor_share_one_input_clamp(tmp_path: Path) -> None:
    def one_range(tensors: dict, _: dict) -> None:  # Google's file gives q, k and v one input range
        for kind in "kv":
            tensors[f"{P}layers.0.self_attn.{kind}_proj.input_activation_scale"] = tensors[
                f"{P}layers.0.self_attn.q_proj.input_activation_scale"
            ].clone()

    folder = tmp_path / "checkpoint"
    checkpoint = qat.QATCheckpoint(folder, write_qat_checkpoint(folder, one_range))
    chunk = export.build_layer_chunk(checkpoint, export.plan(checkpoint.config, 16)[0])
    attention = chunk.blocks[0].attention

    class QKV(nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return attention.q(x), attention.k(x), attention.v(x)

    x = torch.randn(1, HIDDEN, 1, 16, dtype=torch.float16)
    graph = torch.export.export(QKV(), (x,)).graph
    selects = [n for n in graph.nodes if n.op == "call_function" and "where" in str(n.target)]
    assert len(selects) == 2 * (
        1 + 3
    )  # one shared input clamp and three output clamps, two ops each


def test_head_is_palettes_of_googles_two_bit_codes(checkpoint: qat.QATCheckpoint) -> None:
    layout = export.plan(checkpoint.config, 16)[-1]
    chunk = export.build_head_chunk(checkpoint, layout)
    export.require_lite_compressed(chunk)
    codes, scale, bits = checkpoint.linear("lm_head")
    assert bits == 2 and len(chunk.parts) == layout.parts
    weights = [part.parametrizations.weight[0] for part in chunk.parts]
    assert torch.equal(
        torch.cat([w.indices for w in weights])[:, :, 0, 0].to(torch.int8) - 2, codes
    )
    assert all(torch.equal(w.lut.flatten(), torch.tensor([-2.0, -1, 0, 1]).half()) for w in weights)
    assert torch.equal(torch.cat([w.scale for w in weights])[:, :, 0, 0], scale.half())
    head = torch.cat([part.weight for part in chunk.parts])[:, :, 0, 0]
    assert torch.equal(head, expected_weight(checkpoint, "lm_head"))


def test_int2_tables_match_gemma_quant_embedding(
    checkpoint: qat.QATCheckpoint, tmp_path: Path
) -> None:
    descriptors = qat.write_tables(checkpoint, tmp_path / "tables", block_rows=10)
    assert json.loads((tmp_path / "tables" / qat.TABLES_RECORD).read_text()) == descriptors
    for key, name in qat.TABLES.items():
        descriptor = descriptors[key]
        width = descriptor["columns"]
        packed, scales = checkpoint.table(name, 0, VOCAB)
        assert (tmp_path / "tables" / descriptor["file"]).read_bytes() == packed.tobytes()
        scales_file = tmp_path / "tables" / descriptor["scales"]["file"]
        assert scales_file.read_bytes() == scales.astype("<f4").tobytes()
        assert descriptor["scales"]["columnsPerScale"] == width // scales.shape[1]
        embed_scale = math.sqrt(HIDDEN if key == "embeddings" else PLE)
        google = gemma_quant.QuantizedEmbedding(
            VOCAB, width, torch.float32, embed_scale, num_bits=2
        )
        with torch.no_grad():
            google.embedding_quantized.copy_(torch.from_numpy(packed))
            google.embedding_scale = nn.Parameter(torch.from_numpy(scales), requires_grad=False)
        for token in (0, 7, VOCAB - 1):
            row = google.weight[token].numpy()
            assert np.array_equal(qat.dequantize(packed[token], scales[token], 2, width), row)
            assert np.array_equal(
                qat.lookup(tmp_path / "tables", descriptor, token), row.astype(np.float16)
            )
            scaled = google(torch.tensor([token]))[0].numpy().astype(np.float16)
            assert np.array_equal(
                qat.lookup(tmp_path / "tables", descriptor, token, embed_scale), scaled
            )


@pytest.mark.parametrize("change", ["int4_gate", "eight_bit_mlp", "other_config"])
def test_validation_rejects_a_checkpoint_that_differs(tmp_path: Path, change: str) -> None:
    def int4_gate(tensors: dict, _: dict) -> None:  # an I8 projection stored as packed 4-bit codes
        key = P + "layers.3.per_layer_input_gate.weight"
        tensors[key] = torch.zeros(
            tensors[key].shape[0], tensors[key].shape[1] // 2, dtype=torch.uint8
        )

    def eight_bit_mlp(_: dict, quantization: dict) -> None:  # claims 8 bits for 4-bit MLP codes
        quantization["module_quant_configs"]["language_model\\.layers\\.\\d+\\.mlp\\."] = {
            "num_bits": 8
        }

    folder = tmp_path / "checkpoint"
    if change == "other_config":
        reference = write_qat_checkpoint(folder)
        reference["quantization_config"] = dict(reference["quantization_config"], num_bits=8)
    else:
        reference = write_qat_checkpoint(
            folder, {"int4_gate": int4_gate, "eight_bit_mlp": eight_bit_mlp}[change]
        )
    with pytest.raises(ValueError):
        qat.QATCheckpoint(folder, reference)


class Converted(NamedTuple):
    """Work folder of a tiny lite conversion."""

    root: Path
    checkpoint: qat.QATCheckpoint


CONTEXT = 512
PREFILL = 8
BASE_URL = "https://example.com/models/tiny-lite/"


@pytest.fixture(scope="module")
def converted(tmp_path_factory: pytest.TempPathFactory, checkpoint: qat.QATCheckpoint) -> Converted:
    """Run every step of the lite conversion once."""
    root = tmp_path_factory.mktemp("lite")
    for index in range(len(export.plan(checkpoint.config, CONTEXT))):
        export.export_chunk(checkpoint, index, root / "export", CONTEXT, PREFILL)
    export.finalize(root / "export", checkpoint.config, CONTEXT, PREFILL)
    qat.write_tables(checkpoint, root / "tables")
    package.assemble(
        root / "export", root / "tables", checkpoint, root / "release", "tiny-lite-01", BASE_URL
    )
    return Converted(root, checkpoint)


def test_lite_contract_and_runtime_describe_the_package(converted: Converted) -> None:
    contract = json.loads((converted.root / "export" / export.CHUNK_CONTRACT).read_text())
    assert contract["weightFormat"] == LITE_WEIGHT_FORMAT
    assert contract["prefillLength"] == PREFILL and contract["contextLimit"] == CONTEXT
    assert [c["file"] for c in contract["chunks"]] == [export.chunk_file(i) for i in range(8)]
    assert all(f"prefill_{PREFILL}" in c["functions"] for c in contract["chunks"][:4])
    exported = json.loads((converted.root / "export" / export.RUNTIME_CONFIG).read_text())
    assert exported["schemaVersion"] == 1 and exported["prefillLength"] == PREFILL
    runtime = json.loads(
        (converted.root / "release" / "package" / export.RUNTIME_CONFIG).read_text()
    )
    assert runtime["schemaVersion"] == 2
    assert runtime["embeddings"] == {
        "file": "embeddings.int2",
        "format": "int2",
        "rows": VOCAB,
        "columns": HIDDEN,
        "scales": {"file": "embeddings.scales.f32", "columnsPerScale": HIDDEN},
    }
    assert runtime["perLayerEmbeddings"]["scales"]["columnsPerScale"] == PLE
    assert {
        k: v
        for k, v in runtime.items()
        if k not in ("schemaVersion", "embeddings", "perLayerEmbeddings")
    } == {
        k: v
        for k, v in exported.items()
        if k not in ("schemaVersion", "embeddings", "perLayerEmbeddings")
    }


def test_lite_package_lists_the_tables_and_matches_its_catalog(converted: Converted) -> None:
    release = converted.root / "release"
    data = json.loads((release / "package" / package.INSTALLED_MANIFEST).read_text())
    assert data == json.loads((release / package.CATALOG).read_text())
    paths = {row["path"] for row in data["files"]}
    assert {
        "embeddings.int2",
        "embeddings.scales.f32",
        "perLayerEmbeddings.int2",
        "perLayerEmbeddings.scales.f32",
        "gemma-runtime.json",
        "chunk-contract.json",
    } <= paths
    assert not any(path.endswith(".f16") for path in paths)
    assert data["contextLimit"] == CONTEXT
    package.validate(data)


def test_finalize_rejects_mixed_weight_formats(converted: Converted, tmp_path: Path) -> None:
    other = tmp_path / "export"
    other.mkdir()
    for index in range(8):
        record = json.loads((converted.root / "export" / export.record_file(index)).read_text())
        if index == 2:
            record["weightFormat"] = "something else"
        (other / export.record_file(index)).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="weight formats"):
        export.finalize(other, converted.checkpoint.config, CONTEXT, PREFILL)
    with pytest.raises(ValueError, match="different plan"):
        export.finalize(
            converted.root / "export", converted.checkpoint.config, CONTEXT, PREFILL + 8
        )


def test_authored_lite_chunks_hold_palettes(checkpoint: qat.QATCheckpoint) -> None:
    layouts = export.plan(checkpoint.config, 16)
    text = str(export.author(export.build_layer_chunk(checkpoint, layouts[0]), 4))
    # 7 palettes x 6 layers x (step_1, prefill_4), each a 4-bit index tensor and an FP16 lut.
    assert text.count("coreai.lut_to_dense") == 84
    assert "(tensor<64x32x1x1xui4>, tensor<4x1x1x1x16x1xf16>" in text
    assert "xsi4>" not in text
    head = str(export.author(export.build_head_chunk(checkpoint, layouts[-1]), 4))
    assert "xui2>" in head and "coreai.lut_to_dense" in head


def test_catalog_lists_one_package_per_tier(converted: Converted, tmp_path: Path) -> None:
    lite = converted.root / "release" / "package"
    data = package.catalog([("liteFast", lite)], "liteFast", tmp_path / "ModelCatalog.json")
    assert data["catalogVersion"] == 2 and data["defaultTier"] == "liteFast"
    assert data["packages"][0]["manifest"] == json.loads(
        (lite / package.INSTALLED_MANIFEST).read_text()
    )
    assert json.loads((tmp_path / "ModelCatalog.json").read_text()) == data
    with pytest.raises(package.PackageError, match="defaultTier"):
        package.catalog([("liteFast", lite)], "full")
    with pytest.raises(package.PackageError, match="tiers repeat"):
        package.catalog([("liteFast", lite), ("liteFast", lite)], "liteFast")


def test_command_line_recognises_the_qat_checkpoint(converted: Converted, tmp_path: Path) -> None:
    folder = converted.checkpoint.folder
    assert qat.is_qat_checkpoint(folder) and not qat.is_qat_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="qat-mobile"):  # the QAT path, against the pinned file
        open_checkpoint(folder)
    arguments = parser().parse_args(
        ["chunk", "0", "--checkpoint", "c", "--output", "o", "--prefill", "128"]
    )
    assert arguments.prefill == 128 and arguments.context == export.CONTEXT
    arguments = parser().parse_args(["finalize", "--output", "o"])
    assert arguments.prefill == export.PREFILL_LENGTH
    arguments = parser().parse_args(
        [
            "catalog",
            "--entry",
            "full",
            "a",
            "--entry",
            "liteFast",
            "b",
            "--default",
            "liteFast",
            "--output",
            "c",
        ]
    )
    assert arguments.entry == [["full", "a"], ["liteFast", "b"]]
