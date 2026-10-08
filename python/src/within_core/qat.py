"""Verified, offline reads from Google's quantization-aware (QAT) mobile checkpoint.

Pipeline position: the lite packages (release 1.1) are converted from
``google/gemma-4-E4B-it-qat-mobile-transformers`` instead of the BF16 checkpoint. That file
ships the integer codes and F32 scales of Google's mobile quantization: 4-bit codes for the
attention and MLP projections, 8-bit for the per-layer-embedding gate and projection, 2-bit
for the vocabulary head and both embedding tables, each with one scale per output row (the
tables: per block of columns), and a calibrated INT8 range (an "SRQ" scale) for the input and
output of every projection. This module reads the codes as stored and never re-quantizes;
:mod:`within_core.quantize` writes them into the chunks as palettes. Unpacking and
dequantization follow Transformers 5.12.1 ``integrations/gemma_quant.py`` bit for bit (the
tests compare them).

The checkpoint folder must hold the files pinned in ``reference-config-qat.json`` byte for
byte; each file's size and SHA-256 are checked before its first use, and nothing is ever
downloaded. Text model only.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch import Tensor, nn
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
from transformers.quantizers.quantizers_utils import should_convert_module

from within_core.checkpoint import CONFIG, TENSOR_PREFIX, WEIGHTS, sha256

REFERENCE = Path(__file__).with_name("reference-config-qat.json")
HEAD = "lm_head"
# Stored beside each projection; read for the clamps, never loaded into a layer.
SIDE_TENSORS = (
    ".weight_scale",
    ".input_activation_scale",
    ".output_activation_scale",
    ".k_cache_scale",
    ".v_cache_scale",
)
# Package table -> checkpoint table. Host tables, never in a graph.
TABLES = {"embeddings": "embed_tokens", "perLayerEmbeddings": "embed_tokens_per_layer"}
TABLE_BITS = 2
TABLES_RECORD = "tables.json"


def load_reference() -> dict:
    """Return the pinned QAT checkpoint description."""
    return json.loads(REFERENCE.read_text())


def is_qat_checkpoint(folder: Path) -> bool:
    """Return whether ``folder`` holds a Gemma QAT checkpoint (by its ``config.json``)."""
    path = Path(folder) / CONFIG
    if not path.is_file():
        return False
    quantization = json.loads(path.read_text()).get("quantization_config") or {}
    return quantization.get("quant_method") == "gemma"


def packed_width(columns: int, bits: int) -> int:
    """Return the stored width of a row of ``columns`` codes of ``bits`` bits."""
    return columns if bits == 8 else -(-columns * bits // 8)


def unpack(packed: np.ndarray, bits: int, columns: int) -> np.ndarray:
    """Return Google's codes as ``int8`` (``gemma_quant._unpack_int2`` / ``_unpack_int4``).

    Codes of 2 or 4 bits are packed ``8 // bits`` per byte, lowest bits first, stored unsigned
    with an offset of ``2 ** (bits - 1)``; 8-bit codes are stored signed.

    Raises:
        ValueError: If the array's dtype does not match ``bits``.
    """
    packed = np.asarray(packed)
    if bits == 8:
        if packed.dtype != np.int8:
            raise ValueError("8-bit codes are stored as I8")
        return packed[..., :columns]
    if bits not in (2, 4) or packed.dtype != np.uint8:
        raise ValueError(f"{bits}-bit codes are not packed U8")
    shifts = np.arange(0, 8, bits, dtype=np.uint8)
    codes = (packed[..., None] >> shifts) & np.uint8((1 << bits) - 1)
    offset = np.int8(1 << (bits - 1))
    return (codes.astype(np.int8) - offset).reshape(*packed.shape[:-1], -1)[..., :columns]


def dequantize(packed: np.ndarray, scales: np.ndarray, bits: int, columns: int) -> np.ndarray:
    """Return FP32 codes times scales, as ``gemma_quant`` does.

    ``scales`` holds one scale per row, or one per block of ``columns // scales.shape[-1]``
    columns.
    """
    codes = unpack(packed, bits, columns).astype(np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    return codes * np.repeat(scales, columns // scales.shape[-1], axis=-1)


class QATCheckpoint:
    """A local QAT checkpoint folder whose files equal a pinned reference.

    Offers :class:`~within_core.checkpoint.LocalCheckpoint`'s reads (``layer`` dequantizes the
    projections so that the graph modules can be built) plus Google's codes (:meth:`linear`),
    the calibrated ranges (:meth:`activation_scales`) and the packed tables (:meth:`table`).

    Attributes:
        folder: The checkpoint folder.
        reference: The pinned description (``source``, ``revision``, ``files``,
            ``quantization_config`` and ``text_config``).
        config: The text model configuration.
    """

    def __init__(self, folder: Path, reference: dict | None = None) -> None:
        """Open a QAT checkpoint folder and verify its configuration file.

        Args:
            folder: Folder with the checkpoint's files.
            reference: Pinned description; defaults to ``reference-config-qat.json``.

        Raises:
            ValueError: If the text or quantization configuration differs from the pinned
                one, or a stored tensor does not match its stated bit width.
        """
        self.folder = Path(folder)
        self.reference = load_reference() if reference is None else reference
        self._verified: set[str] = set()
        configuration = json.loads(self.path(CONFIG).read_text())
        if configuration["text_config"] != self.reference["text_config"]:
            raise ValueError("The text configuration differs from the pinned one")
        self.quantization = configuration.get("quantization_config")
        if self.quantization != self.reference["quantization_config"]:
            raise ValueError("The quantization configuration differs from the pinned one")
        if self.quantization.get("quant_method") != "gemma" or not self.quantization.get(
            "quantize_embeddings"
        ):
            raise ValueError("Not a Gemma QAT checkpoint with quantized embeddings")
        # As gemma_quant.replace_with_quant_layers: one regex of named groups; the group
        # that matched picks the bit width.
        patterns = self.quantization.get("module_quant_configs") or {}
        self._overrides = list(patterns.values())
        self._matcher = (
            re.compile("|".join(f"(?P<g{i}>{p})" for i, p in enumerate(patterns)))
            if patterns
            else None
        )
        self.config = Gemma4TextConfig(**configuration["text_config"])
        self.config._attn_implementation = "eager"
        self._validate()

    def path(self, name: str) -> Path:
        """Return the path of a pinned file, verifying its size and SHA-256 on first use.

        Raises:
            ValueError: If ``name`` is not pinned, or the file is missing or differs.
        """
        pinned = self.reference["files"].get(name)
        if pinned is None:
            raise ValueError(f"{name} is not a pinned checkpoint file")
        path = self.folder / name
        if name not in self._verified:
            if (
                not path.is_file()
                or path.stat().st_size != pinned["bytes"]
                or sha256(path) != pinned["sha256"]
            ):
                source = f"{self.reference['source']}@{self.reference['revision']}"
                raise ValueError(f"{path} is not the pinned file of {source}")
            self._verified.add(name)
        return path

    @staticmethod
    def key(name: str) -> str:
        """Return the stored name of projection ``name`` (``"lm_head"`` is not prefixed)."""
        return name if name == HEAD else TENSOR_PREFIX + name

    def bits(self, module: str) -> int | None:
        """Return Google's bit width for the stored module name, or ``None`` if unquantized."""
        if not should_convert_module(module, self.quantization.get("modules_to_not_convert")):
            return None
        match = self._matcher.search(module) if self._matcher else None
        if match is None:
            return self.quantization["num_bits"]
        group = next(g for g, v in match.groupdict().items() if v is not None)
        return self._overrides[int(group[1:])].get("num_bits", self.quantization["num_bits"])

    def _validate(self) -> None:
        """Check every text projection and table against the quantization configuration.

        Codes must be packed for their bit width (I8 for 8 bits, else U8) with F32 scales of
        shape ``[rows, 1]`` (tables: ``[rows, blocks]``), the shapes must be the model's, and
        each projection must carry its two activation scales. Reads the header only.
        """
        config = self.config
        with safe_open(self.path(WEIGHTS), framework="pt") as tensors:
            header = {
                k: (tensors.get_slice(k).get_dtype(), tensors.get_slice(k).get_shape())
                for k in tensors.keys()  # noqa: SIM118
                if k.startswith((TENSOR_PREFIX, HEAD + "."))
            }
        expected = {HEAD: (config.vocab_size, config.hidden_size)}
        with torch.device("meta"):
            for index in range(config.num_hidden_layers):
                layer = Gemma4TextDecoderLayer(config, index)
                for name, module in layer.named_modules():
                    if isinstance(module, nn.Linear):
                        key = f"{TENSOR_PREFIX}layers.{index}.{name}"
                        expected[key] = (module.out_features, module.in_features)
        quantized = {k.removesuffix(".weight_scale") for k in header if k.endswith(".weight_scale")}
        if expected.keys() - quantized:
            missing = sorted(expected.keys() - quantized)[:4]
            raise ValueError(f"Unquantized projections: {missing}")
        for module in sorted(quantized):
            bits, (dtype, shape) = self.bits(module), header[module + ".weight"]
            rows = shape[0]
            stored_columns = shape[1] * 8 // (bits or 8)
            columns = expected[module][1] if module in expected else stored_columns
            scales = ("input", "output")
            if (
                bits not in (2, 4, 8)
                or (module in expected and expected[module][0] != rows)
                or columns % (8 // bits)
                or dtype != ("I8" if bits == 8 else "U8")
                or shape != [rows, packed_width(columns, bits)]
                or header[module + ".weight_scale"] != ("F32", [rows, 1])
                or any(header.get(f"{module}.{s}_activation_scale") != ("F32", []) for s in scales)
            ):
                raise ValueError(
                    f"{module}: stored codes do not match {bits}-bit {expected.get(module)}"
                )
        per_layer = config.num_hidden_layers * config.hidden_size_per_layer_input
        for table, columns in (
            ("embed_tokens", config.hidden_size),
            ("embed_tokens_per_layer", per_layer),
        ):
            module, rows = TENSOR_PREFIX + table, config.vocab_size
            bits, (dtype, shape) = self.bits(module), header[module + ".embedding_quantized"]
            scale_dtype, scale_shape = header[module + ".embedding_scale"]
            if (
                bits != TABLE_BITS
                or columns % (8 // bits)
                or (dtype, shape) != ("U8", [rows, packed_width(columns, bits)])
                or scale_dtype != "F32"
                or len(scale_shape) != 2
                or scale_shape[0] != rows
                or columns % scale_shape[1]
            ):
                raise ValueError(
                    f"{module}: stored table does not match {bits}-bit [{rows}, {columns}]"
                )

    def tensor(self, name: str, dtype: torch.dtype = torch.float16) -> Tensor:
        """Return an unquantized text-model tensor, e.g. ``"norm.weight"``, cast to ``dtype``."""
        with safe_open(self.path(WEIGHTS), framework="pt") as tensors:
            return tensors.get_tensor(TENSOR_PREFIX + name).to(dtype)

    def row_range(
        self, name: str, start: int, end: int, dtype: torch.dtype = torch.float16
    ) -> Tensor:
        """Return rows ``[start, end)`` of an unquantized text-model tensor, cast to ``dtype``."""
        with safe_open(self.path(WEIGHTS), framework="pt") as tensors:
            return tensors.get_slice(TENSOR_PREFIX + name)[start:end].to(dtype)

    def linear(
        self, name: str, start: int | None = None, end: int | None = None
    ) -> tuple[Tensor, Tensor, int]:
        """Return Google's codes of projection ``name`` for rows ``[start, end)``.

        Args:
            name: ``"layers.<i>.<module>"`` (e.g. ``"layers.3.mlp.down_proj"``) or ``"lm_head"``.
            start: First row; ``None`` for the whole projection.
            end: Row after the last; ``None`` for the whole projection.

        Returns:
            Signed ``int8`` codes ``[rows, in_features]``, F32 scales ``[rows, 1]`` and the
            bit width.
        """
        key = self.key(name)
        bits = self.bits(key)
        with safe_open(self.path(WEIGHTS), framework="pt") as tensors:
            packed = tensors.get_slice(key + ".weight")[start:end].numpy()
            scale = tensors.get_slice(key + ".weight_scale")[start:end]
        codes = unpack(packed, bits, packed.shape[-1] * 8 // bits)
        return torch.from_numpy(np.ascontiguousarray(codes)), scale, bits

    def activation_scales(self, name: str) -> tuple[float, float]:
        """Return the calibrated INT8 input and output scales of projection ``name``.

        A scale of ``0`` means the side was not calibrated.
        """
        with safe_open(self.path(WEIGHTS), framework="pt") as tensors:
            return tuple(
                float(tensors.get_tensor(f"{self.key(name)}.{side}_activation_scale"))
                for side in ("input", "output")
            )

    def layer(self, index: int, dtype: torch.dtype = torch.float16) -> Gemma4TextDecoderLayer:
        """Return decoder layer ``index`` with its projections dequantized to ``dtype``.

        The projections are codes times scales in FP32, then cast; the exporter replaces them
        with Google's codes afterwards. Layers that reuse a donor's KV drop their unused K/V
        projections, as Transformers does.

        Raises:
            ValueError: If the index is outside the decoder.
        """
        if not 0 <= index < self.config.num_hidden_layers:
            raise ValueError(f"Layer {index} is outside the decoder")
        with torch.device("meta"):
            layer = Gemma4TextDecoderLayer(self.config, index)
        prefix = f"{TENSOR_PREFIX}layers.{index}."
        unused = (
            ("self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.k_norm.weight")
            if layer.self_attn.is_kv_shared_layer
            else ()
        )
        with safe_open(self.path(WEIGHTS), framework="pt") as tensors:
            keys = set(tensors.keys())  # noqa: SIM118
            state: dict[str, Tensor | None] = {}
            for key in keys:
                if not key.startswith(prefix) or key.endswith(SIDE_TENSORS):
                    continue
                quantized = key.removesuffix(".weight") + ".weight_scale" in keys
                state[key.removeprefix(prefix)] = (
                    None if quantized else tensors.get_tensor(key).to(dtype)
                )
        for name in unused:
            state.pop(name, None)
        for name in [n for n, v in state.items() if v is None]:
            codes, scale, _ = self.linear(f"layers.{index}.{name.removesuffix('.weight')}")
            state[name] = (codes * scale).to(dtype)
        layer.load_state_dict(state, strict=True, assign=True)
        return layer.eval()

    def table(self, name: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """Return rows ``[start, end)`` of a table as stored: packed U8 codes and F32 scales.

        Args:
            name: ``"embed_tokens"`` or ``"embed_tokens_per_layer"``.
        """
        with safe_open(self.path(WEIGHTS), framework="pt") as tensors:
            return (
                tensors.get_slice(f"{TENSOR_PREFIX}{name}.embedding_quantized")[start:end].numpy(),
                tensors.get_slice(f"{TENSOR_PREFIX}{name}.embedding_scale")[start:end].numpy(),
            )


def table_descriptors(checkpoint: QATCheckpoint) -> dict[str, dict]:
    """Return the ``gemma-runtime.json`` descriptor of each 2-bit table."""
    config = checkpoint.config
    columns = {
        "embeddings": config.hidden_size,
        "perLayerEmbeddings": config.num_hidden_layers * config.hidden_size_per_layer_input,
    }
    descriptors = {}
    for key, name in TABLES.items():
        blocks = checkpoint.table(name, 0, 1)[1].shape[-1]
        descriptors[key] = {
            "file": f"{key}.int2",
            "format": "int2",
            "rows": config.vocab_size,
            "columns": columns[key],
            "scales": {"file": f"{key}.scales.f32", "columnsPerScale": columns[key] // blocks},
        }
    return descriptors


def table_bytes(checkpoint: QATCheckpoint) -> int:
    """Return the bytes both tables take on disk."""
    return sum(
        d["rows"] * d["columns"] // 4
        + d["rows"] * (d["columns"] // d["scales"]["columnsPerScale"]) * 4
        for d in table_descriptors(checkpoint).values()
    )


def write_tables(
    checkpoint: QATCheckpoint, output: Path, block_rows: int = 1024
) -> dict[str, dict]:
    """Write both 2-bit tables of ``checkpoint`` into ``output`` and their descriptors.

    The packed codes and the little-endian F32 scales are copied as stored, one row block at
    a time; ``tables.json`` holds the descriptors that :mod:`within_core.package` writes into
    ``gemma-runtime.json`` (schema 2). The host decodes a row as FP32 code times scale,
    rounded once to FP16, and applies Gemma's embedding scale after the lookup.

    Returns:
        The descriptors, keyed ``embeddings`` and ``perLayerEmbeddings``.

    Raises:
        ValueError: If a written file does not have the expected size.
    """
    output.mkdir(parents=True, exist_ok=True)
    descriptors = table_descriptors(checkpoint)
    for key, descriptor in descriptors.items():
        rows, columns = descriptor["rows"], descriptor["columns"]
        blocks = columns // descriptor["scales"]["columnsPerScale"]
        sizes = {
            descriptor["file"]: rows * columns // 4,
            descriptor["scales"]["file"]: rows * blocks * 4,
        }
        codes_path, scales_path = (output / name for name in sizes)
        with codes_path.open("xb") as codes, scales_path.open("xb") as scales:
            for start in range(0, rows, block_rows):
                packed, scale = checkpoint.table(TABLES[key], start, min(start + block_rows, rows))
                packed.tofile(codes)
                scale.astype("<f4", copy=False).tofile(scales)
        for name, size in sizes.items():
            if (output / name).stat().st_size != size:
                raise ValueError(f"{output / name} is not a complete table")
    (output / TABLES_RECORD).write_text(json.dumps(descriptors, indent=2) + "\n")
    return descriptors


def lookup(folder: Path, descriptor: dict, token: int, embed_scale: float = 1.0) -> np.ndarray:
    """Return one table row as the host computes it: FP32 code x scale x ``embed_scale``, FP16.

    ``embed_scale`` of 1 gives the unscaled row; the square root of the table's width gives
    ``gemma_quant.QuantizedEmbedding.forward``.
    """
    rows, columns = descriptor["rows"], descriptor["columns"]
    per = descriptor["scales"]["columnsPerScale"]
    folder = Path(folder)
    packed = np.memmap(
        folder / descriptor["file"], dtype=np.uint8, mode="r", shape=(rows, columns // 4)
    )
    scales = np.memmap(
        folder / descriptor["scales"]["file"], dtype="<f4", mode="r", shape=(rows, columns // per)
    )
    row = dequantize(packed[token], scales[token], TABLE_BITS, columns)
    return (row * np.float32(embed_scale)).astype(np.float16)
