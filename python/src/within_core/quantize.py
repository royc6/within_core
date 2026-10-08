"""Weight-only compression of the chunks' 1x1 projections.

Pipeline position: :mod:`within_core.export` quantizes each decoder layer right after it
is built from the checkpoint, so only one layer is ever dense in memory, and builds the
vocabulary head row block by row block. Activations, norms and KV stay FP16.

Mixed precision, chosen by quality per byte:

* MLP projections (``gate``, ``up``, ``down``) hold most of the weights: 6-bit palettes
  (k-means lookup tables of 64 entries per group of 16 output channels, with per-channel
  scales).
* Attention and per-layer-embedding projections: INT8, per-channel symmetric.
* Vocabulary head (tied to the embedding): INT4, per-channel symmetric.

All three use Apple's ``coreai-opt`` quantizers.

The lite packages (release 1.1) take Google's quantization-aware codes instead
(:mod:`within_core.qat`): 4-bit attention and MLP codes as 16-entry palettes, 8-bit
per-layer-embedding codes as INT8, the 2-bit vocabulary head as 4-entry palettes, every one
with Google's per-channel scale rounded once to FP16, and each projection clamped to its
calibrated INT8 range without rounding. Nothing is re-quantized.
"""

from collections.abc import Callable

import torch
from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
    PalettizationSpec,
)
from coreai_opt.palettization.spec.granularity import PerGroupedChannelGranularity
from coreai_opt.quantization import ExecutionMode, Quantizer, QuantizerConfig
from coreai_torch._compression.custom_layers import ScaledPalettizeModule, WeightDequantizeModule
from coreai_torch._compression.utils import wrap_for_parametrization
from torch import Tensor, nn
from torch.nn.utils.parametrize import is_parametrized, register_parametrization

MLP_SCHEME = "lut6"
PROJECTION_SCHEME = "w8"
HEAD_SCHEME = "w4"
MLP_PROJECTIONS = ("gate", "up", "down")
WEIGHT_FORMAT = "int8-per-channel-symmetric-weight-only; feedforward lut6; head w4"
LITE_WEIGHT_FORMAT = (
    "qat-mobile; attention, feedforward int4-per-channel-as-lut4; ple int8-per-channel; "
    "ple model projection float16; head int2-as-lut2; srq clamp both-where"
)
# Chunk projection -> the QAT checkpoint's module. The per-layer model projection is not
# listed: Google left it unquantized, and so does the lite package.
QAT_SOURCES = {
    "attention.q": "self_attn.q_proj",
    "attention.k": "self_attn.k_proj",
    "attention.v": "self_attn.v_proj",
    "attention.o": "self_attn.o_proj",
    "feedforward.gate": "mlp.gate_proj",
    "feedforward.up": "mlp.up_proj",
    "feedforward.down": "mlp.down_proj",
    "feedforward.ple_gate": "per_layer_input_gate",
    "feedforward.ple_projection": "per_layer_projection",
}
PALETTE_GROUP = 16
CONTAINERS = {8: torch.int8, 4: torch.int4, 2: torch.int2}


def quantize_conv(conv: nn.Conv2d, name: str, scheme: str) -> tuple[nn.Module, nn.Module]:
    """Compress one 1x1 projection.

    Args:
        conv: The projection, with FP16 weights.
        name: Name used in error messages.
        scheme: ``"w8"`` (INT8 per channel), ``"w4"`` (INT4 per channel) or ``"lut6"``
            (6-bit palette per 16 output channels with per-channel scales).

    Returns:
        The finalized module that replaces ``conv`` and its compressed-weight module.

    Raises:
        ValueError: If the projection is not pointwise or compression gives an unexpected
            or non-finite result.
    """
    if conv.kernel_size != (1, 1):
        raise ValueError(f"Only pointwise projections are expected: {name}")
    example = (torch.zeros(1, conv.in_channels, 1, 1, dtype=conv.weight.dtype),)
    if scheme == "lut6":
        spec = PalettizationSpec(
            n_bits=6,
            granularity=PerGroupedChannelGranularity(axis=0, group_size=16),
            enable_per_channel_scale=True,
        )
        config = ModuleKMeansPalettizerConfig(op_state_spec={"weight": spec})
        palettizer = KMeansPalettizer(conv, KMeansPalettizerConfig(global_config=config))
        palettizer.prepare(example)
        finalized = palettizer.finalize()
        tables = [m for m in finalized.modules() if hasattr(m, "indices") and hasattr(m, "lut")]
        if len(tables) != 1 or not torch.isfinite(tables[0].lut).all():
            raise ValueError(f"Projection did not become exactly one finite palette: {name}")
        return finalized, tables[0]
    preset = getattr(QuantizerConfig.presets, scheme)
    quantizer = Quantizer(conv, preset(execution_mode=ExecutionMode.EAGER))
    quantizer.prepare(example)
    finalized = quantizer.finalize()
    weights = [m for m in finalized.modules() if isinstance(m, WeightDequantizeModule)]
    expected = torch.int8 if scheme == "w8" else torch.int4
    if len(weights) != 1 or weights[0].input_dtype != expected:
        raise ValueError(f"Projection did not become exactly one {expected} weight: {name}")
    scale = weights[0].scale
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError(f"Non-finite or non-positive quantization scales: {name}")
    return finalized, weights[0]


def quantize_projections(module: nn.Module) -> int:
    """Compress every uncompressed 1x1 projection of ``module`` in place.

    Projections named ``feedforward.<gate|up|down>`` (the MLP of a decoder block) take
    :data:`MLP_SCHEME`; all others take :data:`PROJECTION_SCHEME`.

    Args:
        module: A decoder block or a per-layer projection.

    Returns:
        The number of projections compressed.
    """
    count = 0
    convs = [(n, c) for n, c in module.named_modules() if isinstance(c, nn.Conv2d)]
    with torch.no_grad():
        for name, conv in convs:
            if is_parametrized(conv):
                continue
            mlp = name.startswith("feedforward.") and name.rsplit(".", 1)[-1] in MLP_PROJECTIONS
            finalized, _ = quantize_conv(conv, name, MLP_SCHEME if mlp else PROJECTION_SCHEME)
            parent, _, leaf = name.rpartition(".")
            setattr(module.get_submodule(parent) if parent else module, leaf, finalized)
            count += 1
    return count


def require_compressed(module: nn.Module) -> None:
    """Raise ``ValueError`` if any 1x1 projection of ``module`` still has dense weights."""
    convs = [(n, c) for n, c in module.named_modules() if isinstance(c, nn.Conv2d)]
    dense = [name for name, conv in convs if not is_parametrized(conv)]
    if dense:
        raise ValueError(f"Projections escaped compression: {dense}")


def quantize_head(
    read_rows: Callable[[int, int], Tensor],
    rows: int,
    columns: int,
    parts: int,
    block_rows: int = 4096,
) -> list[nn.Conv2d]:
    """Compress the vocabulary head to INT4 without holding it dense.

    Scales are per output row, so quantizing a block of rows gives exactly the rows of the
    whole-head result; only one FP16 block is in memory at a time.

    Args:
        read_rows: Returns FP16 rows ``[start, end)`` of the head weight ``[rows, columns]``.
        rows: Vocabulary size.
        columns: Hidden size.
        parts: Number of equal row slices to return.
        block_rows: Rows quantized at a time.

    Returns:
        One 1x1 projection with INT4 weights per row slice, in vocabulary order.

    Raises:
        ValueError: If the slices do not divide the vocabulary or a zero point is nonzero.
    """
    if rows % parts:
        raise ValueError("Head parts must divide the vocabulary")
    data = torch.empty((rows, columns, 1, 1), dtype=torch.int8)
    scale = zero = None
    with torch.no_grad():
        for start in range(0, rows, block_rows):
            end = min(start + block_rows, rows)
            conv = nn.Conv2d(columns, end - start, 1, bias=False, device="meta")
            conv.load_state_dict({"weight": read_rows(start, end)[:, :, None, None]}, assign=True)
            _, weight = quantize_conv(conv, f"head[{start}:{end}]", HEAD_SCHEME)
            if scale is None or zero is None:
                scale = torch.empty((rows, 1, 1, 1), dtype=weight.scale.dtype)
                zero = torch.empty((rows, 1, 1, 1), dtype=weight.zero_point.dtype)
            data[start:end].copy_(weight.quantized_data)
            scale[start:end].copy_(weight.scale)
            zero[start:end].copy_(weight.zero_point)
            del conv, weight
    if torch.count_nonzero(zero):
        raise ValueError("Symmetric head quantization produced a nonzero zero point")
    wrapper = wrap_for_parametrization(WeightDequantizeModule)
    size = rows // parts
    result = []
    for start in range(0, rows, size):
        piece = nn.Conv2d(columns, size, 1, bias=False, device="meta", dtype=torch.float16)
        piece.weight = nn.Parameter(torch.empty(0, dtype=torch.float16), requires_grad=False)
        dequantize = wrapper(
            quantized_data=data[start : start + size].clone(),
            scale=scale[start : start + size].clone(),
            zero_point=zero[start : start + size].clone(),
            minval=None,
            input_dtype=torch.int4,
            output_dtype=None,
        )
        register_parametrization(piece, "weight", dequantize, unsafe=True)
        result.append(piece)
    return result


def qat_weight(codes: Tensor, scale: Tensor, bits: int, name: str) -> nn.Module:
    """Return Google's codes and scales as a weight-dequantize module.

    Args:
        codes: Signed ``int8`` codes ``[rows, columns]`` of ``bits`` bits.
        scale: F32 per-row scales ``[rows, 1]``, rounded once to FP16 here.
        bits: 8, 4 or 2.
        name: Used in error messages.

    Raises:
        ValueError: If a rounded scale is not finite and positive.
    """
    half = scale.to(torch.float16)
    if not torch.isfinite(half).all() or not (half > 0).all():
        raise ValueError(f"Non-finite or non-positive FP16 scales: {name}")
    return wrap_for_parametrization(WeightDequantizeModule)(
        quantized_data=codes[:, :, None, None].contiguous(),
        scale=half[:, :, None, None].contiguous(),
        zero_point=torch.zeros((len(codes), 1, 1, 1), dtype=torch.int8),
        minval=None,
        input_dtype=CONTAINERS[bits],
        output_dtype=None,
    )


def qat_palette(weight: nn.Module, bits: int, group: int = PALETTE_GROUP) -> nn.Module:
    """Return the codes of ``weight`` as a palette with the same values.

    The indices are the codes plus ``2 ** (bits - 1)``, the palette of every ``group`` output
    channels (fewer rows form one group) lists the ``2 ** bits`` code values in order, and
    the per-channel scale is kept, so the dequantized weight is FP32 code times FP16 scale
    rounded once to FP16, exactly as the dequantize module computes it.

    Raises:
        ValueError: If the rows do not form whole groups or a zero point is nonzero.
    """
    codes = weight.quantized_data
    group = min(group, len(codes))
    if len(codes) % group or torch.count_nonzero(weight.zero_point):
        raise ValueError(f"A palette needs zero offsets and rows in groups of {group}")
    offset = 1 << (bits - 1)
    lut = torch.arange(-offset, offset, dtype=torch.float16).reshape(1, 1, 1, 1, -1, 1)
    return wrap_for_parametrization(ScaledPalettizeModule)(
        indices=(codes + offset).to(torch.uint8),
        lut=lut.repeat(len(codes) // group, 1, 1, 1, 1, 1),
        scale=weight.scale,
    )


def frozen_conv(weight: nn.Module) -> nn.Conv2d:
    """Return a 1x1 FP16 projection whose weight is the parametrization ``weight``."""
    stored = weight.indices if hasattr(weight, "indices") else weight.quantized_data
    rows, columns = stored.shape[:2]
    conv = nn.Conv2d(columns, rows, 1, bias=False, device="meta", dtype=torch.float16)
    conv.weight = nn.Parameter(torch.empty(0, dtype=torch.float16), requires_grad=False)
    register_parametrization(conv, "weight", weight, unsafe=True)
    return conv


class ClampedProjection(nn.Module):
    """A projection whose input and output are clamped to calibrated INT8 ranges.

    Google calibrated a scale ``s`` per side; the range is ``[-128 s, 127 s]``. Values are
    clamped with a compare-and-select (exact in FP16) and never rounded: on the frozen
    quality set the clamp alone brings Google's checkpoint level with Google's reference
    implementation, and the rounding would not compile for the Neural Engine. A scale of
    zero means the side was not calibrated and is left alone. Projections that read the same
    tensor with the same input range (q, k and v; gate and up) share one input clamp through
    ``shared``, so the traced graph clamps that tensor once.
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        scales: tuple[float, float],
        shared: dict[tuple[float, float], tuple[Tensor, Tensor]],
    ) -> None:
        super().__init__()
        self.conv = conv
        self.bounds = [(-128.0 * s, 127.0 * s) if s > 0 else None for s in scales]
        self.shared = shared

    @staticmethod
    def bound(x: Tensor, low: float, high: float) -> Tensor:
        """Clamp ``x`` to ``[low, high]`` by comparing and selecting."""
        low_t, high_t = torch.tensor(low, dtype=x.dtype), torch.tensor(high, dtype=x.dtype)
        return torch.where(x < low_t, low_t, torch.where(x > high_t, high_t, x))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the clamped projection to ``x``."""
        if self.bounds[0]:
            hit = self.shared.get(self.bounds[0])
            if hit is not None and hit[0] is x:
                x = hit[1]
            else:
                source, x = x, self.bound(x, *self.bounds[0])
                self.shared[self.bounds[0]] = (source, x)
        y = self.conv(x)
        return self.bound(y, *self.bounds[1]) if self.bounds[1] else y


def quantize_projections_qat(
    module: nn.Module,
    codes: Callable[[str], tuple[Tensor, Tensor, int]],
    activation_scales: Callable[[str], tuple[float, float]],
) -> int:
    """Replace every 1x1 projection of a decoder block with Google's codes, clamped.

    Args:
        module: A decoder block (``DecoderBlock``).
        codes: Returns ``(codes, scales, bits)`` of a checkpoint module such as
            ``"self_attn.q_proj"`` (see :data:`QAT_SOURCES`).
        activation_scales: Returns the calibrated ``(input, output)`` scales of a module.

    Returns:
        The number of projections replaced.

    Raises:
        ValueError: If a projection is not in :data:`QAT_SOURCES` or its codes do not have
            the projection's shape.
    """
    count = 0
    shared: dict[tuple[float, float], tuple[Tensor, Tensor]] = {}
    convs = [(n, c) for n, c in module.named_modules() if isinstance(c, nn.Conv2d)]
    with torch.no_grad():
        for name, conv in convs:
            if is_parametrized(conv):
                continue
            source = QAT_SOURCES.get(name)
            if source is None:
                raise ValueError(f"A projection has no QAT codes: {name}")
            values, scale, bits = codes(source)
            if list(values.shape) != list(conv.weight.shape[:2]):
                raise ValueError(f"{name}: QAT codes {list(values.shape)} differ from the layer")
            weight = qat_weight(values, scale, bits, name)
            if bits == 4:
                weight = qat_palette(weight, bits)
            replacement = ClampedProjection(frozen_conv(weight), activation_scales(source), shared)
            parent, _, leaf = name.rpartition(".")
            setattr(module.get_submodule(parent) if parent else module, leaf, replacement)
            count += 1
    return count


def quantize_head_qat(
    read_codes: Callable[[int, int], tuple[Tensor, Tensor, int]],
    rows: int,
    columns: int,
    parts: int,
    block_rows: int = 4096,
) -> list[nn.Conv2d]:
    """Return the vocabulary head as palettes of Google's 2-bit codes, in row slices.

    Args:
        read_codes: Returns ``(codes, scales, bits)`` of head rows ``[start, end)``.
        rows: Vocabulary size.
        columns: Hidden size.
        parts: Number of equal row slices to return.
        block_rows: Rows read at a time.

    Raises:
        ValueError: If the slices do not divide the vocabulary or the codes are not 2-bit.
    """
    if rows % parts:
        raise ValueError("Head parts must divide the vocabulary")
    data = torch.empty((rows, columns), dtype=torch.int8)
    scale = torch.empty((rows, 1), dtype=torch.float32)
    for start in range(0, rows, block_rows):
        end = min(start + block_rows, rows)
        codes, scales, bits = read_codes(start, end)
        if bits != 2:
            raise ValueError(f"The head's codes are {bits}-bit, not 2-bit")
        data[start:end].copy_(codes)
        scale[start:end].copy_(scales)
    size = rows // parts
    result = []
    for start in range(0, rows, size):
        weight = qat_weight(data[start : start + size], scale[start : start + size], 2, "head")
        result.append(frozen_conv(qat_palette(weight, 2)))
    return result
