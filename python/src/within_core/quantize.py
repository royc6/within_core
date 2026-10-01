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
from coreai_torch._compression.custom_layers import WeightDequantizeModule
from coreai_torch._compression.utils import wrap_for_parametrization
from torch import Tensor, nn
from torch.nn.utils.parametrize import is_parametrized, register_parametrization

MLP_SCHEME = "lut6"
PROJECTION_SCHEME = "w8"
HEAD_SCHEME = "w4"
MLP_PROJECTIONS = ("gate", "up", "down")
WEIGHT_FORMAT = "int8-per-channel-symmetric-weight-only; feedforward lut6; head w4"


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
