"""FP16 embedding tables that the host reads outside the graph.

Pipeline position: independent of the chunk export; :mod:`within_core.package` places the
two tables next to the chunks. The token embedding (``embeddings.f16``, ``[vocab, hidden]``)
and the per-layer embedding (``perLayerEmbeddings.f16``, ``[vocab, layers * ple_dim]``) are
lookups, not computation: the host reads one row per token from a memory-mapped file
instead of holding a 7 GB gather in a graph. Both are raw little-endian FP16, row-major,
written straight from the checkpoint's BF16 tensors; the host applies Gemma's embedding
scales (the square root of each table's width) after the lookup.
"""

import functools
from collections.abc import Callable
from pathlib import Path

from torch import Tensor

from within_core.checkpoint import LocalCheckpoint

EMBEDDINGS = "embeddings.f16"
PER_LAYER_EMBEDDINGS = "perLayerEmbeddings.f16"


def table_sources(checkpoint: LocalCheckpoint) -> list[tuple[str, str, int]]:
    """Return ``(file name, checkpoint tensor, columns)`` for each table."""
    config = checkpoint.config
    per_layer = config.num_hidden_layers * config.hidden_size_per_layer_input
    return [
        (EMBEDDINGS, "embed_tokens.weight", config.hidden_size),
        (PER_LAYER_EMBEDDINGS, "embed_tokens_per_layer.weight", per_layer),
    ]


def write_table(
    path: Path,
    read_rows: Callable[[int, int], Tensor],
    rows: int,
    columns: int,
    block_rows: int = 1024,
) -> None:
    """Write a ``[rows, columns]`` FP16 table to a new file, one row block at a time.

    Args:
        path: Destination; must not exist.
        read_rows: Returns FP16 rows ``[start, end)`` of the table.
        rows: Number of rows.
        columns: Number of columns.
        block_rows: Rows read at a time.

    Raises:
        ValueError: If the written file does not have the expected size.
    """
    with path.open("xb") as stream:
        for start in range(0, rows, block_rows):
            block = read_rows(start, min(start + block_rows, rows))
            block.numpy().astype("<f2", copy=False).tofile(stream)
    if path.stat().st_size != rows * columns * 2:
        raise ValueError(f"{path} does not hold a [{rows}, {columns}] FP16 table")


def write_tables(checkpoint: LocalCheckpoint, output: Path) -> list[Path]:
    """Write both tables of ``checkpoint`` into ``output``.

    Returns:
        The paths of the written tables.
    """
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, tensor, columns in table_sources(checkpoint):
        read_rows = functools.partial(checkpoint.row_range, tensor)
        write_table(output / name, read_rows, checkpoint.config.vocab_size, columns)
        paths.append(output / name)
    return paths
