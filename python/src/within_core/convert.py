"""Command line: convert a local Gemma 4 E4B checkpoint into the within model package.

Pipeline position: the entry point that runs the steps in order. Every input and output is
an explicit argument; the defaults are the published Full package::

    python -m within_core.convert chunk 0 --checkpoint CKPT --output EXPORT  # 0 to 7
    python -m within_core.convert finalize --output EXPORT
    python -m within_core.convert tables --checkpoint CKPT --output TABLES
    python -m within_core.convert package --export EXPORT --tables TABLES \\
        --checkpoint CKPT --output RELEASE --revision REV --base-url URL

``--context 4096`` gives the Balanced package. With Google's QAT mobile checkpoint as
``CKPT`` (recognised by its ``config.json``) the same commands give a lite package:
``--context 4096 --prefill 128`` (Lite 4K) or ``--context 32000 --prefill 128`` (Lite 32K);
``tables`` then writes the 2-bit tables. ``catalog`` writes the app's catalog of one package
per tier.

Run each ``chunk`` in its own process: one chunk's weights then fit in memory. Each command
refuses to start unless the output volume keeps ``--min-free-gb`` free after its writes.
"""

import argparse
import json
import logging
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from within_core import export, package, qat, tables
from within_core.checkpoint import LocalCheckpoint, pinned_config

# Upper bound of one serialized chunk of the published plan.
CHUNK_BYTES = 600_000_000
# Upper bound of what assembling writes besides files it copies instead of hard-linking.
PACKAGE_BYTES = 100_000_000
GIGABYTE = 1_000_000_000


def require_space(folder: Path, needed: int, min_free_gb: float) -> None:
    """Raise ``OSError`` unless ``folder``'s volume keeps ``min_free_gb`` after ``needed`` bytes.

    Args:
        folder: Output folder (it may not exist yet).
        needed: Bytes the step will write.
        min_free_gb: Space that must remain free afterwards, in gigabytes.
    """
    existing = _existing(folder)
    free = shutil.disk_usage(existing).free
    if free - needed < min_free_gb * GIGABYTE:
        raise OSError(
            f"{existing}: {free / GIGABYTE:.1f} GB free; this step writes {needed / GIGABYTE:.1f}"
            f" GB and must leave {min_free_gb} GB"
        )


def _existing(path: Path) -> Path:
    """Return ``path`` or its nearest existing ancestor."""
    return next(p for p in (path, *path.parents) if p.exists())


def _tree_bytes(folder: Path) -> int:
    """Return the total size of the files under ``folder``."""
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def open_checkpoint(folder: Path) -> LocalCheckpoint | qat.QATCheckpoint:
    """Open the BF16 checkpoint, or Google's QAT one if ``folder`` holds it."""
    if qat.is_qat_checkpoint(folder):
        return qat.QATCheckpoint(folder)
    return LocalCheckpoint(folder)


def run_chunk(arguments: argparse.Namespace) -> dict[str, object]:
    """Export one chunk."""
    require_space(arguments.output, CHUNK_BYTES, arguments.min_free_gb)
    torch.set_num_threads(arguments.threads)
    checkpoint = open_checkpoint(arguments.checkpoint)
    asset = export.export_chunk(
        checkpoint, arguments.index, arguments.output, arguments.context, arguments.prefill
    )
    return {"asset": str(asset), "bytes": _tree_bytes(asset)}


def run_finalize(arguments: argparse.Namespace) -> dict[str, object]:
    """Write the chunk contract and the runtime configuration."""
    require_space(arguments.output, 1_000_000, arguments.min_free_gb)
    export.finalize(arguments.output, pinned_config(), arguments.context, arguments.prefill)
    return {"contract": str(arguments.output / export.CHUNK_CONTRACT)}


def run_tables(arguments: argparse.Namespace) -> dict[str, object]:
    """Write the embedding tables: FP16 from the BF16 checkpoint, 2-bit from the QAT one."""
    checkpoint = open_checkpoint(arguments.checkpoint)
    if isinstance(checkpoint, qat.QATCheckpoint):
        require_space(arguments.output, qat.table_bytes(checkpoint), arguments.min_free_gb)
        descriptors = qat.write_tables(checkpoint, arguments.output)
        return {"tables": [str(arguments.output / d["file"]) for d in descriptors.values()]}
    needed = sum(
        checkpoint.config.vocab_size * columns * 2
        for _, _, columns in tables.table_sources(checkpoint)
    )
    require_space(arguments.output, needed, arguments.min_free_gb)
    paths = tables.write_tables(checkpoint, arguments.output)
    return {"tables": [str(p) for p in paths]}


def run_package(arguments: argparse.Namespace) -> dict[str, object]:
    """Assemble the package and both manifests."""
    # Chunks and tables are hard-linked when they are on the output's volume, else copied.
    volume = _existing(arguments.output).stat().st_dev
    sources = (arguments.export, arguments.tables)
    copied = sum(_tree_bytes(s) for s in sources if s.stat().st_dev != volume)
    require_space(arguments.output, PACKAGE_BYTES + copied, arguments.min_free_gb)
    checkpoint = open_checkpoint(arguments.checkpoint)
    data = package.assemble(
        arguments.export,
        arguments.tables,
        checkpoint,
        arguments.output,
        arguments.revision,
        arguments.base_url,
    )
    return {
        "installationID": f"{data['modelID']}-{data['revision']}",
        "files": len(data["files"]),
        "bytes": sum(row["size"] for row in data["files"]),
    }


def run_catalog(arguments: argparse.Namespace) -> dict[str, object]:
    """Write the app's catalog of one package per tier."""
    entries = [(tier, Path(folder)) for tier, folder in arguments.entry]
    data = package.catalog(entries, arguments.default, arguments.output)
    return {
        "catalog": str(arguments.output),
        "packages": [[row["tier"], row["manifest"]["revision"]] for row in data["packages"]],
    }


def parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    root = argparse.ArgumentParser(prog="python -m within_core.convert", description=__doc__)
    root.formatter_class = argparse.RawDescriptionHelpFormatter
    root.add_argument("--min-free-gb", type=float, default=10.0, help="space to leave free")
    commands = root.add_subparsers(dest="command", required=True)

    chunk = commands.add_parser("chunk", help="export one chunk")
    chunk.add_argument("index", type=int, choices=range(len(export.BOUNDARIES)))
    chunk.add_argument("--checkpoint", type=Path, required=True)
    chunk.add_argument("--output", type=Path, required=True, help="export folder")
    chunk.add_argument("--context", type=int, default=export.CONTEXT)
    chunk.add_argument(
        "--prefill", type=int, default=export.PREFILL_LENGTH, help="tokens per prefill call"
    )
    chunk.add_argument("--threads", type=int, default=4, help="Torch CPU threads")
    chunk.set_defaults(run=run_chunk)

    finalize = commands.add_parser("finalize", help="write the contract once all chunks exist")
    finalize.add_argument("--output", type=Path, required=True, help="export folder")
    finalize.add_argument("--context", type=int, default=export.CONTEXT)
    finalize.add_argument("--prefill", type=int, default=export.PREFILL_LENGTH)
    finalize.set_defaults(run=run_finalize)

    table = commands.add_parser("tables", help="write the embedding tables")
    table.add_argument("--checkpoint", type=Path, required=True)
    table.add_argument("--output", type=Path, required=True, help="tables folder")
    table.set_defaults(run=run_tables)

    assemble = commands.add_parser("package", help="assemble the package and its manifests")
    assemble.add_argument("--export", type=Path, required=True, help="finalized export folder")
    assemble.add_argument("--tables", type=Path, required=True, help="tables folder")
    assemble.add_argument("--checkpoint", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True, help="new release folder")
    assemble.add_argument("--revision", required=True)
    assemble.add_argument("--base-url", required=True)
    assemble.set_defaults(run=run_package)

    catalog = commands.add_parser("catalog", help="write the app's catalog of one package per tier")
    catalog.add_argument(
        "--entry", nargs=2, action="append", metavar=("TIER", "PACKAGE"), required=True
    )
    catalog.add_argument("--default", required=True, choices=package.TIERS, help="the default tier")
    catalog.add_argument("--output", type=Path, required=True, help="ModelCatalog.json to write")
    catalog.set_defaults(run=run_catalog)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and print its result as JSON."""
    arguments = parser().parse_args(argv)
    logging.basicConfig(format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("within_core").setLevel(logging.INFO)
    began = time.monotonic()
    result = arguments.run(arguments)
    result["seconds"] = round(time.monotonic() - began, 1)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
