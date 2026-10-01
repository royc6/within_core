"""Command line: the numerics reference and the placement verdict of one recorded run.

Pipeline position: after a device recording (see the README). Every input and output is an
explicit argument::

    python -m within_core.verify reference RUN --checkpoint CKPT --output NUMERICS.json
    python -m within_core.verify review RUN --numerics NUMERICS.json ... --output REVIEW.json

``reference`` builds one chunk at a time and saves its report after each; with ``--chunks``
a long run can be split across processes and files, which ``review`` then takes together.
``review`` exits with status 1 when the verdict is not a pass.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from within_core.verify import review


def run_reference(arguments: argparse.Namespace) -> int:
    """Recompute and score the chosen chunks; print one summary line per output."""
    # Imported here so that ``review`` runs without PyTorch.
    import torch

    from within_core.checkpoint import LocalCheckpoint
    from within_core.export import CONTEXT
    from within_core.verify import reference

    torch.set_num_threads(arguments.threads)
    chunks = [int(x) for x in arguments.chunks.split(",")] if arguments.chunks else None
    checkpoint = LocalCheckpoint(arguments.checkpoint)
    context = arguments.context or CONTEXT
    result = reference.reference(arguments.run, checkpoint, arguments.output, chunks, context)
    for row in result["rows"]:
        if "scores" in row:
            cells = (
                f"{arm} psnr={_round(s['psnrDb'])} cos={s['cosine']:.6f}"
                for arm, s in row["scores"].items()
            )
            print(row["chunk"], row["output"], "  ".join(cells))
        else:
            print(row["chunk"], row["output"], json.dumps(row["distribution"]))
    return 0


def run_review(arguments: argparse.Namespace) -> int:
    """Write the verdict and print its summary; status 1 unless it passed."""
    if arguments.output.exists():
        raise FileExistsError(f"Use a new output file: {arguments.output}")
    result = review.review(arguments.run, arguments.numerics, arguments.subsystem)
    arguments.output.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    summary = {key: result[key] for key in ("aneExecV1Passed", "pid", "revision")}
    summary |= {key: result[key]["passed"] for key in ("P0", "C0", "C1", "C3", "C6")}
    summary["C3decidedBy"] = result["C3"]["decidedBy"]
    summary["C3permutationP"] = result["C3"]["permutationP"]
    print(json.dumps(summary))
    return 0 if result["aneExecV1Passed"] else 1


def _round(value: float | None) -> float | None:
    """Round a PSNR for display; ``None`` stands for an infinite PSNR."""
    return None if value is None else round(value, 1)


def parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    root = argparse.ArgumentParser(prog="python -m within_core.verify", description=__doc__)
    root.formatter_class = argparse.RawDescriptionHelpFormatter
    commands = root.add_subparsers(dest="command", required=True)

    numerics = commands.add_parser("reference", help="FP32 recomputation of the chunk outputs")
    numerics.add_argument("run", type=Path, help="run folder, or its device folder")
    numerics.add_argument("--checkpoint", type=Path, required=True)
    numerics.add_argument("--output", type=Path, required=True, help="new report file")
    numerics.add_argument("--chunks", help="comma-separated chunk indices (default: all)")
    numerics.add_argument("--context", type=int, help="package context limit (default 16384)")
    numerics.add_argument("--threads", type=int, default=4, help="Torch CPU threads")
    numerics.set_defaults(run_command=run_reference)

    verdict = commands.add_parser("review", help="the placement and numerics verdict")
    verdict.add_argument("run", type=Path, help="run folder")
    verdict.add_argument(
        "--numerics", type=Path, nargs="+", help="reference reports covering every chunk"
    )
    verdict.add_argument("--output", type=Path, required=True, help="new verdict file")
    verdict.add_argument("--subsystem", default=review.SUBSYSTEM, help="signpost subsystem")
    verdict.set_defaults(run_command=run_review)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command."""
    arguments = parser().parse_args(argv)
    return arguments.run_command(arguments)


if __name__ == "__main__":
    sys.exit(main())
