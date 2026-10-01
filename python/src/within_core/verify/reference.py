"""FP32 same-representation recomputation of each chunk's outputs from a run's own inputs.

Pipeline position: between the device recording and the numerics criterion (C6) of
:mod:`within_core.verify.review`. The host app writes, for round 0 of every chunk, the
``step_1`` input bytes it fed and each arm's output bytes (FP16). This module rebuilds each
chunk exactly as :mod:`within_core.export` builds it for the package (the same graph and
the same compressed weights), converts it to FP32 and runs it on the phone's own input
bytes: the reference for this representation, free of FP16 rounding but not of
quantization. Every arm's outputs are then scored against it; the head's 16 slices are
also joined into one distribution over the whole vocabulary.

PyTorch on the CPU only. The checkpoint is read locally; nothing is downloaded.
"""

import hashlib
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from within_core.checkpoint import LocalCheckpoint
from within_core.chunks import ChunkLayout, HeadChunk, LayerChunk, Layout
from within_core.export import CONTEXT, build_head_chunk, build_layer_chunk, plan

ARMS = ("neuralEngine", "gpu", "cpu")
CONTRACT = "ane-exec-v1-reference"


def psnr(actual: np.ndarray, reference: np.ndarray) -> float | None:
    """Return the peak signal-to-noise ratio of ``actual`` against ``reference`` in dB.

    The peak is the reference's largest magnitude. The mean squared error is computed on
    the error scaled by its own peak, so that squaring neither overflows nor underflows.

    Returns:
        The PSNR, or ``None`` when it is infinite (no error, or an all-zero reference).

    Raises:
        ValueError: If the shapes differ, a tensor is empty or a value is not finite.
    """
    if actual.shape != reference.shape or not actual.size:
        raise ValueError("Expected nonempty tensors with identical shapes")
    x, y = actual.astype(np.float64), reference.astype(np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite values cannot be scored")
    delta = x - y
    error_peak = float(np.max(np.abs(delta)))
    reference_peak = float(np.max(np.abs(y)))
    if error_peak == 0 or reference_peak == 0:
        return None
    scaled_mse = float(np.mean(np.square(delta / error_peak)))
    return 20 * (math.log10(reference_peak) - math.log10(error_peak)) - 10 * math.log10(scaled_mse)


def score(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    """Score one output tensor of an arm against the reference.

    Returns:
        ``psnrDb``, ``elements``, ``relativeL2`` (the error's L2 norm over the reference's),
        ``cosine``, ``maxAbs`` (largest absolute error) and ``referencePeak``.
    """
    a, b = actual.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
    return {
        "psnrDb": psnr(actual.astype(np.float32), reference.astype(np.float32)),
        "elements": int(a.size),
        "relativeL2": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)),
        "cosine": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
        "maxAbs": float(np.max(np.abs(a - b))),
        "referencePeak": float(np.max(np.abs(b))),
    }


def distribution_score(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    """Compare the softmax distributions of two logit vectors over the whole vocabulary.

    Returns:
        ``top1Equal``, the reference's ``top1`` token, ``klNats`` (KL divergence of the
        actual distribution from the reference's, in nats) and ``top10Overlap``.
    """
    a, b = actual.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
    pa, pb = np.exp(a - a.max()), np.exp(b - b.max())
    pa, pb = pa / pa.sum(), pb / pb.sum()
    # The tiny offset keeps log(0) finite for tokens whose probability underflows.
    kl = float(np.sum(pb * (np.log(pb + 1e-300) - np.log(pa + 1e-300))))
    return {
        "top1Equal": bool(a.argmax() == b.argmax()),
        "top1": int(b.argmax()),
        "klNats": kl,
        "top10Overlap": len(set(np.argsort(-a)[:10]) & set(np.argsort(-b)[:10])),
    }


def device_folder(run: Path) -> Path:
    """Return the folder with the host app's ``report.json`` and dumps.

    Args:
        run: A run folder (its ``result.json`` names ``device/<folder>``) or that device
            folder itself.
    """
    if (run / "result.json").is_file():
        return run / "device" / json.loads((run / "result.json").read_text())["folder"]
    return run


def load(folder: Path, row: dict) -> np.ndarray:
    """Read one dumped FP16 tensor described by a report row (``file``, ``shape``)."""
    return np.fromfile(folder / row["file"], dtype=np.float16).reshape(row["shape"])


def recompute(
    chunk: LayerChunk | HeadChunk, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Run ``chunk`` in FP32 on recorded inputs.

    The chunk is converted to FP32 in place: compressed weights stay compressed and are
    expanded in FP32, so the weights are the package's and only the arithmetic changes.

    Args:
        chunk: A chunk built for the package.
        inputs: The recorded ``step_1`` inputs by name.

    Returns:
        The ``step_1`` outputs by name.
    """
    layout = chunk.layout
    chunk.float()
    values = tuple(torch.from_numpy(inputs[n].astype(np.float32)) for n in layout.input_names)
    with torch.no_grad():
        produced = chunk(values)
    return {
        name: tensor.numpy() for name, tensor in zip(layout.step_outputs, produced, strict=True)
    }


def score_chunk(
    folder: Path, report: dict, index: int, layout: Layout, truth: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    """Score every arm's recorded outputs of chunk ``index`` against the FP32 outputs.

    Returns:
        One row per output (``chunk``, ``layers``, ``output``, ``scores`` per arm) and, for
        the head, one more row (``distribution`` per arm) over the whole vocabulary.
    """
    recorded = report["chunks"][index]
    start, end = recorded["layers"]
    rows: list[dict[str, object]] = []
    arms = [arm for arm in ARMS if f"outputs{arm}" in recorded]

    def output(arm: str, name: str) -> np.ndarray:
        """One recorded output of an arm, as FP32."""
        row = next(r for r in recorded[f"outputs{arm}"] if r["output"] == name)
        return load(folder, row).astype(np.float32)

    for name in sorted(truth):
        actual = {arm: output(arm, name) for arm in arms}
        reference = truth[name].reshape(actual[arms[0]].shape)
        rows.append(
            {
                "chunk": index,
                "layers": [start, end],
                "output": name,
                "scores": {arm: score(value, reference) for arm, value in actual.items()},
            }
        )
    if not isinstance(layout, ChunkLayout):
        # The host joins the slices in name order; so does the distribution.
        names = layout.step_outputs
        exact = np.concatenate([np.ravel(truth[n]) for n in names])
        full = {arm: np.concatenate([np.ravel(output(arm, n)) for n in names]) for arm in arms}
        rows.append(
            {
                "chunk": index,
                "output": f"logits (all {exact.size})",
                "distribution": {arm: distribution_score(v, exact) for arm, v in full.items()},
            }
        )
    return rows


def reference(
    run: Path,
    checkpoint: LocalCheckpoint,
    output: Path,
    chunks: Sequence[int] | None = None,
    context: int = CONTEXT,
) -> dict[str, object]:
    """Recompute and score the chosen chunks of a run, saving the report after each chunk.

    Saving after each chunk lets a long run stop and continue with the remaining chunks
    into another file; :func:`within_core.verify.review.review` accepts several files.

    Args:
        run: The run folder or its device folder.
        checkpoint: The checkpoint the package was converted from.
        output: The new report file.
        chunks: Chunk indices; all chunks of the run by default.
        context: Context limit of the package that ran.

    Returns:
        The report: ``contract``, ``run`` (the device folder), ``reportSHA256`` (of its
        ``report.json``), ``revision``, ``rows``, ``chunksDone`` and ``seconds`` per chunk.

    Raises:
        FileExistsError: If ``output`` exists.
        ValueError: If a recorded chunk's layers or input shapes differ from the plan.
    """
    if output.exists():
        raise FileExistsError(f"Use a new output file: {output}")
    folder = device_folder(run).resolve()
    report_path = folder / "report.json"
    report = json.loads(report_path.read_text())
    layouts = plan(checkpoint.config, context)
    selected = range(len(report["chunks"])) if chunks is None else chunks
    rows: list[dict[str, object]] = []
    done: list[int] = []
    seconds: dict[str, float] = {}
    result = {
        "contract": CONTRACT,
        "run": str(folder),
        "reportSHA256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "revision": report["revision"],
        "rows": rows,
        "chunksDone": done,
        "seconds": seconds,
    }
    for index in selected:
        layout = layouts[index]
        expected = (
            [layout.start, layout.end] if isinstance(layout, ChunkLayout) else [layout.layers] * 2
        )
        if report["chunks"][index]["layers"] != expected:
            raise ValueError(f"Chunk {index} covers other layers than the plan")
        inputs = {row["input"]: load(folder, row) for row in report[f"inputs{index}"]}
        shapes = {name: list(value.shape) for name, value in inputs.items()}
        if shapes != layout.shapes(1):
            raise ValueError(f"Chunk {index} inputs differ from the plan (check --context)")
        began = time.monotonic()
        if isinstance(layout, ChunkLayout):
            chunk: LayerChunk | HeadChunk = build_layer_chunk(checkpoint, layout)
        else:
            chunk = build_head_chunk(checkpoint, layout)
        truth = recompute(chunk, inputs)
        del chunk
        rows.extend(score_chunk(folder, report, index, layout, truth))
        done.append(index)
        seconds[str(index)] = round(time.monotonic() - began, 1)
        output.write_text(json.dumps(result, indent=1) + "\n")
    return result
