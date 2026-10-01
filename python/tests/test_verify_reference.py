"""The FP32 reference on a tiny checkpoint, and its metrics on hand-computed values."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from tiny import CAPACITY, Host, build_chunks, write_checkpoint
from within_core import export
from within_core.checkpoint import LocalCheckpoint
from within_core.chunks import ChunkLayout
from within_core.verify import reference
from within_core.verify.__main__ import parser

# Chunk 0 and the head: a layer chunk and the vocabulary distribution.
CHUNKS = (0, 7)


def test_psnr_and_scores_on_known_values() -> None:
    truth = np.array([1.0, -1.0], dtype=np.float32)
    # Error peak 0.1 and scaled MSE 0.5: 20 * log10(1 / 0.1) - 10 * log10(0.5).
    expected = 20.0 + 10 * math.log10(2.0)
    assert reference.psnr(np.array([1.0, -0.9], dtype=np.float32), truth) == pytest.approx(expected)
    assert reference.psnr(truth, truth) is None
    assert reference.psnr(truth, np.zeros(2, dtype=np.float32)) is None
    with pytest.raises(ValueError, match="identical shapes"):
        reference.psnr(truth, truth[:1])
    with pytest.raises(ValueError, match="Non-finite"):
        reference.psnr(np.array([1.0, np.nan], dtype=np.float32), truth)
    row = reference.score(np.array([2.0, 0.0]), np.array([1.0, 0.0]))
    assert (row["relativeL2"], row["cosine"], row["maxAbs"], row["referencePeak"]) == (1, 1, 1, 1)
    assert row["elements"] == 2


def test_distribution_score_on_known_values() -> None:
    logits = np.arange(12, dtype=np.float32)
    same = reference.distribution_score(logits, logits)
    assert same == {"top1Equal": True, "top1": 11, "klNats": 0.0, "top10Overlap": 10}
    swapped = logits.copy()
    swapped[[10, 11]] = swapped[[11, 10]]
    moved = reference.distribution_score(swapped, logits)
    assert not moved["top1Equal"] and moved["top1"] == 11 and moved["top10Overlap"] == 10
    p = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    q = np.exp(swapped - swapped.max()) / np.exp(swapped - swapped.max()).sum()
    assert moved["klNats"] == pytest.approx(float(np.sum(p * np.log(p / q))))


def phone_inputs(model: Gemma4ForCausalLM, layout: export.Layout) -> dict[str, np.ndarray]:
    """``step_1`` inputs as the host app builds them, rounded to FP16 like its dumps."""
    if isinstance(layout, ChunkLayout):
        host = Host(model, build_chunks(model, export.BOUNDARIES, export.HEAD_PARTS))
        values = host.inputs(torch.tensor([[7]]), position=9)
        values["token_identity"] = values["identity"][:, :, layout.start : layout.end]
    else:
        values = {"hidden": torch.randn(1, layout.hidden_size, 1, 1, generator=generator())}
    return {name: values[name].detach().half().numpy() for name in layout.input_names}


def generator() -> torch.Generator:
    """A seeded generator, so the phone stand-ins are the same on every run."""
    return torch.Generator().manual_seed(11)


def write_run(folder: Path, checkpoint: LocalCheckpoint, model: Gemma4ForCausalLM) -> None:
    """Write a device folder: inputs, and three arms' outputs of the chosen chunks.

    ``neuralEngine`` is the package's own FP16 arithmetic on the CPU, ``gpu`` the exact
    outputs rounded to FP16, and ``cpu`` the exact outputs plus noise as large as they are.
    """
    folder.mkdir()
    layouts = export.plan(checkpoint.config, CAPACITY)
    report: dict[str, object] = {"revision": "tiny-01", "chunks": []}
    for index, layout in enumerate(layouts):
        layers = [layout.start, layout.end] if isinstance(layout, ChunkLayout) else [42, 42]
        entry: dict[str, object] = {"chunk": index, "layers": layers}
        report["chunks"].append(entry)
        if index not in CHUNKS:
            continue
        inputs = phone_inputs(model, layout)
        report[f"inputs{index}"] = [
            {"input": name, "file": f"in-c{index}-{name}.f16", "shape": list(value.shape)}
            for name, value in inputs.items()
        ]
        for name, value in inputs.items():
            value.tofile(folder / f"in-c{index}-{name}.f16")
        chunk = (
            export.build_layer_chunk(checkpoint, layout)
            if isinstance(layout, ChunkLayout)
            else export.build_head_chunk(checkpoint, layout)
        )
        with torch.no_grad():
            fp16 = chunk(tuple(torch.from_numpy(inputs[n]) for n in layout.input_names))
        exact = reference.recompute(chunk, inputs)
        noise = generator()
        for name, value in zip(layout.step_outputs, fp16, strict=True):
            scale = float(np.sqrt(np.mean(np.square(exact[name]))))
            error = torch.randn(value.shape, generator=noise).numpy() * scale
            arms = {
                "neuralEngine": value.numpy(),
                "gpu": exact[name].astype(np.float16),
                "cpu": (exact[name] + error).astype(np.float16),
            }
            for arm, data in arms.items():
                file = f"c{index}-{arm}-{name}.f16"
                data.tofile(folder / file)
                row = {"output": name, "file": file, "shape": list(data.shape)}
                entry.setdefault(f"outputs{arm}", []).append(row)
    (folder / "report.json").write_text(json.dumps(report))


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory, reference: Gemma4ForCausalLM) -> Path:
    """A tiny checkpoint and a device folder recorded from it."""
    root = tmp_path_factory.mktemp("reference")
    pinned = write_checkpoint(root / "checkpoint", reference)
    (root / "pinned.json").write_text(json.dumps(pinned))
    write_run(root / "device", LocalCheckpoint(root / "checkpoint", pinned), reference)
    return root


def checkpoint_of(root: Path) -> LocalCheckpoint:
    """Open the tiny checkpoint against its pinned description."""
    return LocalCheckpoint(root / "checkpoint", json.loads((root / "pinned.json").read_text()))


def test_reference_scores_every_arm(tiny_run: Path) -> None:
    output = tiny_run / "numerics.json"
    result = reference.reference(
        tiny_run / "device", checkpoint_of(tiny_run), output, CHUNKS, CAPACITY
    )
    assert json.loads(output.read_text()) == json.loads(json.dumps(result))
    assert result["chunksDone"] == list(CHUNKS)
    assert result["run"] == str((tiny_run / "device").resolve())
    rows = [r for r in result["rows"] if "scores" in r]
    layouts = export.plan(checkpoint_of(tiny_run).config, CAPACITY)
    expected = [*layouts[0].step_outputs, *layouts[7].step_outputs]
    assert sorted(r["output"] for r in rows) == sorted(expected)
    for row in rows:
        scores = row["scores"]
        # FP16 execution of the same graph and weights stays close to the FP32 reference.
        assert scores["neuralEngine"]["psnrDb"] is None or scores["neuralEngine"]["psnrDb"] > 35
        assert scores["neuralEngine"]["cosine"] > 0.999
        assert scores["gpu"]["psnrDb"] is None or scores["gpu"]["psnrDb"] > 50
        assert scores["cpu"]["cosine"] < 0.999
    (head,) = [r for r in result["rows"] if "distribution" in r]
    assert head["output"] == "logits (all 64)"
    assert head["distribution"]["neuralEngine"]["top1Equal"]
    assert head["distribution"]["neuralEngine"]["klNats"] < 1e-3
    assert head["distribution"]["gpu"]["top10Overlap"] == 10


def test_reference_refuses_another_context_and_an_existing_output(tiny_run: Path) -> None:
    checkpoint = checkpoint_of(tiny_run)
    with pytest.raises(ValueError, match="inputs differ"):
        reference.reference(tiny_run / "device", checkpoint, tiny_run / "other.json", [0], 32)
    (tiny_run / "exists.json").write_text("{}")
    with pytest.raises(FileExistsError):
        reference.reference(tiny_run / "device", checkpoint, tiny_run / "exists.json", [7])


def test_reference_command_line_defaults() -> None:
    arguments = parser().parse_args(["reference", "run", "--checkpoint", "c", "--output", "o"])
    assert (arguments.chunks, arguments.context, arguments.threads) == (None, None, 4)
