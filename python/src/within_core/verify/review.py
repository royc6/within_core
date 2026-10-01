"""The ANE-EXEC-v1 verdict of one recorded run, computed from the exported trace tables.

Pipeline position: after the device recording and, for the numerics criterion, after
:mod:`within_core.verify.reference`. ``aneExecV1Passed`` is computed from the rows, never
written as a literal: PASS = P0 and C0 and C1 and C3 and C6; C2, C4 and C5 are recorded.

* P0: the recording is an all-process run on the device the recorder targeted, holding the
  app's PID; every native call of the report has exactly one completed
  PartInputMaterialize, PartInference and PartReadback interval, in that order, and none is
  unfinished; the trace bundles' bytes are unchanged since the recorder listed them.
* Call window: from the start of PartInference to the end of PartReadback.
* C0: the candidate (Neural Engine) arm has windows on every chunk of the package.
* C1: the app's own GPU intervals overlapping a window: candidate 0/N, GPU control N/N.
* C3: Neural Engine Prediction rows contained in a window: candidate N/N; the GPU control
  either 0/N or separated from the candidate by a label-permutation test with p < 0.001;
  each idle recording's prediction rate below 1% of the rate inside candidate windows.
  ``decidedBy`` names the branch of the GPU-control condition that held,
  ``gpuControlZero`` or ``permutation`` (``None`` if neither); the p-value is always
  reported, so a zero control is also shown to be significant.
* C4: the app's Core AI ``Function Execute`` rows per window (``CoreAIProfile`` in Xcode
  27.2, ``ODIEProfile`` in 27.0), recorded only.
* C5: per call, the union of the app's GPU intervals inside the window and its share of
  the window, recorded only.
* C2: needs a Time Profiler table, which this recording does not have; not recorded.
* C6: the candidate arm's outputs of every chunk against the FP32 same-representation
  recomputation from the run's own inputs: PSNR >= 35 dB and cosine >= 0.999 per output;
  the head also needs the same top-1 token and KL <= 1e-3 nats over the whole vocabulary.
  Without a numerics report C6 is not passed.

What this can and cannot prove: the proof is by exclusion plus coincidence, not
attribution. It cannot show which operation ran on the ANE; the idle control bounds, and
does not exclude, ANE rows from other processes inside a window; it covers only the
instrumented stages.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from within_core.verify.trace import (
    Row,
    bundle_manifest,
    interval,
    owned,
    read_export,
    recording_target,
    require,
    run_node,
)

STAGES = ("PartInputMaterialize", "PartInference", "PartReadback")
ARMS = ("neuralEngine", "gpu", "cpu")
CANDIDATE, GPU_CONTROL, CPU_CONTROL = ARMS
IDLE_RECORDINGS = ("idle-1", "idle-2")
SUBSYSTEM = "com.royc6.within"
MIN_WINDOWS = 90
PERMUTATION_ALPHA = 0.001
IDLE_RATE_FRACTION = 0.01
C6_PSNR_DB, C6_COSINE, C6_KL_NATS = 35.0, 0.999, 1e-3
PERMUTATION_ROUNDS = 20000
PERMUTATION_SEED = 0
PREDICTION = "Neural Engine Prediction"
EXECUTE = "Function Execute"
TABLES = {
    "signposts": (
        "os-signpost",
        {"time", "process", "event-type", "identifier", "name", "subsystem", "message"},
    ),
    "coreai": (None, {"start", "duration", "process", "event-id"}),
    "ane-hw-intervals": ("ane-hw-intervals", {"start", "duration", "event-label"}),
    "metal-gpu-intervals": ("metal-gpu-intervals", {"start", "duration", "process"}),
}
LIMITATIONS = [
    "The proof is by exclusion plus coincidence, not attribution: it cannot show which "
    "operation ran on the ANE.",
    "ANE hardware rows carry no process: the idle control bounds, and does not exclude, rows "
    "from other processes inside a window.",
    "Only the instrumented stages (PartInputMaterialize / PartInference / PartReadback) are "
    "covered.",
    "Compute preference, numerical acceptance, capture coverage and hardware placement are "
    "separate conclusions.",
]


def spans(rows: list[Row], pid: int, subsystem: str) -> tuple[list[dict], list[Row]]:
    """Pair the app's signpost Begin and End rows into completed intervals.

    Returns:
        The completed intervals (``stage``, ``message`` of the Begin row, ``start``,
        ``stop``) and the Begin rows that never ended.
    """
    pending: dict[tuple[str, str], Row] = {}
    completed = []
    for row in rows:
        if not owned(row, pid) or row["subsystem"]["label"] != subsystem:
            continue
        key = (row["name"]["label"], row["identifier"]["raw"])
        if row["event-type"]["raw"] == "Begin":
            require(key not in pending, f"Overlapping marker {key}")
            pending[key] = row
        elif row["event-type"]["raw"] == "End":
            require(key in pending, f"End marker without Begin {key}")
            first = pending.pop(key)
            completed.append(
                {
                    "stage": first["name"]["label"],
                    "message": first["message"]["label"],
                    "start": int(first["time"]["raw"]),
                    "stop": int(row["time"]["raw"]),
                }
            )
    return completed, list(pending.values())


def overlaps(row: Row, start: int, stop: int) -> bool:
    """Whether a row's interval overlaps ``[start, stop)``."""
    a, b = interval(row)
    return a < stop and b > start


def inside(row: Row, start: int, stop: int) -> bool:
    """Whether a row's interval lies within ``[start, stop]``."""
    a, b = interval(row)
    return start <= a and b <= stop


def union(rows: list[Row], start: int, stop: int) -> int:
    """Return the length of the union of the rows' intervals clipped to ``[start, stop]``."""
    total, cursor = 0, start
    clipped = sorted((max(start, interval(r)[0]), min(stop, interval(r)[1])) for r in rows)
    for a, b in clipped:
        if b > cursor:
            total += b - max(a, cursor)
            cursor = b
    return total


def permutation_p(
    candidate: list[int],
    control: list[int],
    rounds: int = PERMUTATION_ROUNDS,
    seed: int = PERMUTATION_SEED,
) -> float:
    """One-sided label-permutation test of the candidate arm against the GPU control.

    Each window is reduced to whether it contains a Neural Engine Prediction row. The
    statistic is the number of such candidate windows minus the number of such control
    windows. Under the null hypothesis that a window's arm label does not affect whether it
    contains a prediction row, the labels are exchangeable, so the statistic is compared
    with its value under ``rounds`` random relabelings drawn from NumPy's PCG64 generator
    seeded with ``seed``. The Monte Carlo p-value ``(1 + hits) / (1 + rounds)`` is never
    zero; with 20000 rounds its smallest value, about 5e-5, is below the 0.001 threshold.

    Args:
        candidate: Prediction rows per candidate window.
        control: Prediction rows per GPU-control window.
        rounds: Random relabelings.
        seed: Generator seed, fixed so that the report is reproducible.

    Returns:
        The p-value.
    """
    values = np.array([bool(x) for x in candidate + control], dtype=np.int64)
    split = len(candidate)
    observed = values[:split].sum() - values[split:].sum()
    shuffled = np.random.Generator(np.random.PCG64(seed)).permuted(
        np.tile(values, (rounds, 1)), axis=1
    )
    statistics = shuffled[:, :split].sum(axis=1) - shuffled[:, split:].sum(axis=1)
    hits = int(np.count_nonzero(statistics >= observed))
    return (hits + 1) / (rounds + 1)


def review(
    run: Path, numerics: list[Path] | None = None, subsystem: str = SUBSYSTEM
) -> dict[str, object]:
    """Compute the verdict of one recorded run.

    Args:
        run: The run folder (``result.json``, the trace bundles, ``exports/`` and
            ``device/<folder>/``).
        numerics: Reference reports (:mod:`within_core.verify.reference`) that together
            cover every chunk of this run; without them C6 is not passed.
        subsystem: Signpost subsystem of the host app's intervals.

    Returns:
        The report: ``aneExecV1Passed``, each criterion, per-arm summaries, every window,
        and the limitations.

    Raises:
        TraceError: If the recording is incomplete or inconsistent, or the markers do not
            cover every native call.
    """
    run = run.resolve()
    recorded = json.loads((run / "result.json").read_text())
    require(recorded.get("complete") is True, "Recording did not complete")
    device_folder = run / "device" / recorded["folder"]
    report = json.loads((device_folder / "report.json").read_text())
    pid = report["pid"]
    require(
        report["state"] == "complete" and pid == recorded["pid"],
        "Native report incomplete or PID differs",
    )
    exports = run / "exports"

    p0: dict[str, object] = {}
    main = run_node(exports / "recording-toc.xml")
    device = main.find("info/target/device")
    p0["device"] = device is not None and device.get("uuid") == recorded["device"]
    p0["allProcessTarget"] = recording_target(main, pid) == "allProcesses"
    p0["traceBytesClosed"] = all(
        bundle_manifest(run / f"{name}.trace") == rows
        for name, rows in recorded["traceManifests"].items()
    )
    toc_tables = main.findall("data/table")
    specs = TABLES | {"coreai": (recorded["coreAITable"], TABLES["coreai"][1])}
    tables = {
        name: read_export(exports / f"recording-{name}.xml", toc_tables, *spec)
        for name, spec in specs.items()
    }
    completed, unfinished = spans(tables["signposts"], pid, subsystem)
    p0["unfinishedSpans"] = len(unfinished)
    samples = report["samples"]
    labels = [sample["label"] for sample in samples]
    known = set(labels)
    require(len(known) == len(labels), "Duplicate native call labels")
    by_stage: dict[str, dict[str, dict]] = {stage: {} for stage in STAGES}
    extra = 0
    for span in completed:
        stage, message = span["stage"], span["message"]
        if stage not in by_stage or message not in known or message in by_stage[stage]:
            extra += 1
            continue
        by_stage[stage][message] = span
    p0["unexpectedSpans"] = extra
    covered = [label for label in labels if all(label in by_stage[s] for s in STAGES)]
    p0["callsWithAllStages"] = len(covered)
    p0["nativeCalls"] = len(labels)
    materialize, inference, readback = (by_stage[stage] for stage in STAGES)
    p0["stagesOrdered"] = all(
        materialize[label]["stop"]
        <= inference[label]["start"]
        <= inference[label]["stop"]
        <= readback[label]["start"]
        for label in covered
    )
    p0["passed"] = (
        p0["device"]
        and p0["allProcessTarget"]
        and p0["traceBytesClosed"]
        and not unfinished
        and not extra
        and len(covered) == len(labels)
        and p0["stagesOrdered"]
    )
    require(len(covered) == len(labels), f"P0 marker coverage: {p0}")

    gpu = [r for r in tables["metal-gpu-intervals"] if owned(r, pid)]
    ane = [r for r in tables["ane-hw-intervals"] if r["event-label"]["label"] == PREDICTION]
    execute = [r for r in tables["coreai"] if owned(r, pid) and r["event-id"]["label"] == EXECUTE]
    windows: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        start = inference[sample["label"]]["start"]
        stop = readback[sample["label"]]["stop"]
        overlapping = [r for r in gpu if overlaps(r, start, stop)]
        windows[sample["arm"]].append(
            {
                "chunk": sample["chunk"],
                "round": sample["round"],
                "wallNs": stop - start,
                "pidGPURows": len(overlapping),
                "pidGPUContainedRows": sum(inside(r, start, stop) for r in gpu),
                "pidGPUUnionNs": union(overlapping, start, stop),
                "aneRows": sum(inside(r, start, stop) for r in ane),
                "aneNs": sum(int(r["duration"]["raw"]) for r in ane if inside(r, start, stop)),
                "executeRows": sum(inside(r, start, stop) for r in execute),
            }
        )
    candidate, control, cpu = windows[CANDIDATE], windows[GPU_CONTROL], windows[CPU_CONTROL]
    n = len(candidate)
    require(
        n >= MIN_WINDOWS and len(control) == n and len(cpu) == n,
        f"Expected >= {MIN_WINDOWS} windows per arm, got {n}",
    )
    per_arm = {arm: _arm_summary(rows) for arm, rows in windows.items()}

    chunks = sorted({c["chunk"] for c in report["chunks"]})
    c0: dict[str, object] = {
        "chunks": chunks,
        "candidateChunks": sorted({r["chunk"] for r in candidate}),
    }
    c0["passed"] = c0["candidateChunks"] == chunks and len(chunks) == len(report["chunks"])
    c1: dict[str, object] = {
        "candidate": per_arm[CANDIDATE]["callsWithPIDGPU"],
        "gpuControl": per_arm[GPU_CONTROL]["callsWithPIDGPU"],
        "n": n,
    }
    c1["passed"] = c1["candidate"] == 0 and c1["gpuControl"] == n
    c3 = _coincidence(exports, candidate, control, per_arm, n)
    c4 = {
        "recorded": True,
        "candidateCallsWithPIDExecute": per_arm[CANDIDATE]["callsWithPIDExecute"],
        "n": n,
        "coreAITable": recorded["coreAITable"],
        "coreAIRows": len(tables["coreai"]),
        "note": "Recorded only; an empty Core AI table is not a platform conclusion.",
    }
    c5 = {
        arm: {"pidGPUUnionShare": row["pidGPUUnionShare"], "meanWindowMs": row["meanWindowMs"]}
        for arm, row in per_arm.items()
    }
    c2 = {"recorded": False, "status": "not recorded: this recording has no Time Profiler table"}
    c6 = (
        _numerics(numerics, device_folder, chunks)
        if numerics
        else {"passed": False, "status": "no numerics report"}
    )

    passed = p0["passed"] and c0["passed"] and c1["passed"] and c3["passed"] and c6["passed"]
    return {
        "contract": "ane-exec-v1-review",
        "aneExecV1Passed": passed,
        "pid": pid,
        "revision": report["revision"],
        "configuration": report["configuration"],
        "P0": p0,
        "C0": c0,
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "C4": c4,
        "C5": c5,
        "C6": c6,
        "perArm": per_arm,
        "windows": dict(windows),
        "limitations": LIMITATIONS,
    }


def _arm_summary(rows: list[dict]) -> dict[str, object]:
    """Counts and shares over one arm's windows."""
    wall = sum(r["wallNs"] for r in rows)
    return {
        "windows": len(rows),
        "callsWithPIDGPU": sum(r["pidGPURows"] > 0 for r in rows),
        "callsWithTemporalANE": sum(r["aneRows"] > 0 for r in rows),
        "callsWithPIDExecute": sum(r["executeRows"] > 0 for r in rows),
        "pidGPUUnionShare": sum(r["pidGPUUnionNs"] for r in rows) / wall,
        "meanWindowMs": wall / len(rows) / 1e6,
    }


def _coincidence(
    exports: Path, candidate: list[dict], control: list[dict], per_arm: dict, n: int
) -> dict[str, object]:
    """C3: Neural Engine rows inside candidate windows, against the controls and idle runs."""
    c3: dict[str, object] = {
        "candidate": per_arm[CANDIDATE]["callsWithTemporalANE"],
        "gpuControl": per_arm[GPU_CONTROL]["callsWithTemporalANE"],
        "cpuControl": per_arm[CPU_CONTROL]["callsWithTemporalANE"],
        "n": n,
    }
    p = permutation_p([r["aneRows"] for r in candidate], [r["aneRows"] for r in control])
    c3["permutationP"] = p
    c3["permutation"] = {
        "rounds": PERMUTATION_ROUNDS,
        "seed": PERMUTATION_SEED,
        "alpha": PERMUTATION_ALPHA,
        "statistic": "candidate windows with ANE rows minus GPU-control windows with ANE rows",
    }
    if c3["gpuControl"] == 0:
        c3["decidedBy"] = "gpuControlZero"
    elif p < PERMUTATION_ALPHA:
        c3["decidedBy"] = "permutation"
    else:
        c3["decidedBy"] = None
    seconds = sum(r["wallNs"] for r in candidate) / 1e9
    window_rate = sum(r["aneRows"] for r in candidate) / seconds
    idle = {}
    for name in IDLE_RECORDINGS:
        node = run_node(exports / f"{name}-toc.xml")
        rows = read_export(
            exports / f"{name}-ane-hw-intervals.xml",
            node.findall("data/table"),
            *TABLES["ane-hw-intervals"],
        )
        duration = float(node.findtext("info/summary/duration"))
        predictions = sum(r["event-label"]["label"] == PREDICTION for r in rows)
        idle[name] = {
            "seconds": duration,
            "predictions": predictions,
            "ratePerSecond": predictions / duration,
            "belowOnePercent": predictions / duration < IDLE_RATE_FRACTION * window_rate,
        }
    c3.update(windowRatePerSecond=window_rate, idle=idle)
    c3["passed"] = (
        c3["candidate"] == n
        and c3["decidedBy"] is not None
        and all(row["belowOnePercent"] for row in idle.values())
    )
    return c3


def _numerics(numerics: list[Path], device_folder: Path, chunks: list[int]) -> dict[str, object]:
    """C6 from reference reports computed on this run's own inputs and outputs."""
    parts = [json.loads(Path(path).read_text()) for path in numerics]
    report_digest = _sha256(device_folder / "report.json")
    for part in parts:
        require(
            Path(part["run"]).resolve() == device_folder.resolve()
            and part.get("reportSHA256", report_digest) == report_digest,
            "Numerics were not computed from this run",
        )
    all_rows = [row for part in parts for row in part["rows"]]
    rows = [r for r in all_rows if "scores" in r]
    scored = {(r["chunk"], r["output"]) for r in rows}
    require(len(scored) == len(rows), "A chunk output is scored twice")
    covered = sorted({r["chunk"] for r in rows})

    def tensor_passed(score: dict) -> bool:
        """Whether one output meets the PSNR and cosine thresholds; no error passes."""
        psnr = score["psnrDb"]
        return (psnr is None or psnr >= C6_PSNR_DB) and score["cosine"] >= C6_COSINE

    failing = [
        {
            "chunk": r["chunk"],
            "output": r["output"],
            "psnrDb": r["scores"][CANDIDATE]["psnrDb"],
            "cosine": r["scores"][CANDIDATE]["cosine"],
        }
        for r in rows
        if not tensor_passed(r["scores"][CANDIDATE])
    ]
    heads = [r["distribution"][CANDIDATE] for r in all_rows if "distribution" in r]
    head_passed = len(heads) == 1 and heads[0]["top1Equal"] and heads[0]["klNats"] <= C6_KL_NATS
    return {
        "chunksChecked": covered,
        "outputsChecked": len(rows),
        "failing": failing,
        "head": heads,
        "passed": covered == chunks and not failing and head_passed,
        "criterion": (
            f"C6-v2: candidate arm vs FP32 same-representation recomputation, PSNR >= "
            f"{C6_PSNR_DB} dB and cosine >= {C6_COSINE} per output; head top-1 equal and KL "
            f"<= {C6_KL_NATS} nats"
        ),
    }


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
