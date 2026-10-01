"""The placement verdict on small synthetic recordings: the pass and each way to fail."""

import json
import math
from pathlib import Path

import pytest

from synthetic_trace import (
    MS,
    OTHER_PID,
    PID,
    PREDICTION,
    SUBSYSTEM,
    SyntheticRun,
    numerics,
)
from within_core.verify import review
from within_core.verify.__main__ import main
from within_core.verify.trace import TraceError, read_export, run_node

CANDIDATE = "neuralEngine"


def verdict(run: SyntheticRun, tmp_path: Path, with_numerics: bool = True, **head: object) -> dict:
    """Write ``run`` and review it, with a passing reference report unless told otherwise."""
    root = run.write(tmp_path / "run")
    parts = [numerics(root, run.chunks, **head)] if with_numerics else None
    return review.review(root, parts, SUBSYSTEM)


def test_a_clean_recording_passes(tmp_path: Path) -> None:
    result = verdict(SyntheticRun(), tmp_path)
    assert result["aneExecV1Passed"]
    assert result["P0"] == {
        "device": True,
        "allProcessTarget": True,
        "traceBytesClosed": True,
        "unfinishedSpans": 0,
        "unexpectedSpans": 0,
        "callsWithAllStages": 270,
        "nativeCalls": 270,
        "stagesOrdered": True,
        "passed": True,
    }
    assert result["C0"] == {"chunks": [0, 1], "candidateChunks": [0, 1], "passed": True}
    # Another process's GPU work overlaps every window and is not the app's.
    assert result["C1"] == {"candidate": 0, "gpuControl": 90, "n": 90, "passed": True}
    c3 = result["C3"]
    assert (c3["candidate"], c3["gpuControl"], c3["cpuControl"]) == (90, 0, 0)
    assert c3["decidedBy"] == "gpuControlZero" and c3["passed"]
    assert c3["permutationP"] == pytest.approx(1 / 20001)
    assert c3["windowRatePerSecond"] == pytest.approx(1 / 10.1e-3)
    assert c3["idle"]["idle-1"] == {
        "seconds": 30.0,
        "predictions": 0,
        "ratePerSecond": 0.0,
        "belowOnePercent": True,
    }
    assert result["C4"]["candidateCallsWithPIDExecute"] == 90 and result["C4"]["coreAIRows"] == 90
    assert result["C5"][CANDIDATE]["pidGPUUnionShare"] == 0
    assert result["C5"]["gpu"]["pidGPUUnionShare"] == pytest.approx(6 / 10.1)
    assert result["C5"]["gpu"]["meanWindowMs"] == pytest.approx(10.1)
    assert not result["C2"]["recorded"]
    assert result["C6"]["passed"] and result["C6"]["outputsChecked"] == 4
    assert result["limitations"] == review.LIMITATIONS
    assert len(result["windows"][CANDIDATE]) == 90
    assert result["perArm"]["cpu"]["callsWithTemporalANE"] == 0


def test_the_statement_of_what_this_proves_is_kept_verbatim() -> None:
    statement = (
        "What this can and cannot prove: the proof is by exclusion plus coincidence, not\n"
        "attribution. It cannot show which operation ran on the ANE; the idle control bounds, "
        "and\ndoes not exclude, ANE rows from other processes inside a window; it covers only "
        "the\ninstrumented stages."
    )
    assert " ".join(statement.split()) in " ".join(review.__doc__.split())
    assert review.LIMITATIONS[0].startswith("The proof is by exclusion plus coincidence")


def test_unfinished_and_unexpected_spans_fail_p0(tmp_path: Path) -> None:
    run = SyntheticRun()
    late = max(t for t, *_ in run.signposts) + MS
    run.signposts.append((late, "Begin", "PartInference", "gpu c9 r0", PID, SUBSYSTEM))
    run.signposts.append((late + MS, "Begin", "PartReadback", "gpu c9 r0", PID, SUBSYSTEM))
    run.signposts.append((late + 2 * MS, "End", "PartReadback", "", PID, SUBSYSTEM))
    result = verdict(run, tmp_path)
    assert (result["P0"]["unfinishedSpans"], result["P0"]["unexpectedSpans"]) == (1, 1)
    assert not result["P0"]["passed"] and not result["aneExecV1Passed"]


def test_other_processes_and_subsystems_are_ignored(tmp_path: Path) -> None:
    run = SyntheticRun()
    late = max(t for t, *_ in run.signposts) + MS
    run.signposts.append((late, "Begin", "PartInference", "gpu c9 r0", OTHER_PID, SUBSYSTEM))
    run.signposts.append((late, "Begin", "PartInference", "gpu c9 r0", PID, "com.example.other"))
    assert verdict(run, tmp_path)["aneExecV1Passed"]


def test_a_call_without_all_stages_is_refused(tmp_path: Path) -> None:
    run = SyntheticRun()
    run.signposts = [s for s in run.signposts if not (s[2] == "PartReadback" and s[0] < 30 * MS)]
    with pytest.raises(TraceError, match="P0 marker coverage"):
        verdict(run, tmp_path)


def test_an_end_without_begin_is_refused(tmp_path: Path) -> None:
    run = SyntheticRun()
    run.signposts.append((1, "End", "PartInference", "", PID, SUBSYSTEM))
    with pytest.raises(TraceError, match="End marker without Begin"):
        verdict(run, tmp_path)


def test_stages_out_of_order_fail_p0(tmp_path: Path) -> None:
    run = SyntheticRun()
    first = run.samples[0]["label"]
    begin = next(i for i, s in enumerate(run.signposts) if s[1] == "Begin" and s[3] == first)
    t, kind, name, message, pid, subsystem = run.signposts[begin + 1]
    # The materialize interval now ends after inference began.
    run.signposts[begin + 1] = (t + MS, kind, name, message, pid, subsystem)
    result = verdict(run, tmp_path)
    assert not result["P0"]["stagesOrdered"] and not result["aneExecV1Passed"]


def test_changed_trace_bytes_fail_p0(tmp_path: Path) -> None:
    root = SyntheticRun().write(tmp_path / "run")
    (root / "recording.trace" / "Trace1.run").write_text("changed")
    result = review.review(root, [numerics(root, 2)], SUBSYSTEM)
    assert not result["P0"]["traceBytesClosed"] and not result["aneExecV1Passed"]


def test_another_device_or_a_single_process_target_fails_p0(tmp_path: Path) -> None:
    run = SyntheticRun()
    run.device = "ANOTHER-DEVICE"
    assert not verdict(run, tmp_path / "device")["P0"]["device"]
    run = SyntheticRun()
    run.target = f'<process pid="{PID}"/>'
    result = verdict(run, tmp_path / "target")
    assert not result["P0"]["allProcessTarget"] and not result["aneExecV1Passed"]


def test_a_chunk_without_candidate_windows_fails_c0(tmp_path: Path) -> None:
    run = SyntheticRun()
    run.chunks = 3
    result = verdict(run, tmp_path)
    assert result["C0"] == {"chunks": [0, 1, 2], "candidateChunks": [0, 1], "passed": False}
    assert not result["aneExecV1Passed"]


def test_app_gpu_work_in_a_candidate_window_fails_c1(tmp_path: Path) -> None:
    run = SyntheticRun()
    start, _ = run.window(CANDIDATE, 4)
    run.gpu.append((start - MS, 2 * MS, PID))  # overlapping is enough
    result = verdict(run, tmp_path)
    assert result["C1"]["candidate"] == 1 and not result["C1"]["passed"]
    assert not result["aneExecV1Passed"]


def test_a_gpu_control_window_without_gpu_work_fails_c1(tmp_path: Path) -> None:
    run = SyntheticRun()
    start, stop = run.window("gpu", 0)
    run.gpu = [g for g in run.gpu if not (g[2] == PID and start <= g[0] < stop)]
    result = verdict(run, tmp_path)
    assert result["C1"]["gpuControl"] == 89 and not result["aneExecV1Passed"]


def test_a_candidate_window_without_a_prediction_fails_c3(tmp_path: Path) -> None:
    run = SyntheticRun()
    start, stop = run.window(CANDIDATE, 7)
    run.ane = [a for a in run.ane if not start <= a[0] < stop]
    result = verdict(run, tmp_path)
    assert result["C3"]["candidate"] == 89 and not result["C3"]["passed"]
    assert not result["aneExecV1Passed"]


def test_a_prediction_crossing_the_window_edge_does_not_count(tmp_path: Path) -> None:
    run = SyntheticRun()
    start, stop = run.window(CANDIDATE, 7)
    run.ane = [a for a in run.ane if not start <= a[0] < stop]
    run.ane.append((stop - MS, 2 * MS, PREDICTION))
    assert verdict(run, tmp_path)["C3"]["candidate"] == 89


def test_gpu_control_predictions_go_to_the_permutation_test(tmp_path: Path) -> None:
    run = SyntheticRun()
    for index in range(0, 90, 9):  # 10 of the 90 GPU-control windows
        start, _ = run.window("gpu", index)
        run.ane.append((start + MS, MS, PREDICTION))
    c3 = verdict(run, tmp_path)["C3"]
    assert c3["gpuControl"] == 10
    assert c3["decidedBy"] == "permutation" and c3["permutationP"] < 0.001 and c3["passed"]
    assert c3["permutation"]["rounds"] == review.PERMUTATION_ROUNDS


def test_a_gpu_control_like_the_candidate_fails_the_permutation_test(tmp_path: Path) -> None:
    run = SyntheticRun()
    for index in range(90):
        start, _ = run.window("gpu", index)
        run.ane.append((start + MS, MS, PREDICTION))
    result = verdict(run, tmp_path)
    c3 = result["C3"]
    assert c3["gpuControl"] == 90 and c3["permutationP"] == 1.0
    assert c3["decidedBy"] is None and not c3["passed"] and not result["aneExecV1Passed"]


def test_a_busy_idle_recording_fails_c3(tmp_path: Path) -> None:
    run = SyntheticRun()
    # The candidate windows see about 99 predictions per second; 1% of that is about 1/s.
    run.idle["idle-2"] = (30.0, [(i * 100 * MS, MS, PREDICTION) for i in range(60)])
    result = verdict(run, tmp_path)
    idle = result["C3"]["idle"]["idle-2"]
    assert idle["predictions"] == 60 and idle["ratePerSecond"] == 2.0
    assert not idle["belowOnePercent"] and not result["C3"]["passed"]
    assert not result["aneExecV1Passed"]


def test_too_few_windows_are_refused(tmp_path: Path) -> None:
    with pytest.raises(TraceError, match="Expected >= 90 windows per arm, got 88"):
        verdict(SyntheticRun(rounds=44), tmp_path)


def test_numerics_decide_c6(tmp_path: Path) -> None:
    run = SyntheticRun()
    assert verdict(run, tmp_path / "none", with_numerics=False)["C6"] == {
        "passed": False,
        "status": "no numerics report",
    }
    c6 = verdict(run, tmp_path / "kl", klNats=2e-3)["C6"]
    assert not c6["passed"] and not c6["failing"]
    assert not verdict(run, tmp_path / "top1", top1Equal=False)["C6"]["passed"]
    root = run.write(tmp_path / "psnr")
    path = numerics(root, 2)
    data = json.loads(path.read_text())
    data["rows"][0]["scores"][CANDIDATE]["psnrDb"] = 34.9
    data["rows"][1]["scores"][CANDIDATE]["psnrDb"] = None  # infinite: no error
    data["rows"][2]["scores"][CANDIDATE]["cosine"] = 0.998
    path.write_text(json.dumps(data))
    c6 = review.review(root, [path], SUBSYSTEM)["C6"]
    assert [(f["chunk"], f["output"]) for f in c6["failing"]] == [(0, "hidden"), (1, "hidden")]
    assert not c6["passed"]


def test_numerics_must_cover_every_chunk_once_from_this_run(tmp_path: Path) -> None:
    root = SyntheticRun().write(tmp_path / "run")
    path = numerics(root, 2)
    data = json.loads(path.read_text())
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(data | {"rows": data["rows"][2:]}))
    c6 = review.review(root, [missing], SUBSYSTEM)["C6"]
    assert c6["chunksChecked"] == [1] and not c6["passed"]
    with pytest.raises(TraceError, match="scored twice"):
        review.review(root, [path, path], SUBSYSTEM)
    other = tmp_path / "other.json"
    other.write_text(json.dumps(data | {"run": str(tmp_path)}))
    with pytest.raises(TraceError, match="not computed from this run"):
        review.review(root, [other], SUBSYSTEM)
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(data | {"reportSHA256": "0" * 64}))
    with pytest.raises(TraceError, match="not computed from this run"):
        review.review(root, [stale], SUBSYSTEM)


def exact_p(candidate: int, control: int, n: int) -> float:
    """Exact one-sided p-value of the permutation statistic (hypergeometric tail)."""
    total = candidate + control
    observed = candidate - control
    tail = 0
    for k in range(max(0, total - n), min(n, total) + 1):
        if k - (total - k) >= observed:
            tail += math.comb(total, k) * math.comb(2 * n - total, n - k)
    return tail / math.comb(2 * n, n)


@pytest.mark.parametrize(("candidate", "control"), [(12, 8), (15, 5), (10, 10)])
def test_permutation_p_matches_the_exact_tail(candidate: int, control: int) -> None:
    n = 20
    first = [1] * candidate + [0] * (n - candidate)
    second = [3] * control + [0] * (n - control)  # counts are reduced to presence
    p = review.permutation_p(first, second)
    assert p == pytest.approx(exact_p(candidate, control, n), abs=0.01)
    assert p == review.permutation_p(first, second)  # seeded


def test_permutation_p_bounds() -> None:
    assert review.permutation_p([1] * 90, [0] * 90) == 1 / 20001
    assert review.permutation_p([1] * 90, [1] * 90) == 1.0


def test_read_export_refuses_another_table(tmp_path: Path) -> None:
    root = SyntheticRun().write(tmp_path / "run")
    exports = root / "exports"
    tables = run_node(exports / "recording-toc.xml").findall("data/table")
    rows = read_export(exports / "recording-coreai.xml", tables, "CoreAIProfile", {"event-id"})
    assert rows[0]["process"]["label"] == f"host ({PID})"
    assert rows[0]["event-id"]["raw"] == "Function Execute"
    with pytest.raises(TraceError, match="differs from the TOC schema"):
        read_export(exports / "recording-coreai.xml", tables, "ODIEProfile", set())
    with pytest.raises(TraceError, match="Missing or duplicate evidence columns"):
        read_export(exports / "recording-coreai.xml", tables, "CoreAIProfile", {"thread"})
    text = (exports / "recording-signposts.xml").read_text()
    dynamic = tmp_path / "dynamic.xml"
    dynamic.write_text(text.replace("table[3]", "table[2]"))
    with pytest.raises(TraceError, match="Wrong signpost table selection"):
        read_export(dynamic, tables, "os-signpost", set())
    twice = tmp_path / "twice.xml"
    node = text[text.index("<node") : text.index("</trace-query-result>")]
    twice.write_text(text.replace("</trace-query-result>", node + "</trace-query-result>"))
    with pytest.raises(TraceError, match="never concatenated"):
        read_export(twice, tables, "os-signpost", set())


def test_command_line_writes_the_verdict_and_sets_the_status(tmp_path: Path) -> None:
    root = SyntheticRun().write(tmp_path / "run")
    path = numerics(root, 2)
    arguments = ["review", str(root), "--subsystem", SUBSYSTEM, "--numerics", str(path)]
    assert main([*arguments, "--output", str(tmp_path / "pass.json")]) == 0
    assert json.loads((tmp_path / "pass.json").read_text())["aneExecV1Passed"]
    with pytest.raises(FileExistsError):
        main([*arguments, "--output", str(tmp_path / "pass.json")])
    failing = ["review", str(root), "--subsystem", SUBSYSTEM, "--output", str(tmp_path / "f.json")]
    assert main(failing) == 1
