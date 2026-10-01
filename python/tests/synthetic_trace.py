"""Test helper: a small recorded run, written the way the recorder and ``xctrace export`` do.

A :class:`SyntheticRun` holds the events of a placement recording as plain Python values,
which a test changes before :meth:`SyntheticRun.write` lays out the run folder:
``result.json``, three ``.trace`` bundles, the host app's ``device/<folder>/report.json``,
and ``exports/`` with the TOC and table exports. The exports follow the shape of real
``xctrace export`` output: one ``node`` whose ``xpath`` names the table's index in the TOC,
a ``schema`` of ``col``/``mnemonic`` entries, and rows whose repeated values are written
once with an ``id`` (sometimes nested inside another column's value, as ``xctrace`` does
for processes) and later as ``ref``.
"""

import json
from pathlib import Path
from xml.sax.saxutils import quoteattr

from within_core.verify.trace import bundle_manifest

PID = 4242
OTHER_PID = 76
DEVICE = "SYNTHETIC-DEVICE-0001"
SUBSYSTEM = "com.example.host"
ARMS = ("neuralEngine", "gpu", "cpu")
PREDICTION = "Neural Engine Prediction"
MS = 1_000_000
SLOT = 20 * MS
# 1-based TOC indices of the recording's tables; the first signpost table is the
# dynamically traced one that the reviewer must not take.
TABLE_INDEX = {"signposts": 3, "ane-hw-intervals": 4, "coreai": 5, "metal-gpu-intervals": 6}
TOC_TABLES = (
    '<table schema="metal-object-label" documentation="Metal Object Label"/>',
    '<table target-pid="ALL" dynamic-tracing-enabled-subsystems="&quot;com.example.dynamic&quot;"'
    ' category="PointsOfInterest" schema="os-signpost"/>',
    '<table schema="os-signpost" category="PointsOfInterest" target-pid="ALL"/>',
    '<table schema="ane-hw-intervals" documentation="Denotes activity in the ANE"/>',
    '<table schema="CoreAIProfile"/>',
    '<table schema="metal-gpu-intervals" target-pid="ALL"/>',
)


class Values:
    """Writes each distinct value once with an ``id`` and afterwards as a ``ref``."""

    def __init__(self) -> None:
        self.ids: dict[tuple[str, str, str], int] = {}

    def __call__(self, tag: str, text: str, fmt: str | None = None, inner: str = "") -> str:
        """Return the element for a value, defining it on first use."""
        key = (tag, text, fmt if fmt is not None else text)
        if key in self.ids:
            return f'<{tag} ref="{self.ids[key]}"/>'
        self.ids[key] = len(self.ids) + 1
        label = quoteattr(key[2])
        return f'<{tag} id="{self.ids[key]}" fmt={label}>{inner or text}</{tag}>'


def process(values: Values, pid: int) -> str:
    """A process value, as ``xctrace`` writes it (name and PID, with a nested ``pid``)."""
    name = "host" if pid == PID else "backboardd"
    inner = values("pid", str(pid)) + values("device-session", "TODO")
    return values("process", "", f"{name} ({pid})", inner)


def export(index: int, schema: str, columns: list[str], rows: list[str]) -> str:
    """One exported table: a node naming the TOC index, its schema and its rows."""
    cols = "".join(
        f"<col><mnemonic>{c}</mnemonic><name>{c.title()}</name>"
        f"<engineering-type>{c}</engineering-type></col>"
        for c in columns
    )
    body = "".join(f"<row>{row}</row>\n" for row in rows)
    return (
        '<?xml version="1.0"?>\n<trace-query-result>\n'
        f"<node xpath='//trace-toc[1]/run[1]/data[1]/table[{index}]'>"
        f'<schema name="{schema}">{cols}</schema>{body}</node></trace-query-result>\n'
    )


def toc(device: str, target: str, seconds: float, tables: tuple[str, ...]) -> str:
    """A TOC export with one run."""
    return (
        '<?xml version="1.0"?>\n<trace-toc>\n<run number="1"><info><target>'
        f'<device platform="iOS" model="Synthetic" name="Synthetic" uuid="{device}"/>{target}'
        f"</target><summary><duration>{seconds}</duration><template-name>Core AI</template-name>"
        f'</summary></info><processes><process name="host" pid="{PID}" path="Unknown"/>'
        f'<process name="backboardd" pid="{OTHER_PID}" path="Unknown"/></processes>'
        f"<data>{''.join(tables)}</data></run></trace-toc>\n"
    )


class SyntheticRun:
    """The events of one recording; every list may be changed before :meth:`write`.

    Attributes:
        samples: The host app's native calls (``arm``, ``chunk``, ``round``, ``label``).
        signposts: ``(time, event type, name, message, pid, subsystem)`` rows in time order.
        windows: Each label's call window, from PartInference begin to PartReadback end.
        ane: Neural Engine rows ``(start, duration, label)``.
        gpu: GPU rows ``(start, duration, pid)``.
        coreai: Core AI rows ``(start, duration, pid, event)``.
        idle: Per idle recording, its duration in seconds and its ANE rows.
    """

    def __init__(self, chunks: int = 2, rounds: int = 45) -> None:
        """Lay out ``chunks`` x ``rounds`` calls per arm, as the host app runs them.

        Each chunk runs all its rounds before the next; each round runs the three arms,
        with the arm order rotated per round. A call takes 10.1 ms: the candidate's window
        holds one Neural Engine Prediction, the GPU control's window one GPU interval of
        the app, and the CPU control's window neither. Another process's GPU work overlaps
        every window, and an ANE row that is not a prediction sits in every CPU window.
        """
        self.device = DEVICE
        self.target = "<all-processes/>"
        self.samples: list[dict[str, object]] = []
        self.signposts: list[tuple[int, str, str, str, int, str]] = []
        self.windows: dict[str, tuple[int, int]] = {}
        self.ane: list[tuple[int, int, str]] = []
        self.gpu: list[tuple[int, int, int]] = []
        self.coreai: list[tuple[int, int, int, str]] = []
        self.idle: dict[str, tuple[float, list[tuple[int, int, str]]]] = {
            "idle-1": (30.0, []),
            "idle-2": (30.0, []),
        }
        self.chunks = chunks
        slot = 0
        for chunk in range(chunks):
            for round_ in range(rounds):
                shift = round_ % len(ARMS)
                for arm in ARMS[shift:] + ARMS[:shift]:
                    self.call(arm, chunk, round_, slot * SLOT + 5 * MS)
                    slot += 1

    def call(self, arm: str, chunk: int, round_: int, t: int) -> None:
        """Add one native call starting at ``t`` and its hardware activity."""
        label = f"{arm} c{chunk} r{round_}"
        self.samples.append({"arm": arm, "chunk": chunk, "round": round_, "label": label})
        stages = (
            ("PartInputMaterialize", t, t + MS // 10),
            ("PartInference", t + MS // 5, t + 10 * MS),
            ("PartReadback", t + 10 * MS + MS // 10, t + 10 * MS + 3 * MS // 10),
        )
        for name, begin, end in stages:
            self.signposts.append((begin, "Begin", name, label, PID, SUBSYSTEM))
            self.signposts.append((end, "End", name, "", PID, SUBSYSTEM))
        self.windows[label] = (stages[1][1], stages[2][2])
        if arm == "neuralEngine":
            self.ane.append((t + MS, 5 * MS, PREDICTION))
            self.coreai.append((t + MS // 2, 8 * MS, PID, "Function Execute"))
        elif arm == "gpu":
            self.gpu.append((t + MS, 6 * MS, PID))
        else:
            self.ane.append((t + MS, MS, "Neural Engine Compile"))
        self.gpu.append((t - MS, 15 * MS, OTHER_PID))

    def window(self, arm: str, index: int) -> tuple[int, int]:
        """Return the call window of an arm's ``index``-th call."""
        label = [s["label"] for s in self.samples if s["arm"] == arm][index]
        return self.windows[label]

    def write(self, root: Path) -> Path:
        """Write the run folder under ``root`` and return it."""
        root.mkdir(parents=True)
        folder = "DEVICE-FOLDER"
        device = root / "device" / folder
        device.mkdir(parents=True)
        report = {
            "version": "synthetic",
            "state": "complete",
            "pid": PID,
            "revision": "synthetic-01",
            "configuration": "Release",
            "chunks": [{"chunk": c, "file": f"chunk-{c:02d}.aimodel"} for c in range(self.chunks)],
            "samples": self.samples,
        }
        (device / "report.json").write_text(json.dumps(report))
        exports = root / "exports"
        exports.mkdir()
        self.write_recording(exports)
        manifests = {}
        for name in ("recording", "idle-1", "idle-2"):
            bundle = root / f"{name}.trace"
            (bundle / "corespace").mkdir(parents=True)
            (bundle / "corespace" / "data").write_bytes(name.encode() * 64)
            (bundle / "Trace1.run").write_text(name)
            manifests[name] = bundle_manifest(bundle)
        for name, (seconds, rows) in self.idle.items():
            tables = ('<table schema="ane-hw-intervals"/>',)
            (exports / f"{name}-toc.xml").write_text(toc(self.device, self.target, seconds, tables))
            (exports / f"{name}-ane-hw-intervals.xml").write_text(self.ane_export(1, rows))
        result = {
            "device": DEVICE,
            "folder": folder,
            "pid": PID,
            "coreAITable": "CoreAIProfile",
            "traceManifests": manifests,
            "complete": True,
        }
        (root / "result.json").write_text(json.dumps(result))
        return root

    def write_recording(self, exports: Path) -> None:
        """Write the recording's TOC and its four table exports."""
        span = max(t for t, *_ in self.signposts) / 1e9 + 1
        text = toc(self.device, self.target, round(span, 6), TOC_TABLES)
        (exports / "recording-toc.xml").write_text(text)
        values = Values()
        rows = []
        for t, kind, name, message, pid, subsystem in sorted(self.signposts):
            owner = process(values, pid)
            thread = values("thread", "", f"host (0x1) (host, pid: {pid})", owner)
            text = values("string", message)
            rows.append(
                values("event-time", str(t))
                + thread
                + process(values, pid)
                + values("event-type", kind)
                + values("os-signpost-identifier", "17216892719917625070", "EXCLUSIVE")
                + values("signpost-name", name)
                + "<sentinel/>"
                + values("subsystem", subsystem)
                + values("os-log-metadata", "", message, text)
            )
        columns = ["time", "thread", "process", "event-type", "identifier", "name"]
        columns += ["backtrace", "subsystem", "message"]
        signposts = export(TABLE_INDEX["signposts"], "os-signpost", columns, rows)
        (exports / "recording-signposts.xml").write_text(signposts)
        ane = self.ane_export(TABLE_INDEX["ane-hw-intervals"], self.ane)
        (exports / "recording-ane-hw-intervals.xml").write_text(ane)
        values = Values()
        rows = []
        for start, duration, pid in sorted(self.gpu):
            owner = process(values, pid)
            label = values("formatted-label", "", f" ({pid}) encoder", owner)
            rows.append(
                values("start-time", str(start))
                + values("duration", str(duration))
                + values("gpu-channel-name", "Compute")
                + label
                + process(values, pid)
            )
        columns = ["start", "duration", "channel-name", "event-label", "process"]
        gpu = export(TABLE_INDEX["metal-gpu-intervals"], "metal-gpu-intervals", columns, rows)
        (exports / "recording-metal-gpu-intervals.xml").write_text(gpu)
        values = Values()
        rows = [
            values("start-time", str(start))
            + values("duration", str(duration))
            + process(values, pid)
            + values("event-concept", event)
            for start, duration, pid, event in sorted(self.coreai)
        ]
        columns = ["start", "duration", "process", "event-id"]
        coreai = export(TABLE_INDEX["coreai"], "CoreAIProfile", columns, rows)
        (exports / "recording-coreai.xml").write_text(coreai)

    @staticmethod
    def ane_export(index: int, events: list[tuple[int, int, str]]) -> str:
        """An ``ane-hw-intervals`` export; labels nest their parts as ``xctrace`` does."""
        values = Values()
        rows = []
        for start, duration, label in sorted(events):
            tail = label.removeprefix("Neural Engine")
            parts = values("string", "Neural Engine") + values("narrative-text", tail)
            rows.append(
                values("start-time", str(start))
                + values("duration", str(duration))
                + values("ane-event-name", "Apple Neural Engine")
                + values("formatted-label", "", label, parts)
                + values("gpu-state", "Active")
            )
        columns = ["start", "duration", "channel-name", "event-label", "state"]
        return export(index, "ane-hw-intervals", columns, rows)


def numerics(run: Path, chunks: int, **changes: object) -> Path:
    """Write a passing reference report for every chunk; ``changes`` override head values.

    The last chunk stands for the head: it also gets the whole-vocabulary distribution.
    """
    rows: list[dict[str, object]] = []
    for chunk in range(chunks):
        for output in ("hidden", "fresh_key_0"):
            score = {"psnrDb": 60.0, "cosine": 0.99999}
            rows.append({"chunk": chunk, "output": output, "scores": {"neuralEngine": score}})
    head = {"top1Equal": True, "top1": 3, "klNats": 1e-5, "top10Overlap": 10} | changes
    distribution = {"neuralEngine": head}
    rows.append({"chunk": chunks - 1, "output": "logits (all 64)", "distribution": distribution})
    path = run / "numerics.json"
    path.write_text(json.dumps({"run": str(run / "device" / "DEVICE-FOLDER"), "rows": rows}))
    return path
