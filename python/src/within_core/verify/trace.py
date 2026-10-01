"""Reading the tables that ``xctrace export`` writes for a placement recording.

Pipeline position: the first step of :mod:`within_core.verify.review`. A recording is
exported as one table-of-contents file (``--toc``) and one file per table (``--xpath``
selecting a single table of run 1). Each table export holds one ``node`` whose ``xpath``
names the table's index in the TOC, one ``schema`` listing the columns by mnemonic, and
``row`` elements with one child per column. Repeated values are written once with an
``id`` and referenced later with ``ref``; ``sentinel`` marks an empty cell. Every check
here refuses an export that could have come from another table, run or recording.
"""

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

Row = dict[str, dict[str, str]]
"""One table row: column mnemonic to ``{"raw": element text, "label": formatted text}``."""

# A single-table export names its table by index within run 1 of the TOC.
TABLE_XPATH = re.compile(r"//trace-toc\[1\]/run\[1\]/data\[1\]/table\[(\d+)\]")


class TraceError(ValueError):
    """A recording or export is incomplete, ambiguous or inconsistent."""


def require(condition: bool, message: str) -> None:
    """Raise :class:`TraceError` with ``message`` unless ``condition`` holds."""
    if not condition:
        raise TraceError(message)


def run_node(toc_path: Path) -> ET.Element:
    """Return the only run of a TOC export; a recording must hold exactly run 1."""
    runs = ET.parse(toc_path).getroot().findall("run")
    require(len(runs) == 1 and runs[0].get("number") == "1", f"Expected one trace run: {toc_path}")
    return runs[0]


def recording_target(run: ET.Element, pid: int) -> str:
    """Return ``"allProcesses"`` or ``"singleProcess"`` from a TOC run's target descriptor.

    An all-process recording must say so explicitly (``<all-processes/>``) and list the
    process ``pid`` in its inventory; the absence of a ``<process>`` target is not enough.
    """
    targets = run.findall("info/target")
    require(len(targets) == 1, "Expected one recording target descriptor")
    target = targets[0]
    processes, all_processes = target.findall("process"), target.findall("all-processes")
    if len(processes) == 1 and not all_processes:
        require(int(processes[0].get("pid", "-1")) == pid, "TOC target PID differs from the report")
        return "singleProcess"
    marker = all_processes[0] if len(all_processes) == 1 else None
    require(
        not processes
        and marker is not None
        and not marker.attrib
        and len(marker) == 0
        and not (marker.text or "").strip(),
        "Missing or ambiguous TOC process target; explicit all-processes required",
    )
    matches = [p for p in run.findall("processes/process") if p.get("pid") == str(pid)]
    require(len(matches) == 1, "The report's PID is missing or duplicated in the recording")
    return "allProcesses"


def read_export(
    path: Path, toc_tables: list[ET.Element], schema: str, required: set[str]
) -> list[Row]:
    """Read one exported table, checking it against its entry in the run's TOC.

    Args:
        path: The table export.
        toc_tables: The ``data/table`` elements of the run's TOC, in order.
        schema: Expected schema name.
        required: Column mnemonics that must be present.

    Returns:
        The rows, with every ``ref`` resolved.

    Raises:
        TraceError: If the export is not exactly one table of run 1 with that schema and
            those columns, or a row is truncated or refers to a missing value.
    """
    root = ET.parse(path).getroot()
    nodes = root.findall("node")
    require(
        root.tag == "trace-query-result" and len(nodes) == 1 and len(root) == 1,
        f"Expected one table node, never concatenated exports: {path}",
    )
    node = nodes[0]
    match = TABLE_XPATH.fullmatch(node.get("xpath", ""))
    require(match is not None, f"Export is not from the single run 1: {path}")
    index = int(match[1]) - 1
    require(0 <= index < len(toc_tables), "Export table index absent from the TOC")
    table = toc_tables[index]
    require(table.get("schema") == schema, "Export table differs from the TOC schema")
    if schema == "os-signpost":
        require(
            table.get("category") == "PointsOfInterest"
            and "dynamic-tracing-enabled-subsystems" not in table.attrib,
            "Wrong signpost table selection",
        )
    schemas = node.findall("schema")
    require(len(schemas) == 1 and schemas[0].get("name") == schema, "Missing or duplicate schema")
    columns = [column.findtext("mnemonic") for column in schemas[0].findall("col")]
    require(
        len(set(columns)) == len(columns) and required <= set(columns),
        "Missing or duplicate evidence columns",
    )
    references = {}
    for element in node.iter():
        if element.get("id"):
            require(element.get("id") not in references, "Duplicate XML reference ID")
            references[element.get("id")] = element
    rows = []
    for row in node.findall("row"):
        require(len(row) == len(columns), "Truncated evidence row")
        data = {}
        for name, element in zip(columns, row, strict=True):
            if element.get("ref"):
                require(element.get("ref") in references, "Unresolved XML reference")
                element = references[element.get("ref")]
            text = element.text or ""
            data[name] = {"raw": text, "label": element.get("fmt", text)}
        rows.append(data)
    return rows


def interval(row: Row) -> tuple[int, int]:
    """Return a row's ``[start, stop)`` in nanoseconds from its start and duration."""
    start, duration = int(row["start"]["raw"]), int(row["duration"]["raw"])
    require(start >= 0 and duration >= 0, "Negative trace interval")
    return start, start + duration


def owned(row: Row, pid: int) -> bool:
    """Whether a row's process column names process ``pid`` (labels end in ``(pid)``)."""
    return row["process"]["label"].endswith(f"({pid})")


def bundle_manifest(trace: Path) -> list[dict[str, object]]:
    """List every file of a ``.trace`` bundle with its size and SHA-256, sorted by path."""
    rows = []
    for path in sorted(p for p in trace.rglob("*") if p.is_file()):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        relative = str(path.relative_to(trace))
        rows.append({"file": relative, "bytes": path.stat().st_size, "sha256": digest})
    return rows
