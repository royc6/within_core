"""Manifest rules that the app applies before installing a package."""

import copy
import os
from pathlib import Path

import pytest

from within_core.package import (
    INSTALLED_MANIFEST,
    PackageError,
    files_for,
    link_or_copy,
    validate,
)

DIGEST = "0" * 64
GOOD = {
    "schemaVersion": 1,
    "modelID": "gemma-4-e4b",
    "revision": "lut6-c16384-01",
    "backend": "coreAIANE",
    "baseURL": "https://example.com/models/",
    "contextLimit": 16384,
    "licenseURL": "https://ai.google.dev/gemma/docs/gemma_4_license",
    "files": [
        {"path": "chunk-00.aimodel/main.mlirb", "size": 10, "sha256": DIGEST},
        {"path": "embeddings.f16", "size": 20, "sha256": DIGEST},
    ],
}


def changed(**fields: object) -> dict:
    """A copy of the valid manifest with ``fields`` replaced."""
    return copy.deepcopy(GOOD) | fields


def with_file(path: str, size: int = 1, sha256: str = DIGEST) -> dict:
    """A copy of the valid manifest with one more file."""
    data = copy.deepcopy(GOOD)
    data["files"].append({"path": path, "size": size, "sha256": sha256})
    return data


def test_valid_manifest_passes() -> None:
    assert validate(copy.deepcopy(GOOD)) == GOOD


@pytest.mark.parametrize(
    ("data", "rule"),
    [
        (changed(revision="lut6_c16384"), "revision may contain only"),
        (changed(revision="x" * 81), "1 to 80"),
        (changed(contextLimit=256), "contextLimit"),
        (changed(backend="coreMLGPU"), "backend"),
        (changed(baseURL="http://example.com/"), "baseURL must be https"),
        (changed(baseURL="https://example.com/?tag=1"), "query"),
        (changed(baseURL="https://user@example.com/"), "user or password"),
        (changed(files=[]), "non-empty list"),
        (with_file("installed-manifest.json"), "reserved"),
        (with_file("chunk-00.aimodel"), "parent directory"),
        (with_file("Embeddings.F16"), "duplicate"),
        (with_file("tokenizer/../config.json"), "relative component"),
        (with_file("tokenizer\\config.json"), "contains"),
        (with_file("empty", size=0), "size must be positive"),
        (with_file("short-digest", sha256="abc"), "64 hex"),
    ],
)
def test_manifest_rules_name_the_failure(data: dict, rule: str) -> None:
    with pytest.raises(PackageError, match=rule):
        validate(data)


def test_files_are_sorted_and_skip_the_installed_manifest(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    for name in ("b/z", "a", INSTALLED_MANIFEST):
        (tmp_path / name).write_text(name)
    assert [row["path"] for row in files_for(tmp_path)] == ["a", "b/z"]
    assert files_for(tmp_path)[0]["size"] == 1


def test_symbolic_links_are_refused(tmp_path: Path) -> None:
    (tmp_path / "a").write_text("a")
    (tmp_path / "link").symlink_to(tmp_path / "a")
    with pytest.raises(PackageError, match="symlink"):
        files_for(tmp_path)


def test_trees_are_hard_linked(tmp_path: Path) -> None:
    source = tmp_path / "source.aimodel"
    source.mkdir()
    (source / "main.mlirb").write_bytes(b"graph")
    link_or_copy(source, tmp_path / "linked.aimodel")
    linked = tmp_path / "linked.aimodel" / "main.mlirb"
    assert os.path.samefile(linked, source / "main.mlirb")
