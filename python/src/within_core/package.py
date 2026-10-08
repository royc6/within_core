"""Assemble the downloadable model package and its two manifests.

Pipeline position: the last step, after the chunk export (:mod:`within_core.export`) and the
tables (:mod:`within_core.tables`, or :mod:`within_core.qat` for a lite package). The
package folder holds the chunk assets, ``chunk-contract.json``, ``gemma-runtime.json``, both
tables, the checkpoint's tokenizer and generation configuration (unchanged), and the Gemma
license and notice. Large files are hard-linked where the file system allows it. A lite
package's ``gemma-runtime.json`` is written here as schema 2, with the 2-bit tables'
descriptors in place of the FP16 ones.

The app bundles either one package's manifest or, since release 1.1, a catalog of one
package per tier (:func:`catalog`).

The app installs a package only if the ``installed-manifest.json`` inside it equals the
catalog bundled with the app (``ModelCatalog.json``) and every listed file has the listed
size and SHA-256. Both manifests are therefore written from one text, and :func:`validate`
applies the app's manifest rules so that a package the app would refuse fails here.
"""

import hashlib
import json
import os
import shutil
import unicodedata
from pathlib import Path

from within_core.checkpoint import LocalCheckpoint
from within_core.export import CHUNK_CONTRACT, RUNTIME_CONFIG
from within_core.qat import TABLES_RECORD, QATCheckpoint

SCHEMA_VERSION = 1
CATALOG_VERSION = 2
LITE_RUNTIME_SCHEMA = 2
TIERS = ("full", "balance", "liteLong", "liteFast")
MODEL_ID = "gemma-4-e4b"
BACKEND = "coreAIANE"
LICENSE_URL = "https://ai.google.dev/gemma/docs/gemma_4_license"
MIN_CONTEXT = 512
MAX_FILE_BYTES = 30_000_000_000
MAX_TOTAL_BYTES = 40_000_000_000
INSTALLED_MANIFEST = "installed-manifest.json"
CATALOG = "ModelCatalog.json"
# Written by the app on the device; a manifest never lists them.
RESERVED = frozenset({INSTALLED_MANIFEST, "download-state.json"})
LEGAL = Path(__file__).with_name("legal")
LEGAL_FILES = ("LICENSE", "NOTICE")
GENERATION_CONFIG = "generation_config.json"
TOKENIZER = "tokenizer"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json")


class PackageError(ValueError):
    """A package or manifest breaks a named rule of the app's manifest validation."""


def require(condition: bool, rule: str) -> None:
    """Raise :class:`PackageError` naming ``rule`` unless ``condition`` holds."""
    if not condition:
        raise PackageError(rule)


def sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def files_for(package: Path) -> list[dict[str, object]]:
    """List every regular file of ``package`` as ``{path, size, sha256}``, sorted by path.

    Raises:
        PackageError: If the package holds a symbolic link or a special file.
    """
    require(package.is_dir(), f"package directory does not exist: {package}")
    rows = []
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package).as_posix()
        require(not path.is_symlink(), f"symlink in package: {relative}")
        if path.is_dir() or relative in RESERVED:
            continue
        require(path.is_file(), f"not a regular file: {relative}")
        rows.append({"path": relative, "size": path.stat().st_size, "sha256": sha256(path)})
    return rows


def manifest(package: Path, revision: str, context_limit: int, base_url: str) -> dict[str, object]:
    """Return the manifest of ``package`` (see the module docstring)."""
    return {
        "schemaVersion": SCHEMA_VERSION,
        "modelID": MODEL_ID,
        "revision": revision,
        "backend": BACKEND,
        "baseURL": base_url,
        "contextLimit": context_limit,
        "files": files_for(package),
        "licenseURL": LICENSE_URL,
    }


def _check_url(value: object, field: str, *, plain: bool = True) -> None:
    """Require an https URL with a host and no credentials (and, if ``plain``, no query)."""
    require(isinstance(value, str) and value.startswith("https://"), f"{field} must be https")
    host = value.removeprefix("https://").split("/", 1)[0]
    require(bool(host), f"{field} has no host")
    require("@" not in host, f"{field} must not carry user or password")
    if plain:
        require("?" not in value, f"{field} must not carry a query")
        require("#" not in value, f"{field} must not carry a fragment")


def validate(data: dict) -> dict:
    """Apply the app's manifest rules to ``data`` and return it.

    Raises:
        PackageError: Naming the first rule that fails.
    """
    require(data.get("schemaVersion") == SCHEMA_VERSION, f"schemaVersion must be {SCHEMA_VERSION}")
    require(data.get("modelID") == MODEL_ID, f"modelID must be {MODEL_ID}")
    context = data.get("contextLimit")
    require(
        isinstance(context, int) and context >= MIN_CONTEXT,
        f"contextLimit must be >= {MIN_CONTEXT}",
    )
    revision = data.get("revision")
    require(
        isinstance(revision, str) and 0 < len(revision) <= 80,
        "revision must have 1 to 80 characters",
    )
    require(
        all(c.isascii() and (c.isalnum() or c == "-") for c in revision),
        "revision may contain only ASCII letters, digits and hyphens",
    )
    require(data.get("backend") == BACKEND, f"backend must be {BACKEND}")
    _check_url(data.get("baseURL"), "baseURL")
    _check_url(data.get("licenseURL"), "licenseURL", plain=False)
    files = data.get("files")
    require(isinstance(files, list) and bool(files), "files must be a non-empty list")
    seen: set[str] = set()
    total = 0
    for row in files:
        path = row.get("path")
        require(isinstance(path, str) and bool(path), "file path must be a non-empty string")
        require(
            all(c and c not in (".", "..") for c in path.split("/")),
            f"file path has an empty or relative component: {path}",
        )
        for bad in ("\\", "%", ":"):
            require(bad not in path, f"file path contains {bad!r}: {path}")
        require(
            not any(ord(c) < 32 or ord(c) == 127 for c in path),
            f"file path contains a control character: {path}",
        )
        size = row.get("size")
        require(isinstance(size, int) and size > 0, f"file size must be positive: {path}")
        require(size < MAX_FILE_BYTES, f"file exceeds the per-file cap: {path}")
        digest = row.get("sha256")
        require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(c in "0123456789abcdefABCDEF" for c in digest),
            f"sha256 must be 64 hex characters: {path}",
        )
        key = unicodedata.normalize("NFC", path).lower()
        require(key not in seen, f"duplicate file path: {path}")
        seen.add(key)
        total += size
        require(total < MAX_TOTAL_BYTES, "package exceeds the total size cap")
    require(seen.isdisjoint(RESERVED), f"manifest lists a reserved name: {sorted(RESERVED)}")
    for key in seen:
        parts = key.split("/")
        for count in range(1, len(parts)):
            require(
                "/".join(parts[:count]) not in seen,
                f"a file is also a parent directory of another: {key}",
            )
    return data


def link_or_copy(source: Path, destination: Path) -> None:
    """Hard-link ``source`` (a file or a folder tree) to a new ``destination``.

    Falls back to copying where hard links are impossible (another volume).

    Raises:
        PackageError: If the tree holds a symbolic link.
    """
    require(not source.is_symlink(), f"symlink in package source: {source}")
    if source.is_dir():
        destination.mkdir()
        for path in sorted(source.iterdir()):
            link_or_copy(path, destination / path.name)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def assemble(
    export: Path,
    tables: Path,
    checkpoint: LocalCheckpoint | QATCheckpoint,
    output: Path,
    revision: str,
    base_url: str,
) -> dict:
    """Assemble ``<output>/package`` and write both manifests.

    Args:
        export: Finalized export folder (chunk assets, contract, runtime configuration).
        tables: Folder with the FP16 tables, or the 2-bit tables and their ``tables.json``.
        checkpoint: The verified checkpoint (tokenizer and generation configuration).
        output: New folder for ``package/`` and ``ModelCatalog.json``.
        revision: Package revision, e.g. the Hugging Face tag it is published under.
        base_url: URL that package paths are resolved against for download.

    Returns:
        The manifest.

    Raises:
        FileExistsError: If ``output`` exists.
        PackageError: If the manifest breaks a rule.
    """
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    contract = json.loads((export / CHUNK_CONTRACT).read_text())
    runtime = json.loads((export / RUNTIME_CONFIG).read_text())
    package = output / "package"
    package.mkdir(parents=True)
    for entry in contract["chunks"]:
        link_or_copy(export / entry["file"], package / entry["file"])
    link_or_copy(export / CHUNK_CONTRACT, package / CHUNK_CONTRACT)
    record = tables / TABLES_RECORD
    if record.is_file():
        descriptors = json.loads(record.read_text())
        runtime.update(schemaVersion=LITE_RUNTIME_SCHEMA, **descriptors)
        (package / RUNTIME_CONFIG).write_text(json.dumps(runtime, indent=2) + "\n")
        for descriptor in descriptors.values():
            for name in (descriptor["file"], descriptor["scales"]["file"]):
                link_or_copy(tables / name, package / name)
    else:
        link_or_copy(export / RUNTIME_CONFIG, package / RUNTIME_CONFIG)
        for key in ("embeddings", "perLayerEmbeddings"):
            name = runtime[key]["file"]
            link_or_copy(tables / name, package / name)
    link_or_copy(checkpoint.path(GENERATION_CONFIG), package / GENERATION_CONFIG)
    (package / TOKENIZER).mkdir()
    for name in TOKENIZER_FILES:
        link_or_copy(checkpoint.path(name), package / TOKENIZER / name)
    # Copies, not links: the license files belong to the package, not to this source tree.
    for name in LEGAL_FILES:
        shutil.copyfile(LEGAL / name, package / name)
    data = validate(manifest(package, revision, runtime["contextLimit"], base_url))
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    (package / INSTALLED_MANIFEST).write_text(text)
    (output / CATALOG).write_text(text)
    return data


def catalog(entries: list[tuple[str, Path]], default_tier: str, output: Path | None = None) -> dict:
    """Return a ``catalogVersion`` 2 ``ModelCatalog.json`` listing one package per tier.

    The app reads this catalog since release 1.1: each entry is a package's own
    ``installed-manifest.json`` under its tier, and the default tier is the one a fresh
    installation downloads. The rules mirror the app's: one to four packages, distinct tiers
    and installation IDs, the default among them, and one size per SHA-256 across packages.

    Args:
        entries: ``(tier, package folder)`` pairs; tiers are :data:`TIERS`.
        default_tier: The tier a fresh installation downloads.
        output: Where to write the catalog, if anywhere.

    Raises:
        PackageError: Naming the first rule that fails.
    """
    require(1 <= len(entries) <= 4, "a catalog holds 1 to 4 packages")
    rows = []
    for tier, package in entries:
        require(tier in TIERS, f"unknown tier: {tier}")
        installed = json.loads((Path(package) / INSTALLED_MANIFEST).read_text())
        rows.append({"tier": tier, "manifest": validate(installed)})
    require(len({row["tier"] for row in rows}) == len(rows), "tiers repeat")
    ids = {f"{row['manifest']['modelID']}-{row['manifest']['revision']}" for row in rows}
    require(len(ids) == len(rows), "installation IDs repeat")
    require(any(row["tier"] == default_tier for row in rows), "defaultTier is not in the catalog")
    sizes: dict[str, int] = {}
    for row in rows:
        for file in row["manifest"]["files"]:
            require(
                sizes.setdefault(file["sha256"], file["size"]) == file["size"],
                f"one SHA-256 with two sizes: {file['path']}",
            )
    data = {"catalogVersion": CATALOG_VERSION, "defaultTier": default_tier, "packages": rows}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return data
