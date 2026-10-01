"""The whole conversion on a tiny checkpoint: chunks, contract, tables and package."""

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from tiny import build_chunks, write_checkpoint
from within_core import export, package, tables
from within_core.checkpoint import LocalCheckpoint
from within_core.convert import parser

BASE_URL = "https://example.com/models/tiny/"
# The smallest context a package manifest accepts.
CONTEXT = 512


class Converted(NamedTuple):
    """Work folder of a tiny conversion and the configuration it used."""

    root: Path
    config: Gemma4TextConfig


@pytest.fixture(scope="module")
def converted(tmp_path_factory: pytest.TempPathFactory, reference: Gemma4ForCausalLM) -> Converted:
    """Run every step once on a tiny checkpoint."""
    root = tmp_path_factory.mktemp("convert")
    pinned = write_checkpoint(root / "checkpoint", reference)
    checkpoint = LocalCheckpoint(root / "checkpoint", pinned)
    for index in range(len(export.plan(checkpoint.config, CONTEXT))):
        export.export_chunk(checkpoint, index, root / "export", CONTEXT)
    export.finalize(root / "export", checkpoint.config, CONTEXT)
    tables.write_tables(checkpoint, root / "tables")
    package.assemble(
        root / "export", root / "tables", checkpoint, root / "release", "tiny-01", BASE_URL
    )
    return Converted(root, checkpoint.config)


def unused_constants(graph: object) -> list[object]:
    """Top-level constants of an authored graph that no operation uses."""
    return [
        op
        for op in graph.regions[0].blocks[0].operations
        if op.operation.name == "coreai.constant" and not any(op.results[0].uses)
    ]


def test_every_chunk_is_exported_with_its_contract(converted: Converted) -> None:
    contract = json.loads((converted.root / "export" / export.CHUNK_CONTRACT).read_text())
    assert contract["contract"] == "gemma4-chunked-hostkv-v1"
    assert (contract["contextLimit"], contract["slidingCacheLength"]) == (CONTEXT, 5)
    assert contract["firstSharedLayer"] == 24
    files = [entry["file"] for entry in contract["chunks"]]
    assert files == [f"chunk-{i:02d}.aimodel" for i in range(8)]
    for name in files:
        assert (converted.root / "export" / name / "main.mlirb").is_file()
    functions = [list(entry["functions"]) for entry in contract["chunks"]]
    assert functions == [["step_1", "prefill_16"]] * 4 + [["step_1"]] * 4


def test_export_never_replaces_a_chunk(converted: Converted) -> None:
    checkpoint = LocalCheckpoint.__new__(LocalCheckpoint)
    checkpoint.config = converted.config
    with pytest.raises(FileExistsError):
        export.export_chunk(checkpoint, 7, converted.root / "export", CONTEXT)


def test_finalize_rejects_a_chunk_from_another_plan(converted: Converted, tmp_path: Path) -> None:
    for record in (converted.root / "export").glob("chunk-*.json"):
        (tmp_path / record.name).write_text(record.read_text())
    with pytest.raises(ValueError, match="different plan"):
        export.finalize(tmp_path, converted.config, CONTEXT + 1)
    (tmp_path / export.record_file(7)).unlink()
    with pytest.raises(FileNotFoundError):
        export.finalize(tmp_path, converted.config, CONTEXT)


def test_authored_prefill_holds_no_unused_weights(reference: Gemma4ForCausalLM) -> None:
    """The donor chunk's headless prefill drops its last layer's tail, and its weights."""
    chunk = build_chunks(reference, export.BOUNDARIES, 4)[3].half()
    program = export.author(chunk)
    names = ["step_1", "prefill_16"]
    assert all(not unused_constants(program.get_graph(name)) for name in names)
    before = str(program)
    assert export.prune_unused_constants(program, names) == dict.fromkeys(names, 0)
    with pytest.raises(ValueError, match="Missing"):
        export.prune_unused_constants(program, ["prefill_16", "missing"])
    assert str(program) == before


def test_tables_hold_the_checkpoint_embeddings_in_fp16(
    converted: Converted, reference: Gemma4ForCausalLM
) -> None:
    model = reference.model
    for name, source in (
        (tables.EMBEDDINGS, model.embed_tokens.weight),
        (tables.PER_LAYER_EMBEDDINGS, model.embed_tokens_per_layer.weight),
    ):
        table = np.fromfile(converted.root / "tables" / name, dtype="<f2")
        expected = source.detach().bfloat16().half().numpy().reshape(-1)
        assert np.array_equal(table.view(np.uint16), expected.view(np.uint16))


def test_package_manifest_equals_catalog_and_lists_every_file(converted: Converted) -> None:
    release = converted.root / "release"
    installed = (release / "package" / package.INSTALLED_MANIFEST).read_text()
    assert installed == (release / package.CATALOG).read_text()
    manifest = json.loads(installed)
    assert (manifest["revision"], manifest["baseURL"]) == ("tiny-01", BASE_URL)
    assert manifest["contextLimit"] == CONTEXT
    paths = [row["path"] for row in manifest["files"]]
    assert paths == sorted(paths)
    top = {
        "LICENSE",
        "NOTICE",
        "chunk-contract.json",
        "gemma-runtime.json",
        "embeddings.f16",
        "perLayerEmbeddings.f16",
        "generation_config.json",
        "tokenizer",
        *(f"chunk-{i:02d}.aimodel" for i in range(8)),
    }
    assert {path.split("/")[0] for path in paths} == top
    tokenizer = {path for path in paths if path.startswith("tokenizer/")}
    assert tokenizer == {f"tokenizer/{name}" for name in package.TOKENIZER_FILES}
    license_text = (release / "package" / "LICENSE").read_bytes()
    assert license_text == (package.LEGAL / "LICENSE").read_bytes()


def test_command_line_defaults_are_the_published_configuration() -> None:
    arguments = parser().parse_args(["chunk", "3", "--checkpoint", "c", "--output", "o"])
    assert (arguments.index, arguments.context, arguments.threads) == (3, 16384, 4)
    with pytest.raises(SystemExit):
        parser().parse_args(["chunk", "8", "--checkpoint", "c", "--output", "o"])
