"""Verified, offline reads from a local Gemma 4 E4B checkpoint.

Pipeline position: every conversion step reads its inputs through this module. The
checkpoint folder must hold the files pinned in ``reference-config.json``
(``google/gemma-4-e4b-it`` at a fixed revision), byte for byte; each file's size and SHA-256
are checked before its first use, and nothing is ever downloaded. Reads are bounded: a
tensor, a row range, or one decoder layer at a time, so converting never holds the whole
model in memory.
"""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

REFERENCE = Path(__file__).with_name("reference-config.json")
CONFIG = "config.json"
WEIGHTS = "model.safetensors"
TENSOR_PREFIX = "model.language_model."
# Transformers ignores these tensors on layers that reuse a donor's KV.
UNUSED_SHARED_TENSORS = ("k_proj.weight", "v_proj.weight", "k_norm.weight", "v_norm.weight")


def load_reference() -> dict:
    """Return the pinned checkpoint description (source, revision, files, text config)."""
    return json.loads(REFERENCE.read_text())


def pinned_config() -> Gemma4TextConfig:
    """Return the pinned text configuration, without reading a checkpoint."""
    return Gemma4TextConfig(**load_reference()["text_config"])


def sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class LocalCheckpoint:
    """A local checkpoint folder whose files equal a pinned reference.

    Attributes:
        folder: The checkpoint folder.
        reference: The pinned description (``source``, ``revision``, ``files`` with the
            size and SHA-256 of each file, and ``text_config``).
        config: The text model configuration.
    """

    def __init__(self, folder: Path, reference: dict | None = None) -> None:
        """Open a checkpoint folder and verify its configuration file.

        Args:
            folder: Folder with the checkpoint's files.
            reference: Pinned description; defaults to the packaged ``reference-config.json``.

        Raises:
            ValueError: If the configuration differs from the pinned one.
        """
        self.folder = Path(folder)
        self.reference = load_reference() if reference is None else reference
        self._verified: set[str] = set()
        config = json.loads(self.path(CONFIG).read_text())["text_config"]
        if config != self.reference["text_config"]:
            raise ValueError("The text configuration differs from the pinned one")
        self.config = Gemma4TextConfig(**config)

    def path(self, name: str) -> Path:
        """Return the path of a pinned file, verifying its size and SHA-256 on first use.

        Raises:
            ValueError: If ``name`` is not pinned, or the file is missing or differs.
        """
        pinned = self.reference["files"].get(name)
        if pinned is None:
            raise ValueError(f"{name} is not a pinned checkpoint file")
        path = self.folder / name
        if name not in self._verified:
            if (
                not path.is_file()
                or path.stat().st_size != pinned["bytes"]
                or sha256(path) != pinned["sha256"]
            ):
                source = f"{self.reference['source']}@{self.reference['revision']}"
                raise ValueError(f"{path} is not the pinned file of {source}")
            self._verified.add(name)
        return path

    def tensor(self, name: str, dtype: torch.dtype = torch.float16) -> Tensor:
        """Return a text-model tensor, e.g. ``"norm.weight"``, cast to ``dtype``."""
        with safe_open(self.path(WEIGHTS), framework="pt") as tensors:
            return tensors.get_tensor(TENSOR_PREFIX + name).to(dtype)

    def row_range(
        self, name: str, start: int, end: int, dtype: torch.dtype = torch.float16
    ) -> Tensor:
        """Return rows ``[start, end)`` of a text-model tensor, cast to ``dtype``."""
        with safe_open(self.path(WEIGHTS), framework="pt") as tensors:
            return tensors.get_slice(TENSOR_PREFIX + name)[start:end].to(dtype)

    def layer(self, index: int, dtype: torch.dtype = torch.float16) -> Gemma4TextDecoderLayer:
        """Return decoder layer ``index`` as a Transformers module with ``dtype`` weights.

        Raises:
            ValueError: If the index is outside the decoder.
        """
        if not 0 <= index < self.config.num_hidden_layers:
            raise ValueError(f"Layer {index} is outside the decoder")
        with torch.device("meta"):
            layer = Gemma4TextDecoderLayer(self.config, index)
        prefix = f"{TENSOR_PREFIX}layers.{index}."
        with safe_open(self.path(WEIGHTS), framework="pt") as tensors:
            # A safe_open handle is not a mapping; keys() is how it lists its tensors.
            names = [name for name in tensors.keys() if name.startswith(prefix)]  # noqa: SIM118
            state = {n.removeprefix(prefix): tensors.get_tensor(n).to(dtype) for n in names}
        if layer.self_attn.is_kv_shared_layer:
            for name in UNUSED_SHARED_TENSORS:
                state.pop(f"self_attn.{name}", None)
        layer.load_state_dict(state, strict=True, assign=True)
        return layer.eval()
