"""Model files that run in visitors' browsers (the free in-browser subtitles), served from this computer at
`/api/downloads/browser/<model>/<file>` to signed-in browsers only (acw/api/downloads_routes.py).

Each model id carries the version of its files, so the browser may keep them for good (it does, in its own cache):
a new pin gets a new id. The files are fetched and checked by deploy/fetch-models.sh (deploy/models.lock lists the
same files with the same sha256 and size; tests/test_browser_models.py keeps the two in step).
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

from .config import get_settings

# model id -> (folder under the models directory, {file: (sha256, size)})
MODELS: dict[str, tuple[str, dict[str, tuple[str, int]]]] = {
    # Whisper base (OpenAI, MIT), ONNX export by onnx-community/whisper-base at 1846881b, 8-bit weights: about 80 MB.
    "whisper-base-1846881": ("browser/whisper-base", {
        "config.json": ("f4d0608f7d918166da7edb3e188de5ef1bfe70d9802e785d271fd88111e9cf4b", 2243),
        "generation_config.json": ("61070cf8de25b1e9256e8e102ded49d8d24a8369ed36ef84fdf21549e68125a0", 3832),
        "preprocessor_config.json": ("a6a76d28c93edb273669eb9e0b0636a2bddbb1272c3261e47b7ca6dfdbac1b8d", 339),
        "tokenizer.json": ("27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566", 2480466),
        "tokenizer_config.json": ("2e036e4dbacfdeb7242c7d4ec4149f4a16e86026048f94d1637e3a8ee9c6a573", 282682),
        "onnx/encoder_model_quantized.onnx": ("5862993336bf33acd23736071aae2b32261d3b1b2f37780194460d4ef974dd46", 23201314),
        "onnx/decoder_model_merged_quantized.onnx": ("fa3ef9902734ce5ae6f9ef2bdb2ba9a6c4b5785b09f4f420ce036573dc9d090b", 53693315),
    }),
}


def lookup(model: str, name: str) -> dict | None:
    """The entry for a file of a model (only the files listed above, never anything else in the folder)."""
    m = MODELS.get(model)
    if m is None or name not in m[1]:
        return None
    sha256, size = m[1][name]
    return {"name": name.rsplit("/", 1)[-1], "size": size, "sha256": sha256, "manifest": False,
            "path": get_settings().models_dir / m[0] / name}


def open_file(entry: dict) -> int:
    """Descriptor of the file, opened without following a link at the end, and only if it is a regular file of the
    recorded size (a missing or half-fetched model is a 404, never a wrong file)."""
    fd = os.open(entry["path"], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size != entry["size"]:
            raise OSError("not the pinned file")
    except BaseException:
        os.close(fd)
        raise
    return fd


def installed(model: str) -> bool:
    m = MODELS.get(model)
    if m is None:
        return False
    base: Path = get_settings().models_dir / m[0]
    return all((base / n).is_file() and (base / n).stat().st_size == s for n, (_, s) in m[1].items())
