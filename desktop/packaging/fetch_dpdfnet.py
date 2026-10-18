#!/usr/bin/env python3
"""Get the heavy-noise model (DPDFNet-8, Apache-2.0) that the app bundles, from our Hugging Face models repository
(config.hf_models_url, pinned by revision).

    python fetch_dpdfnet.py [DEST_DIR]        # default: build/dpdfnet

One 15 MB file, downloaded with the app's own pinned download (models.fetch_file: resumes, checks the size and the
SHA-256, and only then puts the file in place). A file that is already right is left alone.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "common"), str(HERE)]

from audio_cleaner_desktop import config, models  # noqa: E402
from prepare_vendor import DPDFNET_FILE, DPDFNET_SHA256  # noqa: E402

DPDFNET_SIZE = 14777093
CHUNK = 1 << 20


def url() -> str:
    return config.hf_models_url("dpdfnet/" + DPDFNET_FILE)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def fetch(dest_dir: Path, url_: str | None = None) -> Path:
    target = dest_dir / DPDFNET_FILE
    if target.is_file() and sha256_of(target) == DPDFNET_SHA256:
        return target
    if url_ is None and config.HF_MODELS_REVISION == "PENDING-UPLOAD":
        raise SystemExit("config.HF_MODELS_REVISION is still PENDING-UPLOAD: set the commit id of the models repository")
    try:
        models.fetch_file(url_ or url(), target, DPDFNET_SIZE, DPDFNET_SHA256)
    except models.DownloadError as e:
        raise SystemExit(f"{url_ or url()}: {e}") from None
    return target


def main(argv: list[str]) -> int:
    dest = Path(argv[0]) if argv else Path("build") / "dpdfnet"
    try:
        path = fetch(dest)
    except OSError as e:
        print(f"fetch_dpdfnet.py: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"{path} ({path.stat().st_size} bytes, sha256 ok)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
