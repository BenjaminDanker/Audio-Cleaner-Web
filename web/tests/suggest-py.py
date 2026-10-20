"""Makes the test files of tests/suggest-agreement.mjs and runs the server's suggestions (common/acw_common/suggest.py) on them.
    python tests/suggest-py.py OUT_DIR   ->  OUT_DIR/<name>.wav and OUT_DIR/<name>.py.json for each file
The browser's version (src/media/suggest-core.js) is run on the same files by the Node test and compared with these."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

HOME = Path(os.environ.get("AC_HOME", Path.home() / "audio-cleaner"))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
from acw_common import dfn, suggest  # noqa: E402

SR = 48000
SPEECH = Path(os.environ.get("ACW_SPEECH_WAV", HOME / "dev" / "live_opus.wav"))
LIB = Path(os.environ.get("ACW_LIBDF", HOME / "bin" / dfn.LIB_NAME))
MODEL = Path(os.environ.get("ACW_DFN_MODEL", HOME / "models" / dfn.MODEL_FILE))


def speech(seconds):
    s, sr = sf.read(SPEECH, dtype="float32", always_2d=True)
    assert sr == SR
    return np.tile(s[:, 0], int(np.ceil(seconds * SR / len(s))))[: int(seconds * SR)].copy()


def active(x):
    return float(np.sqrt(np.mean(x[np.abs(x) > 0.02] ** 2)))


def known_parts():
    """120 s: 0-40 s clean, 40-80 s in loud low-pitched noise, 80-120 s as heard on a phone."""
    rng = np.random.default_rng(1)
    a = speech(40); a *= 0.1 / active(a)
    b = speech(40); b *= 0.1 / active(b)
    n = signal.lfilter([1], [1, -0.9], rng.standard_normal(len(b))).astype(np.float32)
    b += n / np.sqrt(np.mean(n ** 2)) * active(b) * 10 ** (-2 / 20)
    c = speech(40)
    c = signal.sosfilt(signal.butter(6, [300, 3300], btype="band", fs=SR, output="sos"), c).astype(np.float32)
    c *= 0.1 / active(c)
    return np.concatenate([a, b, c]).astype(np.float32)


def clipped_bursts():
    x = speech(60)
    x *= 0.6 / np.abs(x).max()
    for a, b in [(12.0, 12.6), (30.0, 30.4), (47.0, 47.5)]:
        i, j = int(a * SR), int(b * SR)
        x[i:j] = np.clip(x[i:j] * 8, -0.6, 0.6)
    return x


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    k = known_parts()
    files = {
        "parts-mono": k[:, None],
        "parts-stereo": np.stack([k, k * 0.8 + np.roll(k, 7) * 0.1], axis=1).astype(np.float32),
        "clipping": clipped_bursts()[:, None],
        "short": speech(30)[:, None] * 0.3,
        "noisy-then-clean": np.concatenate([k[40 * SR:80 * SR], k[:40 * SR], k[:30 * SR]])[:, None],
    }
    for name, x in files.items():
        sf.write(out / f"{name}.wav", np.clip(x, -1, 1), SR, subtype="PCM_16")
        r = suggest.suggest(x.astype(np.float32), LIB, MODEL, threads=4)
        (out / f"{name}.py.json").write_text(json.dumps({"seconds": len(x) / SR, **r.to_json(), "cpu": r.cpu_seconds}))
        print(name, len(x) / SR, "s,", len(r.suggestions), "suggestions,", round(r.cpu_seconds, 1), "cpu s")


if __name__ == "__main__":
    main(sys.argv[1])
