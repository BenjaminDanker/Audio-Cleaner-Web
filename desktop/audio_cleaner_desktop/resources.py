"""Locate the pieces the app needs: libDF, the noise model, FFmpeg (shipped) and the speech model
for subtitles (downloaded once, see models.py).

Frozen (PyInstaller) builds find the shipped pieces under sys._MEIPASS:

    native/{df.dll|libdf.dylib|libdf.so, ffmpeg[.exe], ffprobe[.exe]}
    models/DeepFilterNet3_onnx.tar.gz
    licenses/

From source the same layout is read from $ACW_DESKTOP_VENDOR (default
desktop/vendor, filled by the packaging scripts). On Linux, missing pieces fall
back to the server layout in ~/audio-cleaner/{bin,models}.

The models for subtitles, restoring highs and rebuilding the voice are not in the app (2.6 GB): the installer
downloads them during setup, and Settings offers the download again when that failed. Runs from source (tests, CI) may use the small test model that
prepare_vendor.py puts in vendor/test-models/ instead.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import models

_EXE = ".exe" if os.name == "nt" else ""
_FFMPEG = f"ffmpeg{_EXE}"
_FFPROBE = f"ffprobe{_EXE}"
_WHISPER_FILES = ("model.bin", "config.json", "tokenizer.json")

_TURBO = "https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo/resolve/0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf/"
SPEECH_MODEL = models.SpeechModel(
    id="whisper-large-v3-turbo",
    title="Speech model for subtitles",
    description="Whisper large-v3-turbo: about 100 languages, runs on this computer.",
    licence="OpenAI Whisper large-v3-turbo (MIT), CTranslate2 conversion by Dropbox (MIT)",
    files=(
        models.ModelFile("model.bin", _TURBO + "model.bin", 1617884929,
                         "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da"),
        models.ModelFile("config.json", _TURBO + "config.json", 2263,
                         "b0253ea6c0d3bea6b1e19e91a02acfd3b53f4467362efcb5a3e6b16c9b3a9b7e"),
        models.ModelFile("preprocessor_config.json", _TURBO + "preprocessor_config.json", 340,
                         "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"),
        models.ModelFile("tokenizer.json", _TURBO + "tokenizer.json", 2710337,
                         "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd"),
        models.ModelFile("vocabulary.json", _TURBO + "vocabulary.json", 1068114,
                         "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1"),
    ),
)
models.register(SPEECH_MODEL)

# AP-BWE (MIT code and weights, github.com/yxlu-0102/AP-BWE), converted to ONNX by tools/export_apbwe_onnx.py
# and published once on this repository's "models-v1" release.
_MODELS_RELEASE = "https://github.com/BenjaminDanker/Audio-Cleaner-Web/releases/download/models-v1/"
HIGHS_MODEL = models.SpeechModel(
    id="restore-highs",
    title="Model that restores missing highs",
    description="AP-BWE: brings back the high frequencies of phone calls and other band-limited recordings.",
    licence="AP-BWE by Ye-Xin Lu et al. (MIT code and weights), trained on VCTK (CC BY 4.0)",
    files=tuple(
        models.ModelFile(f"{name}to48k.onnx", f"{_MODELS_RELEASE}{name}to48k.onnx", 119093242, sha)
        for name, sha in (
            ("8k", "788d45ed37d399c0eeffca0462d7b9c2928871ae8523af67e96f9395720137a3"),
            ("12k", "eaa2d5822e1094d2ec93fd233be8b1f2a20ae76a41b2702d404aa0b7ac00d618"),
            ("16k", "ca91ce8a19b91555e6c49f1cce4e44547bb1073af7f32eae62edad9ba83ecd44"),
            ("24k", "e662a7fe7243507fe766092a2b62869f55d75952fcce5ef8b1deb6d99076cf55"),
        )
    ),
)
models.register(HIGHS_MODEL)

# Sidon v0.1 (MIT; sarulab-speech/sidon-v0.1), converted to ONNX (docs/research/restore-sidon.md) and published
# on the same release: the optional "rebuild the voice" step.
VOICE_MODEL = models.SpeechModel(
    id="rebuild-voice",
    title="Model that rebuilds the voice",
    description="Sidon: removes room echo and noise and restores dull sound by rebuilding the voice.",
    licence="Sidon by Wataru Nakata, Yuki Saito and the sarulab-speech group (MIT), with w2v-BERT 2.0 by Meta (MIT); trained on FLEURS-R and LibriTTS-R (CC BY 4.0)",
    files=(
        models.ModelFile("sidon_fe.onnx", _MODELS_RELEASE + "sidon_fe.onnx", 397857704,
                         "76222cc7f4dd95eb3d137d81783e73b69b00cab66b65a91790e0e72c96452a75"),
        models.ModelFile("sidon_dec.onnx", _MODELS_RELEASE + "sidon_dec.onnx", 104975475,
                         "08c7547572f000d964f7dc3ef120fdf155c2d936721d9d320d20c6c4a1d652c4"),
    ),
)
models.register(VOICE_MODEL)


def _vendor_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    env = os.environ.get("ACW_DESKTOP_VENDOR")
    return Path(env) if env else Path(__file__).resolve().parents[1] / "vendor"


def _dev_fallback() -> Path | None:
    if getattr(sys, "frozen", False) or not sys.platform.startswith("linux"):
        return None
    return Path.home() / "audio-cleaner"


def _locate(vendor_sub: str, fallback_sub: str, *names: str) -> Path:
    """Directory holding all of `names`: vendor first, then the dev fallback.
    Returns the vendor directory when nothing matches, so errors name it."""
    vendor = _vendor_root() / vendor_sub
    candidates = [vendor]
    fallback = _dev_fallback()
    if fallback is not None:
        candidates.append(fallback / fallback_sub)
    for d in candidates:
        if all((d / n).is_file() for n in names):
            return d
    return vendor


def libdf_path() -> Path:
    from acw_common import dfn  # numpy is only needed once the app runs, not to read the model catalog

    return _locate("native", "bin", dfn.LIB_NAME) / dfn.LIB_NAME


def model_path() -> Path:
    from acw_common import dfn

    return _locate("models", "models", dfn.MODEL_FILE) / dfn.MODEL_FILE


def heavy_noise_model_path() -> Path:
    """DPDFNet-8 (48 kHz), the noise remover for recordings with heavy background noise. When it is
    missing, heavy noise falls back to DeepFilterNet3 (pipeline.clean_audio)."""
    from acw_common import dpdfnet

    return _locate("models", "models/dpdfnet", dpdfnet.MODEL_FILE) / dpdfnet.MODEL_FILE


def ffmpeg_dir() -> Path:
    return _locate("native", "bin", _FFMPEG, _FFPROBE)


def whisper_model_path() -> Path | None:
    """Directory of the speech model for subtitles, or None until it has been downloaded.
    From source, the small test model (vendor/test-models) or the server's models also count."""
    found = models.location(SPEECH_MODEL)
    if found is not None:
        return found
    if getattr(sys, "frozen", False):
        return None
    candidates = [_vendor_root() / "test-models" / "whisper-small"]
    fallback = _dev_fallback()
    if fallback is not None:
        candidates += [fallback / "models" / "whisper" / "large-v3-turbo", fallback / "models" / "whisper" / "small"]
    for d in candidates:
        if all((d / n).is_file() for n in _WHISPER_FILES):
            return d
    return None


def highs_model_dir() -> Path | None:
    """Folder of the AP-BWE models once downloaded (or the server's copy when run from source), else None."""
    from acw_common import bwe

    found = models.location(HIGHS_MODEL)
    if found is not None:
        return found
    fallback = _dev_fallback()
    if fallback is not None and bwe.available(fallback / "models" / "ap-bwe"):
        return fallback / "models" / "ap-bwe"
    return None


def voice_model_dir() -> Path | None:
    """Folder of Sidon's ONNX files once installed (or the server's copy when run from source), else None."""
    from acw_common import restore

    found = models.location(VOICE_MODEL)
    if found is not None:
        return found
    fallback = _dev_fallback()
    if fallback is not None and restore.available(fallback / "models" / "sidon"):
        return fallback / "models" / "sidon"
    return None


def speech_ready() -> bool:
    return whisper_model_path() is not None


def licenses_dir() -> Path:
    return _vendor_root() / "licenses"


def check() -> list[str]:
    """File names of shipped components that are missing (empty when all present). The speech model is
    not one of them: it is downloaded when subtitles are first wanted (speech_ready())."""
    from acw_common import dfn

    missing = []
    if not libdf_path().is_file():
        missing.append(dfn.LIB_NAME)
    if not model_path().is_file():
        missing.append(dfn.MODEL_FILE)
    ff = ffmpeg_dir()
    missing += [n for n in (_FFMPEG, _FFPROBE) if not (ff / n).is_file()]
    return missing
