"""Locate the pieces the app needs: libDF, FFmpeg (shipped) and the models (downloaded once, see models.py).

Frozen (PyInstaller) builds find the shipped pieces under sys._MEIPASS:

    native/{df.dll|libdf.dylib|libdf.so, ffmpeg[.exe], ffprobe[.exe]}
    licenses/

From source the same layout is read from $ACW_DESKTOP_VENDOR (default
desktop/vendor, filled by the packaging scripts). On Linux, missing pieces fall
back to the server layout in ~/audio-cleaner/{bin,models}.

No model is in the app. The noise removal models (DeepFilterNet3 and DPDFNet-8, 23 MB, REQUIRED: noise removal is what the
app is for) and the ones for subtitles and rebuilding the voice (2.1 GB) are downloaded by the installer during setup;
the app downloads the noise models itself when they are missing, and offers the others again when setup's download
failed (or after a portable run). Runs from source (tests, CI) also find the noise models in
vendor/models (prepare_vendor.py --download-noise-models) or the server's models, and may use the small test speech model that
prepare_vendor.py puts in vendor/test-models/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import config, models

_EXE = ".exe" if os.name == "nt" else ""
_FFMPEG = f"ffmpeg{_EXE}"
_FFPROBE = f"ffprobe{_EXE}"
_WHISPER_FILES = ("model.bin", "config.json", "tokenizer.json")

# The speech model comes straight from Hugging Face, pinned to one commit (its files are published there as they are).
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

# The ONNX models are ours: Sidon v0.1 (MIT; sarulab-speech/sidon-v0.1), converted to ONNX (docs/research/restore-sidon.md), and the
# noise removal ones. Each is a plain per-file model like the speech model, downloaded from our Hugging Face repository
# (config.hf_models_url), pinned by revision, size and SHA-256, so a user who wants one model downloads only its files. The
# licence texts come with each model (`licences`: LICENSES/<name> in the model's folder; not part of `files`, so they are not in the .complete marker, see models.py).
def _hf(path: str, size: int, sha256: str, *, name: str = "") -> models.ModelFile:
    """A file of the models repository; `name` is its path inside the model's folder (default: the repository path)."""
    return models.ModelFile(name or path, config.hf_models_url(path), size, sha256)


_LICENCE_TEXTS = {
    "Sidon-MIT.txt": (1722, "ba0678e7b808bfdf63e376a58586864a93e28345c7c6304fef743de9e1dbf15d"),
    "Training-data-attribution.txt": (1605, "19bb96ffa127f60569675f31e263df72bd683f8d7ed1f2382f9f78843dfab16d"),
    "DeepFilterNet-MIT.txt": (1083, "24e6bb09c928af8d8e56268082f87413247ce36b39dd5d33add2f9893968065e"),
    "DeepFilterNet-Apache-2.0.txt": (10837, "1eaee808c5fb6b4e895ba30425285a5cdc5dd25bba2cd230f264c2200c331aec"),
    "DPDFNet-Apache-2.0.txt": (11357, "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
    "w2v-BERT-2.0-MIT.txt": (1402, "5285eea4fed64499b8713ea99bf1aa947e6cf33c647ef8842b93adac9ce159bf"),
}


def _licences(*names: str) -> tuple[models.ModelFile, ...]:
    return tuple(_hf(f"LICENSES/{n}", *_LICENCE_TEXTS[n]) for n in names)


# Noise removal: DeepFilterNet3 (the main remover, run by the bundled libDF) and DPDFNet-8 (heavy noise). One model with
# both files, because they are used together (the app picks per recording, so either may be needed) and neither is
# optional: one download, one marker, one line in the app, and setup cannot end up with only half of noise removal.
# They are REQUIRED (`required`): the installers insist on them, and the app downloads them by itself when they
# are missing instead of failing on a missing file. The files keep their bare names in the model's folder.
NOISE_MODEL = models.SpeechModel(
    id="noise-removal",
    title="Noise removal models",
    description="DeepFilterNet3 and DPDFNet-8: remove background noise (the second one for heavy noise).",
    licence="DeepFilterNet3 by Hendrik Schroeter et al. (MIT or Apache-2.0), DPDFNet-8 by Ceva (Apache-2.0)",
    files=(
        _hf("deepfilternet3/DeepFilterNet3_onnx.tar.gz", 7983136,
            "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616", name="DeepFilterNet3_onnx.tar.gz"),
        _hf("dpdfnet/dpdfnet8_48khz_hr_batch.onnx", 14777093,
            "b50417b070e63b22d32c3ab4e2ee056805cb8bc5ca8db053b1ce937bfc41bdaa", name="dpdfnet8_48khz_hr_batch.onnx"),
    ),
    licences=_licences("DeepFilterNet-MIT.txt", "DeepFilterNet-Apache-2.0.txt", "DPDFNet-Apache-2.0.txt"),
    required=True,
)
models.register(NOISE_MODEL)


# The optional "rebuild the voice" step.
VOICE_MODEL = models.SpeechModel(
    id="rebuild-voice",
    title="Model that rebuilds the voice",
    description="Sidon: removes room echo and noise and restores dull sound by rebuilding the voice.",
    licence="Sidon by Wataru Nakata, Yuki Saito and the sarulab-speech group (MIT), with w2v-BERT 2.0 by Meta (MIT); trained on FLEURS-R and LibriTTS-R (CC BY 4.0)",
    files=(
        _hf("sidon/sidon_fe.onnx", 397857704, "76222cc7f4dd95eb3d137d81783e73b69b00cab66b65a91790e0e72c96452a75",
            name="sidon_fe.onnx"),
        _hf("sidon/sidon_dec.onnx", 104975475, "08c7547572f000d964f7dc3ef120fdf155c2d936721d9d320d20c6c4a1d652c4",
            name="sidon_dec.onnx"),
    ),
    licences=_licences("Sidon-MIT.txt", "w2v-BERT-2.0-MIT.txt", "Training-data-attribution.txt"),
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


def _noise_file(name: str, *dev_dirs: Path) -> Path:
    """One file of the noise removal models: the downloaded copy (the app's own or the installer's), else, when run from
    source, the first of `dev_dirs` that has it (vendor/models for tests and CI, the server's models). When nothing is
    there it is where the download will put it, so errors name it."""
    found = models.location(NOISE_MODEL)
    if found is not None:
        return found / name
    if not getattr(sys, "frozen", False):
        for d in dev_dirs:
            if (d / name).is_file():
                return d / name
    return models.model_dir(NOISE_MODEL) / name


def _noise_dev_dirs(server_sub: str = "") -> list[Path]:
    dirs = [_vendor_root() / "models"]
    fallback = _dev_fallback()
    if fallback is not None:
        dirs.append(fallback / "models" / server_sub if server_sub else fallback / "models")
    return dirs


def model_path() -> Path:
    """DeepFilterNet3's model tarball (see NOISE_MODEL)."""
    from acw_common import dfn

    return _noise_file(dfn.MODEL_FILE, *_noise_dev_dirs())


def heavy_noise_model_path() -> Path:
    """DPDFNet-8 (48 kHz), the noise remover for recordings with heavy background noise (see NOISE_MODEL)."""
    from acw_common import dpdfnet

    return _noise_file(dpdfnet.MODEL_FILE, *_noise_dev_dirs("dpdfnet"))


def noise_ready() -> bool:
    """Both noise removal models are there (downloaded, or found by the source-run fallbacks)."""
    return model_path().is_file() and heavy_noise_model_path().is_file()


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


def model_licenses_dirs() -> list[Path]:
    """The folders with the licence texts that came with the downloaded models (<model>/LICENSES of each installed model,
    and the shared <models>/LICENSES of older installs); empty before there are any."""
    return models.licence_dirs()


def check() -> list[str]:
    """File names of shipped components that are missing (empty when all present). The models are not shipped: the
    noise removal models are required but downloaded (noise_ready()), the speech model when subtitles are first
    wanted (speech_ready())."""
    from acw_common import dfn

    missing = []
    if not libdf_path().is_file():
        missing.append(dfn.LIB_NAME)
    ff = ffmpeg_dir()
    missing += [n for n in (_FFMPEG, _FFPROBE) if not (ff / n).is_file()]
    return missing
