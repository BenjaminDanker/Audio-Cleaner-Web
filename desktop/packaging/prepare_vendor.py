#!/usr/bin/env python3
"""Assemble the vendor directory the PyInstaller spec bundles (and the app reads when run from source).

    vendor/native/{libdf.so|libdf.dylib|df.dll, ffmpeg[.exe], ffprobe[.exe]}
    vendor/models/DeepFilterNet3_onnx.tar.gz
    vendor/models/whisper-small/{model.bin, config.json, tokenizer.json, vocabulary.txt}
    vendor/licenses/...   notices of everything that ends up in the bundle (shown in the About dialog)

    python prepare_vendor.py --libdf PATH --ffmpeg-dir DIR --model PATH
        [--whisper-model DIR | --download-whisper] [--deepfilter-src DIR | --deepfilter-notices DIR]
        [--rust-target TRIPLE] [--out DIR]

CI builds libDF on one runner and assembles the app on another, so the DeepFilterNet notices (which need the
Rust toolchain and the build checkout) can be produced early and carried along:

    python prepare_vendor.py --write-deepfilter-notices DIR [--deepfilter-src DIR] [--rust-target TRIPLE]
    ... later: prepare_vendor.py ... --deepfilter-notices DIR

The Python package notices are read from the interpreter that runs this script, so run it with the
environment the app is frozen in. Anything missing is an error: a bundle without its notices must not ship.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESKTOP = HERE.parent
SUPPLEMENTAL = HERE / "licenses"

DFN_MODEL_SHA256 = "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616"
LIB_NAMES = ("libdf.so", "libdf.dylib", "df.dll")
FFMPEG_NOTICES = ("COPYING.LGPLv2.1", "FFmpeg-LICENSE.md", "ffmpeg-buildinfo.txt", "lame-COPYING", "opus-COPYING")

# Subtitle model: Systran/faster-whisper-small (CTranslate2 conversion of OpenAI Whisper small, MIT).
WHISPER_REPO = "Systran/faster-whisper-small"
WHISPER_REVISION = "536b0662742c02347bc0e980a01041f333bce120"
WHISPER_FILES = {
    "config.json": "b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828",
    "model.bin": "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671",
    "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
    "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
}
WHISPER_DIR_NAME = "whisper-small"

# Distributions whose notices are collected (plus everything they require at run time).
PYTHON_ROOTS = [
    "PySide6_Essentials", "shiboken6", "numpy", "scipy", "pyloudnorm", "faster-whisper", "ctranslate2",
    "tokenizers", "huggingface_hub", "onnxruntime",
]
OPTIONAL_ROOTS = ["future"]  # older pyloudnorm releases need it
# Installed but not shipped: PyAV carries GPL builds of x264/x265, so the frozen app uses a stub instead
# (see audio_cleaner.spec and stubs/av). Nothing of it is in the bundle, so it gets no notice.
NOT_BUNDLED = {"av"}
# Distributions whose wheel carries no licence file: the text to use instead (in packaging/licenses/).
SUPPLEMENT = {
    "ctranslate2": "CTranslate2-MIT.txt",
    "tokenizers": "Apache-2.0.txt",
    "hf-xet": "Apache-2.0.txt",
    "flatbuffers": "Apache-2.0.txt",
}
QT_DISTS = {"pyside6-essentials", "shiboken6"}  # covered by the Qt notice and LGPL-3.0.txt

LICENSE_NAME = re.compile(r"^(licen[sc]e|copying|notice|unlicense|thirdpartynotices)([-_.].*)?$", re.I)
CRATE_LICENSE_NAME = re.compile(r"^(licen[sc]e|copying|unlicense)([-_.].*)?$", re.I)


def fail(msg: str) -> "NoReturn":  # noqa: F821
    print(f"prepare_vendor: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def need_file(path: Path, what: str) -> Path:
    if not path.is_file():
        fail(f"{what} not found: {path}")
    return path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").rstrip()


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# ---- native pieces and models --------------------------------------------------------------------------
def copy_native(libdf: Path, ffmpeg_dir: Path, out: Path) -> None:
    need_file(libdf, "libDF")
    if libdf.name not in LIB_NAMES:
        fail(f"--libdf must be named one of {', '.join(LIB_NAMES)} (got {libdf.name})")
    native = out / "native"
    native.mkdir(parents=True, exist_ok=True)
    shutil.copy2(libdf, native / libdf.name)
    exe = ".exe" if (ffmpeg_dir / "ffmpeg.exe").is_file() else ""
    for n in (f"ffmpeg{exe}", f"ffprobe{exe}"):
        shutil.copy2(need_file(ffmpeg_dir / n, "FFmpeg program"), native / n)
        if not exe:
            (native / n).chmod(0o755)


def copy_models(model: Path, whisper_dir: Path | None, download: bool, out: Path) -> None:
    need_file(model, "DeepFilterNet3 model")
    if sha256_of(model) != DFN_MODEL_SHA256:
        fail(f"{model} is not the expected DeepFilterNet3_onnx.tar.gz (sha256 mismatch)")
    models = out / "models"
    models.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, models / "DeepFilterNet3_onnx.tar.gz")

    target = models / WHISPER_DIR_NAME
    if download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            fail("--download-whisper needs huggingface_hub (pip install -r desktop/requirements.lock)")
        src = Path(snapshot_download(WHISPER_REPO, revision=WHISPER_REVISION, allow_patterns=list(WHISPER_FILES)))
    else:
        assert whisper_dir is not None
        src = whisper_dir
        if not src.is_dir():
            fail(f"subtitle model directory not found: {src} (use --whisper-model DIR or --download-whisper)")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name, want in WHISPER_FILES.items():
        f = need_file(src / name, f"subtitle model file {name}")
        got = sha256_of(f)
        if got != want:
            fail(f"{f}: sha256 {got} does not match the pinned {WHISPER_REPO}@{WHISPER_REVISION[:8]} ({want})")
        shutil.copy2(f, target / name)  # follows the symlinks of a huggingface cache


# ---- licence collection ---------------------------------------------------------------------------------
class Notices:
    """Writes the files in vendor/licenses and remembers what each one is for (for the index)."""

    def __init__(self, out: Path):
        self.dir = out / "licenses"
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        self.index: list[tuple[str, str]] = []

    def add_text(self, name: str, text: str, description: str) -> None:
        (self.dir / name).write_text(text.rstrip() + "\n", encoding="utf-8")
        self.index.append((name, description))

    def add_file(self, name: str, src: Path, description: str) -> None:
        self.add_text(name, read_text(need_file(src, f"licence file for {name}")), description)


def ffmpeg_notices(n: Notices, ffmpeg_dir: Path) -> None:
    desc = {
        "ffmpeg-buildinfo.txt": "FFmpeg, libopus, LAME: versions, build configuration, where to get the exact sources",
        "COPYING.LGPLv2.1": "GNU LGPL v2.1, the licence of the bundled ffmpeg and ffprobe",
        "FFmpeg-LICENSE.md": "FFmpeg licensing overview",
        "lame-COPYING": "LAME (MP3 encoder inside ffmpeg), LGPL",
        "opus-COPYING": "libopus (inside ffmpeg), BSD",
    }
    for name in FFMPEG_NOTICES:
        n.add_file(name, ffmpeg_dir / name, desc[name])


DFN_NOTICE_FILES = (
    ("DeepFilterNet-LICENSE-MIT", "DeepFilterNet (noise reduction), MIT option"),
    ("DeepFilterNet-LICENSE-APACHE", "DeepFilterNet, Apache-2.0 option"),
    ("rust-crates.txt", "Rust crates compiled into the noise reduction library (libDF)"),
)


def make_deepfilter_notices(src: Path, rust_target: str | None, dest: Path) -> None:
    """Write the DeepFilterNet licence files and the Rust crate list into `dest`."""
    if not (src / "libDF").is_dir():
        fail(f"DeepFilterNet checkout not found: {src} (run native/deepfilter/build.sh, or pass --deepfilter-src)")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "DeepFilterNet-LICENSE-MIT").write_text(read_text(need_file(src / "LICENSE-MIT", "DeepFilterNet LICENSE-MIT")) + "\n", encoding="utf-8")
    (dest / "DeepFilterNet-LICENSE-APACHE").write_text(read_text(need_file(src / "LICENSE-APACHE", "DeepFilterNet LICENSE-APACHE")) + "\n", encoding="utf-8")
    (dest / "rust-crates.txt").write_text(rust_crates(src, rust_target) + "\n", encoding="utf-8")


def deepfilter_notices(n: Notices, notices_dir: Path) -> None:
    for name, desc in DFN_NOTICE_FILES:
        n.add_file(name, notices_dir / name, desc)


def rust_crates(src: Path, rust_target: str | None) -> str:
    """Notices for the crates of deep_filter's C API build. `cargo tree` gives the exact feature-aware list
    (as web/scripts/gen-rust-licenses.mjs does for the WebAssembly build); `cargo metadata` locates the sources."""
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
    host = rust_target or os.environ.get("CARGO_BUILD_TARGET")
    if not host:
        out = subprocess.run(["rustc", "-vV"], capture_output=True, text=True, encoding="utf-8", check=True).stdout
        host = re.search(r"^host: (\S+)", out, re.M).group(1)  # type: ignore[union-attr]
    base = [cargo]
    tree = subprocess.run(
        [*base, "tree", "--locked", "-p", "deep_filter", "--no-default-features", "--features", "capi",
         "-e", "normal", "--prefix", "none", "--target", host, "--format", "{p}|{l}|{r}"],
        cwd=src, capture_output=True, text=True, encoding="utf-8",
    )
    if tree.returncode:
        fail(f"cargo tree failed:\n{tree.stderr}")
    meta = subprocess.run(
        [*base, "metadata", "--manifest-path", str(src / "libDF" / "Cargo.toml"), "--format-version", "1", "--locked",
         "--no-default-features", "--features", "capi", "--filter-platform", host],
        cwd=src, capture_output=True, text=True, encoding="utf-8",
    )
    if meta.returncode:
        fail(f"cargo metadata failed:\n{meta.stderr}")
    manifests = {(p["name"], p["version"]): Path(p["manifest_path"]).parent for p in json.loads(meta.stdout)["packages"]}

    crates: dict[tuple[str, str], tuple[str, str]] = {}
    for raw in tree.stdout.splitlines():
        line = re.sub(r" \(\*\)| \(proc-macro\)", "", raw).strip()
        if not line:
            continue
        m = re.match(r"^(\S+) v(\S+)(?: \([^)]*\))?\|([^|]*)\|(.*)$", line)
        if not m:
            fail(f"cannot parse cargo tree line: {line}")
        name, version, licence, repo = m.groups()
        if name == "deep_filter":
            continue  # DeepFilterNet itself: DeepFilterNet-LICENSE-*
        crates[(name, version)] = (licence or "unspecified", repo)
    if not crates:
        fail("cargo tree listed no crates")

    seen: dict[str, str] = {}
    sections = []
    no_file = 0
    for (name, version), (licence, repo) in sorted(crates.items()):
        d = manifests.get((name, version))
        if d is None:
            fail(f"crate {name} {version} missing from cargo metadata")
        files = sorted(f for f in d.iterdir() if f.is_file() and CRATE_LICENSE_NAME.match(f.name) and f.suffix not in (".rs", ".toml"))
        parts = [f"===== {name} {version} =====", f"Licence: {licence}"] + ([f"Source: {repo}"] if repo else [])
        if not files:
            no_file += 1
            parts.append("(The published crate contains no separate licence file; it is licensed as stated above.)")
        for f in files:
            text = read_text(f)
            if len(text) < 30:
                continue
            key = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
            if key in seen:
                parts += ["", f"--- {f.name}: identical to {seen[key]} (printed above) ---"]
            else:
                seen[key] = f"{name} {version} {f.name}"
                parts += ["", f"--- {f.name} ---", text]
        sections.append("\n".join(parts))
    summary = "\n".join(f"{n} {v}: {lic}" for (n, v), (lic, _) in sorted(crates.items()))
    header = (
        "Third-party Rust crates compiled into libDF (DeepFilterNet's C API, the noise reduction library)\n"
        "=================================================================================================\n"
        f"{len(crates)} crates for {host}, from the dependency tree of native/deepfilter/Cargo.lock (feature capi).\n"
        "Where a crate offers several licences (for example \"MIT OR Apache-2.0\") we use it under either one.\n"
        "DeepFilterNet itself: see DeepFilterNet-LICENSE-MIT and DeepFilterNet-LICENSE-APACHE.\n\n"
        f"Summary\n-------\n{summary}\n\nLicence texts\n-------------\n"
        "Identical texts are printed once; later crates refer back to the first occurrence.\n\n"
    )
    print(f"  rust crates: {len(crates)}, {len(seen)} distinct texts, {no_file} without a licence file")
    return header + "\n\n".join(sections)


def python_closure() -> list[metadata.Distribution]:
    from packaging.requirements import Requirement

    found: dict[str, metadata.Distribution] = {}
    todo = [(r, True) for r in PYTHON_ROOTS] + [(r, False) for r in OPTIONAL_ROOTS]
    while todo:
        name, required = todo.pop()
        key = canon(name)
        if key in found or key in NOT_BUNDLED:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            if required:
                fail(f"Python package {name} is not installed (needed for the notices)")
            continue
        found[key] = dist
        for req_text in dist.requires or []:
            req = Requirement(req_text)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            todo.append((req.name, True))
    return sorted(found.values(), key=lambda d: canon(d.metadata["Name"]))


def dist_license_files(dist: metadata.Distribution) -> list[tuple[str, Path]]:
    """(label, path) of licence files in the dist-info (or, failing that, at the top of the package)."""
    found: list[tuple[str, Path]] = []
    for f in dist.files or []:
        parts = f.parts
        if len(parts) >= 2 and parts[0].endswith(".dist-info") and LICENSE_NAME.match(f.name):
            found.append(("/".join(parts[1:]), Path(dist.locate_file(f))))
    if not found:
        for f in dist.files or []:
            if len(f.parts) == 2 and LICENSE_NAME.match(f.name) and not f.parts[0].endswith((".dist-info", ".libs")):
                found.append(("/".join(f.parts), Path(dist.locate_file(f))))
    return [(label, p) for label, p in found if p.is_file()]


def python_notices(n: Notices) -> None:
    dists = python_closure()
    blocks = []
    seen: dict[str, str] = {}
    names = []
    for dist in dists:
        name, version = dist.metadata["Name"], dist.version
        key = canon(name)
        licence = dist.metadata.get("License-Expression") or dist.metadata.get("License") or ""
        if "\n" in licence or len(licence) > 80:
            licence = ""
        if not licence:
            licence = "; ".join(c.split("::")[-1].strip() for c in dist.metadata.get_all("Classifier") or [] if c.startswith("License ::"))
        head = [f"===== {name} {version} =====", f"Licence: {licence or 'see below'}"]
        home = dist.metadata.get("Home-page") or next(
            (u.split(",", 1)[1].strip() for u in dist.metadata.get_all("Project-URL") or [] if u.lower().startswith(("homepage", "source", "repository"))), "")
        if home:
            head.append(f"Home: {home}")
        names.append(f"{name} {version}")
        parts = list(head)
        if key in QT_DISTS:
            parts.append("Qt for Python: used under the GNU LGPL v3, dynamically linked; see Qt-notice in README.txt and LGPL-3.0.txt.")
        else:
            files = dist_license_files(dist)
            if not files and key in SUPPLEMENT:
                files = [(f"{SUPPLEMENT[key]} (the wheel ships no licence file; text from the project's stated licence)", SUPPLEMENTAL / SUPPLEMENT[key])]
            if not files:
                fail(f"no licence file for Python package {name} {version}; add one to desktop/packaging/licenses and SUPPLEMENT")
            for label, path in files:
                text = read_text(need_file(path, f"licence file of {name}"))
                digest = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
                if digest in seen:
                    parts += ["", f"--- {label}: identical to {seen[digest]} (printed above) ---"]
                else:
                    seen[digest] = f"{name} {version} {label}"
                    parts += ["", f"--- {label} ---", text]
        blocks.append("\n".join(parts))
    header = (
        "Third-party Python packages bundled with Audio Cleaner\n"
        "======================================================\n"
        "numpy and SciPy also include the notices of the OpenBLAS, LAPACK and GCC runtime libraries they ship.\n"
        "Identical texts are printed once.\n\n" + "\n".join(names) + "\n\n"
    )
    n.add_text("python-packages.txt", header + "\n\n".join(blocks), "Python packages (Qt for Python, NumPy, SciPy, faster-whisper, CTranslate2, ...)")
    print(f"  python packages: {len(dists)}")


def python_licence(n: Notices) -> None:
    stdlib = Path(sysconfig.get_path("stdlib"))
    candidates = [Path(sys.base_prefix) / "LICENSE.txt", stdlib / "LICENSE.txt", stdlib.parent / "LICENSE.txt", Path("/usr/share/doc/python3/copyright")]
    for c in candidates:
        if c.is_file():
            n.add_file("Python-LICENSE.txt", c, "Python (PSF licence)")
            return
    fail("Python's LICENSE.txt not found (looked in " + ", ".join(str(c) for c in candidates) + ")")


def pyinstaller_notice(n: Notices) -> None:
    n.add_text(
        "PyInstaller-NOTICE.txt",
        "PyInstaller\n===========\n"
        "The application is packaged with PyInstaller (https://pyinstaller.org), which is licensed under the GNU GPL v2 or\n"
        "later WITH the following exception, so the bundles it creates may be distributed under any licence:\n\n"
        "  \"In addition to the permissions in the GNU General Public License, the authors give you unlimited permission\n"
        "  to link or embed compiled bootloader and related files into combinations with other programs, and to\n"
        "  distribute those combinations without any restriction coming from the use of those files. (The General Public\n"
        "  License restrictions do apply in other respects; for example, they cover modification of the files.)\"\n\n"
        "The bootloader (the small launcher program) and the runtime hooks are the only PyInstaller parts in the bundle.\n",
        "PyInstaller bootloader exception",
    )


def whisper_notices(n: Notices) -> None:
    n.add_file("Whisper-MIT.txt", SUPPLEMENTAL / "Whisper-MIT.txt",
               f"OpenAI Whisper (MIT): the subtitle speech model, as converted by {WHISPER_REPO}")
    n.add_file("SileroVAD-MIT.txt", SUPPLEMENTAL / "SileroVAD-MIT.txt", "Silero VAD (MIT): the voice activity model inside faster-whisper")


def qt_notices(n: Notices) -> None:
    pyside = metadata.version("PySide6_Essentials")
    n.add_file("LGPL-3.0.txt", SUPPLEMENTAL / "LGPL-3.0.txt", "GNU LGPL v3: the licence Qt and PySide6 are used under")
    n.add_file("GPL-3.0.txt", SUPPLEMENTAL / "GPL-3.0.txt", "GNU GPL v3 (the LGPL v3 is an addition to it)")
    n.index.append(("Qt-notice", f"(in README.txt) Qt / PySide6 {pyside}: LGPL notice and written offer"))


def write_readme(n: Notices) -> None:
    from audio_cleaner_desktop import __version__  # noqa: PLC0415 - needs desktop/ on sys.path

    pyside = metadata.version("PySide6_Essentials")
    rows = "\n".join(f"  {name:<32}{desc}" for name, desc in sorted(n.index, key=lambda r: r[0].lower()) if name != "Qt-notice")
    text = f"""Audio Cleaner {__version__}: third-party software notices
==========================================================

Audio Cleaner is free to use; its own licence is shown by the installer. It contains and runs the software
listed below, each under its own licence. The texts are in this folder.

Qt and PySide6 (Qt for Python {pyside})
----------------------------------------
This application uses Qt, through PySide6, under the GNU Lesser General Public License version 3 (LGPL-3.0.txt,
GPL-3.0.txt). Qt and PySide6 are linked dynamically: they are the separate library files in the application
folder (Qt6*.dll / libQt6*.so / QtCore etc. framework files, and the PySide6 / shiboken6 folders). You may replace
them with your own build of Qt / PySide6 of a compatible version; the application will use the replacement.
Source code of Qt for Python and Qt: https://download.qt.io/official_releases/QtForPython/ and https://code.qt.io .
Written offer: if you cannot get the sources from there, ask Benjamin Danker (who gave you this app), who will send
you the complete corresponding source code of the Qt and PySide6 versions in this application, for three years
after the version was released and for as long as we offer it. Qt is a registered trademark of The Qt Company Ltd.
Qt's multimedia library also includes FFmpeg libraries (LGPL v2.1 or later, source: https://ffmpeg.org) that Qt
uses to play the preview; they are separate files in the application folder.

FFmpeg
------
The programs ffmpeg and ffprobe in the "native" folder are FFmpeg built under the LGPL v2.1, without GPL or
non-free components. Audio Cleaner starts them as separate programs. ffmpeg-buildinfo.txt names the exact
source releases and the build configuration; you may replace both files with your own FFmpeg build.

Index of this folder
--------------------
{rows}

Python (PSF licence) and the Python packages are part of the bundled interpreter and libraries.
"""
    (n.dir / "README.txt").write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--libdf", type=Path, help="libdf.so / libdf.dylib / df.dll from native/deepfilter/build.sh")
    ap.add_argument("--ffmpeg-dir", type=Path, help="output directory of desktop/packaging/build_ffmpeg.sh")
    ap.add_argument("--model", type=Path, help="DeepFilterNet3_onnx.tar.gz")
    ap.add_argument("--whisper-model", type=Path, default=Path.home() / "audio-cleaner" / "models" / "whisper" / "small",
                    help="directory of Systran/faster-whisper-small (default: %(default)s)")
    ap.add_argument("--download-whisper", action="store_true", help=f"download {WHISPER_REPO}@{WHISPER_REVISION[:8]} with huggingface_hub (CI)")
    ap.add_argument("--deepfilter-src", type=Path, default=Path.home() / "audio-cleaner" / "src" / "deepfilter-build",
                    help="checkout prepared by native/deepfilter/build.sh (default: %(default)s)")
    ap.add_argument("--deepfilter-notices", type=Path, metavar="DIR",
                    help="notices made earlier by --write-deepfilter-notices (instead of --deepfilter-src)")
    ap.add_argument("--write-deepfilter-notices", type=Path, metavar="DIR",
                    help="only write the DeepFilterNet licence files and the Rust crate list into DIR, then exit")
    ap.add_argument("--rust-target", help="Rust target triple the libDF was built for (default: CARGO_BUILD_TARGET or the host)")
    ap.add_argument("--out", type=Path, default=DESKTOP / "vendor")
    a = ap.parse_args()

    if a.write_deepfilter_notices:
        make_deepfilter_notices(a.deepfilter_src, a.rust_target, a.write_deepfilter_notices)
        print(f"DeepFilterNet notices written to {a.write_deepfilter_notices}")
        return 0
    for flag in ("libdf", "ffmpeg_dir", "model"):
        if getattr(a, flag) is None:
            ap.error(f"--{flag.replace('_', '-')} is required")

    sys.path.insert(0, str(DESKTOP))
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"vendor directory: {out}")
    copy_native(a.libdf, a.ffmpeg_dir, out)
    copy_models(a.model, a.whisper_model, a.download_whisper, out)
    n = Notices(out)
    ffmpeg_notices(n, a.ffmpeg_dir)
    if a.deepfilter_notices:
        deepfilter_notices(n, a.deepfilter_notices)
    else:
        with tempfile.TemporaryDirectory(prefix="acw-dfn-notices-") as tmp:
            make_deepfilter_notices(a.deepfilter_src, a.rust_target, Path(tmp))
            deepfilter_notices(n, Path(tmp))
    python_notices(n)
    python_licence(n)
    pyinstaller_notice(n)
    whisper_notices(n)
    qt_notices(n)
    write_readme(n)
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"done: {total / 1e6:.0f} MB in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
