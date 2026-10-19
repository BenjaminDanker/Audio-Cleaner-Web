#!/usr/bin/env python3
"""Files for `productbuild --distribution` around the app's component package (make_pkg.sh):

    python distribution.py OUT_DIR --version 2.0.0 --arch arm64 --eula ../EULA.txt --models-gb 2.1

writes OUT_DIR/distribution.xml, OUT_DIR/welcome.txt and OUT_DIR/EULA.txt. The distribution makes the Mac
installer refuse macOS older than 14 (the app does not start there) before anything is downloaded, shows the
licence agreement, and says up front that setup downloads the models and takes a while. Pure Python, so the
output is unit-tested on any machine.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

IDENTIFIER = "local.audio-cleaner.desktop"
COMPONENT = "AudioCleaner-component.pkg"
MIN_MACOS = "14.0"


def welcome_text(models_gb: float) -> str:
    return (
        "Audio Cleaner removes background noise from the voice in your videos and recordings, on your own "
        "computer.\n\n"
        f"While it installs, setup downloads about {models_gb:.1f} GB of models (for noise removal, subtitles, "
        "phone-quality recordings and rebuilding voices). That can take several minutes, and the installer may seem to stand "
        "still meanwhile. Please keep your internet connection on until it says it is done.\n\n"
        "If a download does not work, the app is installed anyway and offers the missing models (the noise removal "
        "models, which it cannot work without, as soon as it starts; the others in Settings).\n\n"
        "Already installed? Just run this installer again: it replaces the app with this version, keeps your "
        "settings, history and downloaded models, and does not download the models again.\n"
    )


def distribution_xml(version: str, arch: str, *, title: str = "Audio Cleaner") -> str:
    """The distribution definition. `arch` is arm64 or x86_64 (the package is built for one of them)."""
    if arch not in ("arm64", "x86_64"):
        raise ValueError(f"unknown architecture {arch!r}")
    hosts = "arm64" if arch == "arm64" else "x86_64,arm64"  # the Intel app also runs on Apple silicon (Rosetta)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="1">
    <title>{escape(title)}</title>
    <welcome file="welcome.txt" mime-type="text/plain"/>
    <license file="EULA.txt" mime-type="text/plain"/>
    <options customize="never" hostArchitectures="{hosts}"/>
    <allowed-os-versions>
        <os-version min="{MIN_MACOS}"/>
    </allowed-os-versions>
    <domains enable_anywhere="false" enable_currentUserHome="false" enable_localSystem="true"/>
    <choices-outline>
        <line choice="default">
            <line choice={quoteattr(IDENTIFIER)}/>
        </line>
    </choices-outline>
    <choice id="default"/>
    <choice id={quoteattr(IDENTIFIER)} visible="false">
        <pkg-ref id={quoteattr(IDENTIFIER)}/>
    </choice>
    <pkg-ref id={quoteattr(IDENTIFIER)} version={quoteattr(version)} onConclusion="none">{COMPONENT}</pkg-ref>
</installer-gui-script>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--version", required=True)
    ap.add_argument("--arch", required=True, choices=["arm64", "x86_64"])
    ap.add_argument("--eula", required=True, type=Path)
    ap.add_argument("--models-gb", required=True, type=float)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "distribution.xml").write_text(distribution_xml(a.version, a.arch), encoding="utf-8")
    (a.out / "welcome.txt").write_text(welcome_text(a.models_gb), encoding="utf-8")
    shutil.copyfile(a.eula, a.out / "EULA.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
