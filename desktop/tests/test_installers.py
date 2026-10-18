"""What the installers are made of: the Inno Setup include and the Mac postinstall generated from the model
catalog and the Mac distribution. None of this needs Windows or a Mac. (The check that every model can be found at its
pinned address is test_check_model_urls.py.)"""
from __future__ import annotations

import shutil
import subprocess
import sys
import xml.dom.minidom
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"
for sub in (PACKAGING, PACKAGING / "macos"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

import distribution  # noqa: E402
import installer_models  # noqa: E402

from audio_cleaner_desktop import models, resources  # noqa: E402

# The macOS installer scripts are shell scripts for /bin/bash on a Mac: they run here on Linux and macOS (Git Bash
# on Windows reads paths and permissions differently, and nothing there ever runs them).
runs_bash_scripts = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="the macOS shell scripts run on Linux and macOS")


def test_the_catalog_is_read_without_numpy():
    code = "import sys; sys.modules['numpy'] = None; sys.path[:0] = [r'%s', r'%s']; import installer_models; installer_models.render()"
    r = subprocess.run(
        [sys.executable, "-I", "-c", code % (PACKAGING, PACKAGING.parents[1] / "common")], capture_output=True, text=True,
        cwd=PACKAGING,
    )
    assert r.returncode == 0, r.stderr


def test_windows_include_budgets_space_cleans_up_and_tries_every_download():
    text = installer_models.render()
    plain = [(m, f) for m in installer_models.CATALOG for f in (*m.files, *m.licences)]  # every model's files and licence texts
    # deleteafterinstall would delete the INSTALLED file when setup ends (the temporary download goes with {tmp} anyway)
    assert plain and text.count("ExternalSize:") == len(plain) and "deleteafterinstall" not in text
    assert text.count("    FetchModel(") == len(plain)  # one download each: a failure does not stop the next
    assert "GetSpaceOnDisk64(ExpandConstant('{tmp}')" in text
    assert "AbortedByUser" in text
    assert "Open Audio Cleaner > Settings to finish the download" in text
    assert "when it starts" not in text
    assert "SaveStringToFile" in text and ".complete" in text
    for m, f in plain:
        assert f.url in text and f.sha256 in text and str(f.size) in text
    for gone in ("Archive", "tar.exe", "Unpack", "ExtraDiskSpaceRequired", ".zip"):  # nothing is unpacked any more
        assert gone not in text
    assert "[UninstallDelete]" in text


def test_licence_texts_are_downloaded_into_their_model_but_are_not_part_of_its_marker():
    text = installer_models.render()
    for m in (resources.HIGHS_MODEL, resources.VOICE_MODEL):
        for f in m.licences:
            assert f"DestDir: \"{{app}}\\models\\{m.id}\\LICENSES\"; DestName: \"{f.path.split('/')[1]}\"" in text
        marker = text[text.index(f"MarkerAgrees(ExpandConstant('{{app}}\\models\\{m.id}\\.complete')"):]
        marker = marker[:marker.index("then")]  # the hashes the marker is compared with: the model's files only
        assert all(f.sha256 in marker for f in m.files) and not any(f.sha256 in marker for f in m.licences)
    assert installer_models.total_bytes() == sum(f.size for m in installer_models.CATALOG for f in (*m.files, *m.licences))


def test_the_pascal_strings_are_balanced():
    for line in installer_models.render().splitlines():
        if line.lstrip().startswith(("FetchModel", "if ModelFileReady", "SaveStringToFile", "PlaceModelFile", "ModelDownloads.Add")):
            assert line.replace("''", "").count("'") % 2 == 0, line


def _pascal_words(code: str) -> list[str]:
    """The identifiers of Pascal `code`, without 'strings' and { comments }."""
    words, i, word = [], 0, ""
    while i < len(code):
        c = code[i]
        if c == "'":  # a string; '' inside it is a quote
            i += 1
            while i < len(code) and not (code[i] == "'" and code[i + 1 : i + 2] != "'"):
                i += 2 if code[i] == "'" else 1
        elif c == "{":
            i = code.index("}", i)
        elif c.isalnum() or c == "_":
            word += c
        else:
            if word:
                words.append(word.lower())
            word = ""
        i += 1
    return words


def test_the_pascal_blocks_are_balanced():
    words = _pascal_words(installer_models.render().split("[Code]")[1])
    assert sum(w in ("begin", "try", "case") for w in words) == words.count("end") > 6
    defined = [words[i + 1] for i, w in enumerate(words) if w in ("procedure", "function")]
    assert len(defined) >= 8
    for name in defined:  # Pascal needs a routine to be defined before its first use
        first = words.index(name)
        assert words[first - 1] in ("procedure", "function"), name


@runs_bash_scripts
def test_mac_postinstall_downloads_safely(tmp_path):
    text = installer_models.render_mac()
    for flag in ("--proto '=https'", "--speed-limit 1000", "--speed-time 60", "--max-time 3600", "-C -"):
        assert flag in text
    assert text.count("\nfetch \"") == sum(len(m.files) + len(m.licences) for m in installer_models.CATALOG)
    assert 'shasum -a 256 "$tmp"' in text and "ditto" not in text and "ARCHIVE" not in text
    script = tmp_path / "postinstall"
    script.write_text(text, encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


def _tiny_catalog():
    """Two small models with a licence text each, served by a fake curl: (served bytes by URL, catalog)."""
    import hashlib

    served, catalog = {}, []
    for name, files in (("one", {"a.onnx": b"a" * 5000, "sub/b.json": b"{}"}), ("two", {"c.onnx": b"c" * 3000})):
        def entry(path, data):
            url = f"https://example.invalid/{name}/{path}"
            served[url] = data
            return models.ModelFile(path, url, len(data), hashlib.sha256(data).hexdigest())

        catalog.append(models.SpeechModel(
            name, name, "", "MIT", tuple(entry(p, d) for p, d in files.items()), (entry("LICENSES/L.txt", b"licence " + name.encode()),)))
    return served, tuple(catalog)


def _mac_runner(tmp_path, catalog, served):
    """The generated postinstall as a function to run, with stand-ins for curl, shasum and BSD stat; returns (run, calls, root)."""
    import json

    root, calls = tmp_path / "models", tmp_path / "curl-calls"
    (tmp_path / "served.json").write_text(json.dumps({u: d.decode("latin-1") for u, d in served.items()}))
    text = installer_models.render_mac(catalog).replace(installer_models.MAC_MODELS, str(root)).replace(
        "/var/log/audio-cleaner-install.log", str(tmp_path / "log.txt"))
    prelude = f"""
stat() {{ wc -c < "$2" | tr -d " "; }}
shasum() {{ "{sys.executable}" -I -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest(), sys.argv[1])' "$3"; }}
curl() {{ local out="" url=""; while [ $# -gt 0 ]; do [ "$1" = -o ] && out="$2"; url="$1"; shift; done; echo "$url" >> "{calls}"
  "{sys.executable}" -I -c 'import json, sys; open(sys.argv[2], "wb").write(json.load(open(sys.argv[3]))[sys.argv[1]].encode("latin-1"))' "$url" "$out" "{tmp_path}/served.json"; }}
"""
    script = tmp_path / "run.sh"
    script.write_text(prelude + text.replace("#!/bin/bash", ""), encoding="utf-8")

    def run():
        calls.unlink(missing_ok=True)
        subprocess.run(["bash", str(script)], check=True, capture_output=True, text=True)
        return calls.read_text().split() if calls.exists() else []

    return run, root


@runs_bash_scripts
def test_mac_postinstall_downloads_each_file_once_skips_complete_models_and_refetches_stale_ones(tmp_path):
    served, catalog = _tiny_catalog()
    run, root = _mac_runner(tmp_path, catalog, served)
    first = run()  # first install: every file and licence text, once, from its own address
    assert sorted(first) == sorted(served)
    for m in catalog:
        for f in (*m.files, *m.licences):
            assert (root / m.id / f.path).read_bytes() == served[f.url]
        assert (root / m.id / ".complete").read_text().split() == [f.sha256 for f in m.files]  # not the licences'
    assert run() == []  # the same version again: nothing is downloaded
    (root / "two" / "c.onnx").write_bytes(b"short")  # a truncated file counts as missing
    (root / "two" / ".complete").unlink()
    assert run() == [catalog[1].files[0].url] and (root / "two" / "c.onnx").read_bytes() == served[catalog[1].files[0].url]
    (root / "one" / ".complete").write_text("0" * 64 + "\n")  # a marker naming other hashes: the model's files are fetched again
    assert sorted(run()) == sorted(f.url for f in catalog[0].files)  # (its licence texts are there already)
    assert (root / "one" / ".complete").read_text().split() == [f.sha256 for f in catalog[0].files]
    assert run() == []


@runs_bash_scripts
def test_mac_postinstall_installs_nothing_that_does_not_match_its_sha256(tmp_path):
    served, catalog = _tiny_catalog()
    bad = catalog[1].files[0]
    served[bad.url] = b"x" * bad.size  # right size, wrong content
    run, root = _mac_runner(tmp_path, catalog, served)
    run()
    assert not (root / "two" / "c.onnx").exists() and not (root / "two" / ".complete").exists()
    assert (root / "one" / ".complete").is_file()  # the other model is not held back
    assert "could not download" in (tmp_path / "log.txt").read_text()


def test_mac_distribution_gates_the_os_and_shows_the_licence():
    text = distribution.distribution_xml("2.0.0", "arm64")
    dom = xml.dom.minidom.parseString(text)
    assert dom.getElementsByTagName("os-version")[0].getAttribute("min") == "14.0"
    assert dom.getElementsByTagName("license")[0].getAttribute("file") == "EULA.txt"
    assert dom.getElementsByTagName("welcome")[0].getAttribute("file") == "welcome.txt"
    assert dom.getElementsByTagName("options")[0].getAttribute("hostArchitectures") == "arm64"
    refs = [r for r in dom.getElementsByTagName("pkg-ref") if r.firstChild is not None]
    assert refs[0].getAttribute("version") == "2.0.0" and refs[0].firstChild.data == distribution.COMPONENT
    assert "arm64" in distribution.distribution_xml("2.0.0", "x86_64")  # the Intel app also runs on Apple silicon
    with pytest.raises(ValueError):
        distribution.distribution_xml("2.0.0", "ppc")


def test_mac_welcome_text_names_the_download(tmp_path):
    out = tmp_path / "res"
    eula = PACKAGING / "EULA.txt"
    sys.argv = ["distribution.py", str(out), "--version", "2.0.0", "--arch", "arm64", "--eula", str(eula), "--models-gb", "2.5"]
    assert distribution.main() == 0
    welcome = (out / "welcome.txt").read_text()
    assert "2.5 GB" in welcome and "several minutes" in welcome
    assert (out / "EULA.txt").read_text() == eula.read_text()
    xml.dom.minidom.parse(str(out / "distribution.xml"))


COPYRIGHT = "Copyright (c) 2025-2026 The Audio Cleaner authors"
MIT_LICENSE = f"""MIT License

{COPYRIGHT}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def test_the_eula_is_the_mit_license_and_keeps_the_open_source_rights():
    eula = (PACKAGING / "EULA.txt").read_text(encoding="utf-8")
    assert eula.endswith(MIT_LICENSE)  # the full MIT License with our copyright line, word for word
    assert "Permission is hereby granted, free of charge" in eula
    eula = " ".join(eula.split())  # the prose is wrapped at fixed width
    assert "FFmpeg (LGPL v2.1)" in eula and "Qt (LGPL v3)" in eula
    for model in ("Whisper", "AP-BWE", "Sidon", "w2v-BERT 2.0"):  # the models setup downloads
        assert model in eula, model
    assert "replace the Qt or FFmpeg files with your own builds" in eula  # the LGPL right to swap the libraries
    assert "checks the app's releases on GitHub for a newer version" in eula and "your IP address" in eula  # says what goes online
    assert "from the app's releases on GitHub" in eula and "download server" not in eula  # where the models come from too
    for restriction in ("may not", "reverse engineer", "decompile", "all rights reserved", "end this licence"):
        assert restriction not in eula.lower(), restriction  # no terms on top of the MIT License


def test_the_public_license_is_the_mit_license():
    root = PACKAGING.parents[1]
    # Here the public repository's LICENSE is the template; in the public repository it is the LICENSE at the top.
    template = root / "tools" / "public-desktop" / "template" / "LICENSE"
    assert (template if template.is_file() else root / "LICENSE").read_text(encoding="utf-8") == MIT_LICENSE
    spec = (PACKAGING / "audio_cleaner.spec").read_text(encoding="utf-8")
    assert f'"LegalCopyright", "{COPYRIGHT}"' in spec and f'"NSHumanReadableCopyright": "{COPYRIGHT}"' in spec


# ---- running setup again over an installed app ---------------------------------------------------------------

ISS = (PACKAGING / "windows" / "AudioCleaner.iss").read_text(encoding="utf-8")


def _iss_section(name: str) -> list[str]:
    """The lines of one [Section] of the Inno script, without blank lines and comments."""
    out, inside = [], False
    for line in ISS.splitlines():
        if line.startswith("["):
            inside = line.strip() == f"[{name}]"
        elif inside and line.strip() and not line.lstrip().startswith(";"):
            out.append(line.strip())
    return out


def test_windows_update_replaces_the_program_files_and_never_touches_the_models():
    deletes = _iss_section("InstallDelete")
    assert 'Type: filesandordirs; Name: "{app}\\_internal"' in deletes
    for line in deletes:
        assert "models" not in line.lower() and "{app}\\*" not in line and line.count("{app}") == 1, line
        assert line.split('Name: "')[1].startswith("{app}\\_internal"), line  # only what the build puts there
    # the include may not delete the models either, apart from the uninstaller removing what setup installed
    include = installer_models.render()
    assert "[InstallDelete]" not in include
    assert "DelTree" not in include


def test_windows_update_keeps_the_app_identity_folder_and_mode():
    setup = _iss_section("Setup")
    assert "AppId={{6F0D1A52-3C41-4B7E-9A68-2D5E8B7C41F3}" in setup  # never changes: it is how an upgrade is recognised
    assert "UsePreviousAppDir=yes" in setup and "UsePreviousPrivileges=yes" in setup
    assert "CloseApplications=yes" in setup
    assert not any(line.startswith(("DisableDirPage=yes", "UsePreviousAppDir=no", "UsePreviousPrivileges=no")) for line in setup)


def test_windows_update_downloads_nothing_when_the_models_are_complete():
    text = installer_models.render()
    # the free-space check and every download sit behind "nothing is needed": Need = 0 leaves NextButtonClick first
    body = text[text.index("function NextButtonClick"):]
    assert body.index("ForgetStaleModels;") < body.index("Need := ModelsStillNeeded;") < body.index("if Need = 0 then")
    assert body.index("if Need = 0 then") < body.index("ModelDownloads.Show;")
    needed = text[text.index("function ModelsStillNeeded"):text.index("function NextButtonClick")]
    assert "Result := 0;" in needed
    # each single download is skipped when its file is there with the right size
    assert "if ModelsCancelled or ModelFileReady(ExpandConstant(Dest), Size) then" in text



def test_windows_update_forgets_a_model_made_for_another_catalog():
    text = installer_models.render()
    stale = text[text.index("procedure ForgetStaleModels"):text.index("procedure InitializeWizard")]
    for m in installer_models.CATALOG:
        marker = f"{{app}}\\models\\{m.id}\\.complete"
        assert f"MarkerAgrees(ExpandConstant('{marker}'), " + "#10".join(f"'{f.sha256}'" for f in m.files) + ")" in stale
        for f in m.files:
            assert f"DeleteFile(ExpandConstant('{{app}}\\models\\{m.id}\\{f.path}'))" in stale
        assert f"DeleteFile(ExpandConstant('{marker}'))" in stale
    # no marker: the sizes decide (installs from before the marker existed); it never deletes anything else
    agrees = text[text.index("function MarkerAgrees"):text.index("procedure ForgetStaleModels")]
    assert "if not FileExists(Marker) then" in agrees and "LoadStringFromFile" in agrees
    assert stale.count("DelTree") == 0 and "models'" not in stale.replace("\\models\\", "")


@runs_bash_scripts
def test_mac_preinstall_removes_only_the_installed_app(tmp_path):
    text = installer_models.render_mac_preinstall()
    assert text.startswith("#!/bin/bash\n")
    removals = [ln.strip() for ln in text.splitlines() if "rm -" in ln and not ln.lstrip().startswith("#") and "echo" not in ln.split("rm -")[0]]
    assert removals == ['rm -rf "$APP" >>"$LOG" 2>&1 || echo "could not remove $APP" >>"$LOG"']
    assert 'APP="${VOLUME%/}/Applications/Audio Cleaner.app"' in text
    code = "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))
    assert "Application Support" not in code and "models" not in code  # the models are only mentioned in a comment
    script = tmp_path / "preinstall"
    script.write_text(text, encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    # run it against a pretend volume: the old bundle (with a stale file) goes, everything else stays
    volume = tmp_path / "vol"
    app = volume / "Applications" / "Audio Cleaner.app" / "Contents"
    app.mkdir(parents=True)
    (app / "old-file").write_text("from an older version")
    other = volume / "Applications" / "Other.app"
    other.mkdir()
    models_dir = volume / "Library" / "Application Support" / "Audio Cleaner" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "kept.onnx").write_text("model")
    text = text.replace("/var/log/audio-cleaner-install.log", str(tmp_path / "log.txt"))
    script.write_text(text, encoding="utf-8")
    r = subprocess.run(["bash", str(script), "/some.pkg", "/Applications", str(volume)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert not (volume / "Applications" / "Audio Cleaner.app").exists()
    assert other.is_dir() and (models_dir / "kept.onnx").read_text() == "model"
    # a first install (nothing there) and a second run are fine too
    assert subprocess.run(["bash", str(script), "/some.pkg", "/Applications", str(volume)]).returncode == 0
    # a symlink named like the app is left alone (nothing is followed)
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "file").write_text("x")
    (volume / "Applications" / "Audio Cleaner.app").symlink_to(target)
    assert subprocess.run(["bash", str(script), "/some.pkg", "/Applications", str(volume)]).returncode == 0
    assert (target / "file").is_file()


def test_mac_scripts_are_written_together_and_the_package_uses_them(tmp_path):
    out = tmp_path / "scripts"
    sys.argv = ["installer_models.py", "--mac", str(out)]
    assert installer_models.main() == 0
    assert sorted(p.name for p in out.iterdir()) == ["postinstall", "preinstall"]
    if sys.platform != "win32":  # Windows has no execute bit; the package is built on a Mac
        assert all(p.stat().st_mode & 0o111 for p in out.iterdir())
    make_pkg = (PACKAGING / "macos" / "make_pkg.sh").read_text(encoding="utf-8")
    assert '[ -x "$SCRIPTS/preinstall" ]' in make_pkg and "BundleIsRelocatable -bool false" in make_pkg
    assert "BundleIsVersionChecked -bool false" in make_pkg and "--install-location /Applications" in make_pkg
    assert "does not download the models again" in distribution.welcome_text(2.5)
