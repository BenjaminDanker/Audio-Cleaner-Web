"""What the installers are made of: the Inno Setup include and the Mac postinstall generated from the model
catalog, the Mac distribution, and the catalog URL check. None of this needs Windows or a Mac."""
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

import check_model_urls  # noqa: E402
import distribution  # noqa: E402
import installer_models  # noqa: E402

from audio_cleaner_desktop import models, resources  # noqa: E402


def test_the_catalog_is_read_without_numpy():
    code = "import sys; sys.modules['numpy'] = None; sys.path[:0] = [r'%s', r'%s']; import installer_models; installer_models.render()"
    r = subprocess.run(
        [sys.executable, "-I", "-c", code % (PACKAGING, PACKAGING.parents[1] / "common")], capture_output=True, text=True,
        cwd=PACKAGING,
    )
    assert r.returncode == 0, r.stderr


def test_windows_include_budgets_space_cleans_up_and_tries_every_download():
    text = installer_models.render()
    plain = [m for m in installer_models.CATALOG if m.archive is None for _ in m.files]  # the speech model's files
    assert plain and text.count("ExternalSize:") == len(plain) and text.count("deleteafterinstall") == len(plain)
    assert text.count("    FetchModel(") == len(plain)  # one download each: a failure does not stop the next
    assert text.count("    FetchArchive;") == 1  # and the archive is one download
    assert "GetSpaceOnDisk64(ExpandConstant('{tmp}')" in text
    assert "AbortedByUser" in text
    assert "Open Audio Cleaner > Settings to finish the download" in text
    assert "when it starts" not in text
    assert "SaveStringToFile" in text and ".complete" in text
    for m in installer_models.CATALOG:
        for f in m.files:
            assert f.sha256 in text and str(f.size) in text
            if m.archive is None:
                assert f.url in text
    assert "[UninstallDelete]" in text


def test_windows_include_fetches_the_archive_once_and_unpacks_it_with_the_builtin_tar():
    text = installer_models.render()
    archive = resources.MODELS_ARCHIVE
    assert text.count(archive.url) == 2 and archive.sha256 in text  # the download and the log line
    assert f"ModelDownloads.Add('{archive.url}', 'acw-models.zip', '{archive.sha256}')" in text  # sha256 verified by Inno
    assert "'{win}\\System32\\tar.exe'" in text and "'-xf \"' + Zip + '\" -C \"' + Work + '\"'" in text
    assert text.index("    FetchArchive;") < text.index("    FetchModel(")  # the archive is not tried when tar.exe is missing...
    assert "tar.exe is not available" in text and "ModelsFailed := ModelsFailed + 1" in text  # ...the Settings message appears
    assert "if ArchiveNeeded then Result := Result + %d" % archive.size in text  # free-space check counts the archive
    unpacked = sum(f.size for m in (resources.HIGHS_MODEL, resources.VOICE_MODEL) for f in m.files) + archive.unpacked_size
    assert f"ExtraDiskSpaceRequired={unpacked}" in text
    for m in (resources.HIGHS_MODEL, resources.VOICE_MODEL):
        for f in m.files:  # size-checked move to the model's own folder
            src = "Work + '\\" + f.member.replace("/", "\\") + "'"
            assert f"PlaceModelFile({src}, ExpandConstant('{{app}}\\models\\{m.id}\\{f.path}'), {f.size});" in text
    for f in archive.shared:  # the licence texts land next to the models
        assert f"ExpandConstant('{{app}}\\models\\{f.path.replace('/', chr(92))}'), {f.size});" in text
    assert "ModelFileReady(Src, Size)" in text and "RenameFile(Src, Dest)" in text
    assert "DeleteFile(Zip)" in text and "DelTree(Work" in text  # the archive and the work folder are removed
    assert text.index("  UnpackModels;") < text.index("SaveStringToFile")  # markers only after unpacking
    assert "ssPostInstall" in text
    # one archive download, no individual download for the ONNX models
    assert text.count("ap-bwe/") == 0 and "FetchModel('" + archive.url not in text


def test_the_total_is_the_speech_files_plus_the_archive_once():
    speech = sum(f.size for f in resources.SPEECH_MODEL.files)
    assert installer_models.total_bytes() == speech + resources.MODELS_ARCHIVE.size
    assert 2.4e9 < installer_models.total_bytes() < 2.6e9


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
    assert sum(w in ("begin", "try", "case") for w in words) == words.count("end") > 8
    defined = [words[i + 1] for i, w in enumerate(words) if w in ("procedure", "function")]
    assert len(defined) >= 9
    for name in defined:  # Pascal needs a routine to be defined before its first use
        first = words.index(name)
        assert words[first - 1] in ("procedure", "function"), name


def test_mac_postinstall_downloads_safely(tmp_path):
    text = installer_models.render_mac()
    for flag in ("--proto '=https'", "--speed-limit 1000", "--speed-time 60", "--max-time 3600", "-C -"):
        assert flag in text
    plain = sum(len(m.files) for m in installer_models.CATALOG if m.archive is None)
    assert text.count("\nfetch \"") == plain  # the speech model's files
    archive = resources.MODELS_ARCHIVE
    assert text.count(archive.url) == 2 and archive.sha256 in text  # once downloaded (and named in the log line)
    assert 'ditto -x -k "$ARCHIVE" "$WORK"' in text and 'shasum -a 256 "$ARCHIVE"' in text
    assert 'rm -rf "$WORK" "$ARCHIVE"' in text
    script = tmp_path / "postinstall"
    script.write_text(text, encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_mac_postinstall_writes_the_marker_only_for_complete_models(tmp_path):
    """Run the script's `finish` against a pretend model folder (GNU stat stands in for BSD stat)."""
    text = installer_models.render_mac().replace(installer_models.MAC_MODELS, str(tmp_path / "models"))
    lines = text.splitlines()
    start = lines.index("need_archive=0")
    end = next(i for i, ln in enumerate(lines) if ln.startswith("finish "))
    lines = [ln for ln in lines[:start] if not ln.startswith("fetch ")] + [ln for ln in lines[end:] if not ln.startswith("chmod -R")]
    prelude = 'stat() { command stat -c%s "$2"; }\n'
    script = tmp_path / "run.sh"
    script.write_text(prelude + "\n".join(lines), encoding="utf-8")
    voice = resources.VOICE_MODEL
    d = tmp_path / "models" / voice.id
    d.mkdir(parents=True)
    for f in voice.files[:1]:
        (d / f.path).write_bytes(b"x" * f.size) if f.size < 10_000_000 else (d / f.path).write_bytes(b"")
    subprocess.run(["bash", str(script)], check=True, capture_output=True)
    assert not (d / ".complete").exists()  # not every file is there
    for f in voice.files:
        with open(d / f.path, "wb") as out:
            out.truncate(f.size)  # sparse file of the right size
    subprocess.run(["bash", str(script)], check=True, capture_output=True)
    assert (d / ".complete").read_text().split() == [f.sha256 for f in voice.files]


def _tiny_catalog(url: str):
    """Two small models and a LICENSES file in a real zip, as the installers' catalog describes the real ones."""
    import hashlib
    import io
    import random
    import zipfile

    rnd = random.Random(7)
    members = {"README.txt": b"hi", "LICENSES/L.txt": b"licence", "m1/a.onnx": rnd.randbytes(50_000), "m2/b.onnx": rnd.randbytes(30_000)}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    blob = buf.getvalue()

    def entry(path, member):
        return models.ModelFile(path, "", len(members[member]), hashlib.sha256(members[member]).hexdigest(), member=member)

    archive = models.ModelArchive("tiny.zip", url, len(blob), hashlib.sha256(blob).hexdigest(), shared=(entry("LICENSES/L.txt", "LICENSES/L.txt"),))
    one = models.SpeechModel("one", "One", "", "MIT", (entry("a.onnx", "m1/a.onnx"),), archive)
    two = models.SpeechModel("two", "Two", "", "MIT", (entry("b.onnx", "m2/b.onnx"),), archive)
    return blob, members, (one, two)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_mac_postinstall_downloads_unpacks_and_cleans_up(tmp_path):
    """Run the generated script for a tiny archive, with stand-ins for curl, shasum, ditto and BSD stat."""
    import sys

    blob, members, catalog = _tiny_catalog("https://example.invalid/tiny.zip")
    (tmp_path / "served.zip").write_bytes(blob)
    root = tmp_path / "models"
    text = installer_models.render_mac(catalog).replace(installer_models.MAC_MODELS, str(root)).replace(
        "/var/log/audio-cleaner-install.log", str(tmp_path / "log.txt"))
    prelude = f"""
stat() {{ command stat -c%s "$2"; }}
shasum() {{ sha256sum "$3"; }}
curl() {{ local out=""; while [ $# -gt 0 ]; do [ "$1" = -o ] && out="$2"; shift; done; cp "{tmp_path}/served.zip" "$out"; }}
ditto() {{ "{sys.executable}" -I -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$3" "$4"; }}
"""
    script = tmp_path / "run.sh"
    script.write_text(prelude + text.replace("#!/bin/bash", ""), encoding="utf-8")
    run = lambda: subprocess.run(["bash", str(script)], check=True, capture_output=True, text=True)  # noqa: E731
    run()
    assert (root / "one" / "a.onnx").read_bytes() == members["m1/a.onnx"]
    assert (root / "two" / "b.onnx").read_bytes() == members["m2/b.onnx"]
    assert (root / "LICENSES" / "L.txt").read_bytes() == b"licence"
    assert (root / "one" / ".complete").read_text().split() == [catalog[0].files[0].sha256]
    assert (root / "two" / ".complete").read_text().split() == [catalog[1].files[0].sha256]
    assert sorted(p.name for p in root.iterdir()) == ["LICENSES", "one", "two"]  # archive and work folder are gone
    (tmp_path / "served.zip").write_bytes(b"y" * len(blob))  # a second run must not need (or trust) a download: nothing is missing
    run()
    assert (root / "one" / "a.onnx").read_bytes() == members["m1/a.onnx"]
    # a download that does not match its SHA-256 installs nothing
    shutil.rmtree(root)
    run()
    assert list(root.iterdir()) == []
    assert "could not download" in (tmp_path / "log.txt").read_text()


def test_the_licences_of_the_archive_are_all_in_the_catalog():
    import zipfile

    local = Path.home() / "audio-cleaner-models-release" / "AudioCleaner-models-v1.zip"
    if not local.is_file():
        pytest.skip("the models archive is not on this computer")
    with zipfile.ZipFile(local) as z:
        names = {i.filename: i.file_size for i in z.infolist()}
    archive = resources.MODELS_ARCHIVE
    assert local.stat().st_size == archive.size
    for m in (resources.HIGHS_MODEL, resources.VOICE_MODEL):
        for f in m.files:
            assert names[f.member] == f.size
    for f in archive.shared:
        assert names[f.member] == f.size


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


def test_the_catalog_check_finds_missing_and_wrong_sized_files():
    answers = {
        "https://a/ok": [(200, {"content-length": "10"})],
        "https://a/hf": [(302, {"x-linked-size": "20", "location": "https://cdn/x"}), (200, {"content-length": "20"})],
        "https://a/gone": [(404, {})],
        "https://a/small": [(302, {"location": "/next"}), (200, {"content-length": "5"})],
    }
    queue = {k: list(v) for k, v in answers.items()}
    queue["https://a/next"] = queue["https://a/small"][1:]
    queue["https://a/small"] = queue["https://a/small"][:1]
    queue["https://cdn/x"] = answers["https://a/hf"][1:]
    queue["https://a/hf"] = queue["https://a/hf"][:1]

    def head(url):
        return queue[url].pop(0)

    found = check_model_urls.problems(
        [("https://a/ok", 10), ("https://a/hf", 20), ("https://a/gone", 1), ("https://a/small", 6)], head
    )
    assert len(found) == 2 and "HTTP 404" in found[0] and "5 bytes there, 6 in the catalog" in found[1]


def test_every_catalog_download_is_checked_the_archive_once():
    found = check_model_urls.catalog_files()
    assert len(found) == len(resources.SPEECH_MODEL.files) + 1
    assert (resources.MODELS_ARCHIVE.url, resources.MODELS_ARCHIVE.size) in found
    assert not [url for url, _ in found if "ap-bwe" in url or "sidon_" in url or "48k.onnx" in url]  # no single files any more
    assert all(url.startswith("https://") and size > 0 for url, size in found)
