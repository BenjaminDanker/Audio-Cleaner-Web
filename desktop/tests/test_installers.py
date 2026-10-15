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

from audio_cleaner_desktop import resources  # noqa: E402


def test_the_catalog_is_read_without_numpy():
    code = "import sys; sys.modules['numpy'] = None; sys.path[:0] = [r'%s', r'%s']; import installer_models; installer_models.render()"
    r = subprocess.run(
        [sys.executable, "-I", "-c", code % (PACKAGING, PACKAGING.parents[1] / "common")], capture_output=True, text=True,
        cwd=PACKAGING,
    )
    assert r.returncode == 0, r.stderr


def test_windows_include_budgets_space_cleans_up_and_tries_every_file():
    text = installer_models.render()
    files = [m for m in installer_models.CATALOG for _ in m.files]
    assert text.count("ExternalSize:") == len(files) and text.count("deleteafterinstall") == len(files)
    assert text.count("    FetchModel(") == len(files)  # one download each: a failure does not stop the next
    assert "GetSpaceOnDisk64(ExpandConstant('{tmp}')" in text
    assert "AbortedByUser" in text
    assert "Open Audio Cleaner > Settings to finish the download" in text
    assert "when it starts" not in text
    assert "SaveStringToFile" in text and ".complete" in text
    for m in installer_models.CATALOG:
        for f in m.files:
            assert f.url in text and f.sha256 in text and str(f.size) in text
    assert "[UninstallDelete]" in text


def test_the_pascal_strings_are_balanced():
    for line in installer_models.render().splitlines():
        if line.lstrip().startswith(("FetchModel", "if ModelFileReady", "SaveStringToFile")):
            assert line.replace("''", "").count("'") % 2 == 0, line


def test_mac_postinstall_downloads_safely(tmp_path):
    text = installer_models.render_mac()
    for flag in ("--proto '=https'", "--speed-limit 1000", "--speed-time 60", "--max-time 3600", "-C -"):
        assert flag in text
    assert text.count("\nfetch \"") == sum(len(m.files) for m in installer_models.CATALOG)
    script = tmp_path / "postinstall"
    script.write_text(text, encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_mac_postinstall_writes_the_marker_only_for_complete_models(tmp_path):
    """Run the script's `finish` against a pretend model folder (GNU stat stands in for BSD stat)."""
    text = installer_models.render_mac().replace(installer_models.MAC_MODELS, str(tmp_path / "models"))
    lines = [ln for ln in text.splitlines() if not ln.startswith("fetch ")]  # no downloads
    prelude = 'stat() { command stat -c%s "$2"; }\n'
    script = tmp_path / "run.sh"
    script.write_text(prelude + "\n".join(ln for ln in lines if not ln.startswith("chmod -R")), encoding="utf-8")
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
    sys.argv = ["distribution.py", str(out), "--version", "2.0.0", "--arch", "arm64", "--eula", str(eula), "--models-gb", "2.6"]
    assert distribution.main() == 0
    welcome = (out / "welcome.txt").read_text()
    assert "2.6 GB" in welcome and "several minutes" in welcome
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


def test_every_catalog_file_is_checked():
    assert len(check_model_urls.catalog_files()) == sum(
        len(m.files) for m in (resources.SPEECH_MODEL, resources.HIGHS_MODEL, resources.VOICE_MODEL)
    )
