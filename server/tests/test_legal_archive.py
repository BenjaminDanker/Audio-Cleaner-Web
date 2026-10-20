"""The versions of the Terms and the Privacy Policy are decided by the deploy (docs/operations.md, "Legal versions"): the server reads the live
Terms version and text hash from data/legal/current.json for the "We've updated our Terms" notice and the audit record of an acceptance, and
deploy/legal-archive.sh versions and archives new text by itself (run here against a temporary folder)."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from acw import auth, db

from .conftest import login

ROOT = Path(__file__).resolve().parents[2]
HASH = "ab" * 32


def _audit_details(uid):
    return [r["detail"] for r in db.all_rows("SELECT detail FROM audit_log WHERE user_id = ? AND action = 'tos_accepted' ORDER BY id", (uid,))]


def _current(version="2026-10-20", sha=HASH):
    """What the deploy wrote: data/legal/current.json in the (per-test) data directory."""
    f = db.get_settings().data_dir / "legal" / "current.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"terms": {"version": version, "sha256": sha}, "privacy": {"version": "2026-09-01", "sha256": "cd" * 32}}))
    return f


# ------------------------------------------------------------------ the server reads the live version from current.json

def test_without_an_archive_the_version_is_dev_and_nothing_blocks(client):
    assert auth.tos_version() == "dev"
    uid, _ = login(client, accept_tos=False)  # signing in records the agreement as ever
    assert _audit_details(uid) == ["dev (sign-in)"]
    me = client.get("/api/me").json()["user"]
    assert me["tosVersion"] == "dev" and me["tosAccepted"] is True


def test_acceptance_records_the_live_version_and_the_hash_of_its_text_for_each_way_in(client):
    _current()
    uid, h = login(client, accept_tos=False)
    me = client.get("/api/me").json()["user"]
    assert (me["tosVersion"], me["tosAccepted"]) == ("2026-10-20", True)
    db.execute("UPDATE users SET tos_version = 'old' WHERE id = ?", (uid,))
    assert client.post("/api/account/tos", json={"version": "2026-10-20"}, headers=h).status_code == 200
    db.execute("UPDATE users SET tos_version = 'older' WHERE id = ?", (uid,))
    auth.accept_terms(uid, "continued use", "203.0.113.5")
    assert _audit_details(uid) == [
        f"2026-10-20 sha256:{HASH} (sign-in)",
        f"2026-10-20 sha256:{HASH} (notice)",
        f"2026-10-20 sha256:{HASH} (continued use)",
    ]


def test_a_deploy_with_new_text_shows_the_notice_without_restarting_the_server(client):
    f = _current("2026-10-20")
    uid, h = login(client)
    assert client.get("/api/me").json()["user"]["tosAccepted"] is True
    f.write_text(json.dumps({"terms": {"version": "2026-11-02-2", "sha256": "cd" * 32}}))  # (the deploy wrote new text's version)
    me = client.get("/api/me").json()["user"]
    assert (me["tosVersion"], me["tosAccepted"]) == ("2026-11-02-2", False)  # the "We've updated our Terms" notice
    assert client.post("/api/account/tos", json={"version": "2026-10-20"}, headers=h).status_code == 409  # a tab with the old page
    assert client.post("/api/account/tos", json={"version": "2026-11-02-2"}, headers=h).status_code == 200
    assert client.get("/api/me").json()["user"]["tosAccepted"] is True
    assert _audit_details(uid)[-1] == f"2026-11-02-2 sha256:{'cd' * 32} (notice)"


@pytest.mark.parametrize("content", ["not json", "{}", '{"terms": {"version": "x y", "sha256": "' + "ab" * 32 + '"}}',
                                     '{"terms": {"version": "2026-10-20", "sha256": "nothex"}}', '{"terms": []}'])
def test_a_broken_current_json_falls_back_to_dev_with_a_warning(client, caplog, content):
    f = _current()
    f.write_text(content)
    assert auth.tos_version() == "dev" and auth.tos_detail("x") == "dev (x)"
    assert "cannot use" in caplog.text


# ------------------------------------------------------------------ deploy/legal-archive.sh against a temporary folder

NODE = shutil.which("node")
needs_web = pytest.mark.skipif(not NODE or not (ROOT / "web" / "node_modules" / "react-dom").is_dir(), reason="needs node and web/node_modules")


def _archive(dest, date="2026-10-20", *args):
    env = {**os.environ, "ACW_SERVICE_USER": "", "ACW_DELEGATE": "0", "ACW_LEGAL_DATE": date}
    return subprocess.run(["bash", str(ROOT / "deploy" / "legal-archive.sh"), "--dir", str(dest), *args], capture_output=True, text=True, env=env, timeout=120)


def _versions(run):
    assert run.returncode == 0, run.stderr
    return dict(line.split("=") for line in run.stdout.split())


def _snapshot(dest):
    return {p.name: p.read_bytes() for p in dest.iterdir()}


def _hash_of(page):
    return subprocess.run([NODE, str(ROOT / "web" / "scripts" / "legal-snapshot.mjs"), "--hash-of", str(page)], capture_output=True, text=True).stdout.strip()


@needs_web
def test_the_first_deploy_archives_todays_text_as_the_first_version_and_current_json_follows(tmp_path, client):
    dest = db.get_settings().data_dir / "legal"
    dry = _archive(tmp_path / "dry", "2026-10-20", "--check")
    assert _versions(dry) == {"ACW_LEGAL_TERMS_VERSION": "2026-10-20", "ACW_LEGAL_PRIVACY_VERSION": "2026-10-20"}
    assert not (tmp_path / "dry").exists()  # --check writes nothing

    first = _archive(dest)
    assert _versions(first) == {"ACW_LEGAL_TERMS_VERSION": "2026-10-20", "ACW_LEGAL_PRIVACY_VERSION": "2026-10-20"}  # (stdout: what the web build takes)
    assert sorted(p.name for p in dest.iterdir()) == [
        "current.json", "privacy-2026-10-20.html", "privacy-2026-10-20.html.sha256", "terms-2026-10-20.html", "terms-2026-10-20.html.sha256"]
    current = json.loads((dest / "current.json").read_text())
    for doc in ("terms", "privacy"):
        page = dest / f"{doc}-2026-10-20.html"
        assert current[doc] == {"version": "2026-10-20", "sha256": (dest / f"{page.name}.sha256").read_text().strip()}
        assert re.fullmatch(r"[0-9a-f]{64}", current[doc]["sha256"])
        assert oct(page.stat().st_mode & 0o777) == "0o600" and oct((dest / "current.json").stat().st_mode & 0o777) == "0o600"
        assert _hash_of(page) == current[doc]["sha256"]  # the page can be checked on its own, without React
    page = (dest / "terms-2026-10-20.html").read_text()
    assert 'name="legal-version" content="2026-10-20"' in page and "Terms of Service" in page and "Version 2026-10-20." in page
    assert 'href="/contact"' in page  # the contact form is always in the text

    # the server reads it: the live version and the hash that acceptances record
    assert auth.tos_version() == "2026-10-20"
    assert auth.tos_detail("sign-in") == f"2026-10-20 sha256:{current['terms']['sha256']} (sign-in)"


@needs_web
def test_the_same_text_keeps_its_version_whatever_the_date_and_nothing_is_rewritten(tmp_path):
    dest = tmp_path / "data" / "legal"
    _archive(dest, "2026-10-20")
    before = _snapshot(dest)
    later = _archive(dest, "2026-12-24")
    assert _versions(later) == {"ACW_LEGAL_TERMS_VERSION": "2026-10-20", "ACW_LEGAL_PRIVACY_VERSION": "2026-10-20"}
    assert "saved" not in later.stderr and "wrote" not in later.stderr and "nothing to save" in later.stderr
    assert _snapshot(dest) == before
    # a page whose hash file was lost is still found by its own text
    (dest / "terms-2026-10-20.html.sha256").unlink()
    again = _archive(dest, "2027-01-01")
    assert _versions(again)["ACW_LEGAL_TERMS_VERSION"] == "2026-10-20" and "no hash file" in again.stderr
    assert not (dest / "terms-2027-01-01.html").exists()


@needs_web
def test_new_text_gets_a_new_dated_version_a_second_change_the_same_day_gets_a_number_and_nothing_is_overwritten(tmp_path):
    dest = tmp_path / "data" / "legal"
    _archive(dest, "2026-10-20")
    page_v1 = (dest / "terms-2026-10-20.html").read_bytes()
    privacy = (dest / "privacy-2026-10-20.html").read_bytes()
    # the Terms are now different from what is archived as 2026-10-20 (the archive says another hash for that version)
    (dest / "terms-2026-10-20.html.sha256").write_text("1" * 64 + "\n")
    next_day = _archive(dest, "2026-10-21")
    assert _versions(next_day) == {"ACW_LEGAL_TERMS_VERSION": "2026-10-21", "ACW_LEGAL_PRIVACY_VERSION": "2026-10-20"}  # only the changed page
    assert (dest / "terms-2026-10-21.html").is_file() and not (dest / "privacy-2026-10-21.html").exists()
    assert json.loads((dest / "current.json").read_text())["terms"]["version"] == "2026-10-21"
    assert (dest / "terms-2026-10-20.html").read_bytes() == page_v1 and (dest / "privacy-2026-10-20.html").read_bytes() == privacy  # never overwritten
    assert (dest / "terms-2026-10-20.html.sha256").read_text() == "1" * 64 + "\n"

    # changed again the same day: -2, then -3
    (dest / "terms-2026-10-21.html.sha256").write_text("2" * 64 + "\n")
    second = _archive(dest, "2026-10-21")
    assert _versions(second)["ACW_LEGAL_TERMS_VERSION"] == "2026-10-21-2"
    (dest / "terms-2026-10-21-2.html.sha256").write_text("3" * 64 + "\n")
    third = _archive(dest, "2026-10-21")
    assert _versions(third)["ACW_LEGAL_TERMS_VERSION"] == "2026-10-21-3"
    assert (dest / "terms-2026-10-21-3.html").is_file()
    assert json.loads((dest / "current.json").read_text())["terms"]["version"] == "2026-10-21-3"
    assert (dest / "terms-2026-10-21.html.sha256").read_text() == "2" * 64 + "\n"  # (the older versions are as they were)

    # run again with nothing changed: stays -3
    assert _versions(_archive(dest, "2026-10-21"))["ACW_LEGAL_TERMS_VERSION"] == "2026-10-21-3"


@needs_web
def test_a_name_that_is_taken_stops_the_deploy_before_anything_is_written(tmp_path):
    dest = tmp_path / "data" / "legal"
    _archive(dest, "2026-10-20")
    (dest / "terms-2026-10-20.html.sha256").write_text("1" * 64 + "\n")  # the Terms changed: they would become 2026-10-20-2 ...
    (dest / "terms-2026-10-20-2.html.sha256").write_text("2" * 64 + "\n")  # ... but a file of that name is there (its page lost)
    current = (dest / "current.json").read_bytes()
    before = _snapshot(dest)
    stopped = _archive(dest, "2026-10-20")
    assert stopped.returncode == 1 and "STOP" in stopped.stderr and "terms-2026-10-20-2.html" in stopped.stderr and "never overwritten" in stopped.stderr
    assert _snapshot(dest) == before and (dest / "current.json").read_bytes() == current
