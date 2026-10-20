"""The legal-version archive (docs/operations.md, "Legal versions"): the Terms hash in the audit record of an acceptance, the pin it
comes from, the nightly backup's copy of data/legal, and the deploy step that archives the pages (a dry run against a temporary folder)."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from acw import auth, backup, db

from .conftest import ORIGIN, login

ROOT = Path(__file__).resolve().parents[2]
PIN = ROOT / "web" / "legal-versions.json"
HASH = "ab" * 32       # the Terms text with the contact form on
HASH_OFF = "cd" * 32   # ... and with it off


def _audit_details(uid):
    return [r["detail"] for r in db.all_rows("SELECT detail FROM audit_log WHERE user_id = ? AND action = 'tos_accepted' ORDER BY id", (uid,))]


def _pin(tmp_path, monkeypatch, version=None, sha=HASH, sha_off=HASH_OFF):
    f = tmp_path / "pin.json"
    f.write_text(json.dumps({"terms": {"version": version or auth.TOS_VERSION, "sha256": sha, "sha256NoContact": sha_off},
                             "privacy": {"version": "x", "sha256": "y", "sha256NoContact": "z"}}))
    monkeypatch.setenv("ACW_LEGAL_VERSIONS", str(f))


def _contact(monkeypatch, state):
    """The contact form on (True), off (False) or impossible to tell (None), as /api/me would say."""
    from acw import alerts

    def configured():
        if state is None:
            raise RuntimeError("no way to tell")
        return state

    monkeypatch.setattr(alerts, "contact_configured", configured)


def test_the_pin_in_the_repository_is_for_the_terms_version_the_server_records():
    pin = json.loads(PIN.read_text())
    assert pin["terms"]["version"] == auth.TOS_VERSION
    for field in ("sha256", "sha256NoContact"):
        assert re.fullmatch(r"[0-9a-f]{64}", pin["terms"][field])
    assert pin["terms"]["sha256"] != pin["terms"]["sha256NoContact"]
    assert auth.terms_sha256(True) == pin["terms"]["sha256"]  # (found in the checkout: no override set)
    assert auth.terms_sha256(False) == pin["terms"]["sha256NoContact"]


def test_acceptance_at_sign_in_records_version_text_hash_of_the_live_variant_and_how(client, tmp_path, monkeypatch):
    _pin(tmp_path, monkeypatch)
    _contact(monkeypatch, True)
    uid, _ = login(client, accept_tos=False)
    assert _audit_details(uid) == [f"{auth.TOS_VERSION} sha256:{HASH} contact-form:on (sign-in)"]
    _contact(monkeypatch, False)
    uid2, _ = login(client, "second@example.com", accept_tos=False)
    assert _audit_details(uid2) == [f"{auth.TOS_VERSION} sha256:{HASH_OFF} contact-form:off (sign-in)"]


def test_the_contact_form_state_comes_from_the_mail_setup_the_same_way_api_me_has_it(client, monkeypatch):
    # (no mail set up on the test server: the form is off, and the page the visitor got has no contact form)
    assert client.get("/api/me").json()["contactForm"] is False
    assert auth.contact_form_on() is False
    assert auth.tos_detail("x") == f"{auth.TOS_VERSION} sha256:{auth.terms_sha256(False)} contact-form:off (x)"


def test_when_the_contact_form_state_is_unknown_the_on_hash_is_recorded_with_a_note(client, tmp_path, monkeypatch):
    _pin(tmp_path, monkeypatch)
    _contact(monkeypatch, None)
    assert auth.tos_detail("sign-in") == f"{auth.TOS_VERSION} sha256:{HASH} contact-form:unknown (sign-in)"


def test_acceptance_through_the_notice_and_by_continued_use_record_the_hash_too(client, tmp_path, monkeypatch):
    _pin(tmp_path, monkeypatch)
    _contact(monkeypatch, True)
    uid, h = login(client, accept_tos=False)
    db.execute("UPDATE users SET tos_version = 'old' WHERE id = ?", (uid,))
    assert client.post("/api/account/tos", json={"version": auth.TOS_VERSION}, headers=h).status_code == 200
    db.execute("UPDATE users SET tos_version = 'older' WHERE id = ?", (uid,))
    auth.accept_terms(uid, "continued use", "203.0.113.5")
    assert _audit_details(uid) == [
        f"{auth.TOS_VERSION} sha256:{HASH} contact-form:on (sign-in)",
        f"{auth.TOS_VERSION} sha256:{HASH} contact-form:on (notice)",
        f"{auth.TOS_VERSION} sha256:{HASH} contact-form:on (continued use)",
    ]


def test_a_missing_stale_or_broken_pin_records_the_version_alone(client, tmp_path, monkeypatch, caplog):
    uid, _ = login(client, accept_tos=False)
    _contact(monkeypatch, True)
    monkeypatch.setenv("ACW_LEGAL_VERSIONS", str(tmp_path / "nothing-here.json"))
    monkeypatch.setattr(auth.Path, "is_file", lambda self: False)  # no pin anywhere
    assert auth.tos_detail("sign-in") == f"{auth.TOS_VERSION} (sign-in)"
    monkeypatch.undo()
    _contact(monkeypatch, True)
    _pin(tmp_path, monkeypatch, version="1999-01-01")  # a pin for other text is never recorded as this one's hash
    assert auth.tos_detail() == auth.TOS_VERSION and "pins Terms version 1999-01-01" in caplog.text
    (tmp_path / "pin.json").write_text("not json")
    assert auth.tos_detail("x") == f"{auth.TOS_VERSION} (x)"
    _pin(tmp_path, monkeypatch, sha="nothex")
    assert auth.terms_sha256() is None
    # a pin that has only the "on" text (an older pin): the "off" variant is never replaced by it
    (tmp_path / "pin.json").write_text(json.dumps({"terms": {"version": auth.TOS_VERSION, "sha256": HASH}}))
    assert auth.terms_sha256(True) == HASH and auth.terms_sha256(False) is None
    assert uid  # (the sign-in itself was never in doubt)


# ------------------------------------------------------------------ the nightly backup copies data/legal

def test_the_backup_copies_the_archive_next_to_the_snapshots_and_never_overwrites(client, tmp_path):
    legal = db.get_settings().data_dir / "legal"
    legal.mkdir(parents=True, exist_ok=True)
    (legal / "terms-2026-10-20.html").write_text("<p>terms</p>")
    (legal / "terms-2026-10-20.html.sha256").write_text(HASH + "\n")
    (legal / "terms-2026-10-20-no-contact.html").write_text("<p>terms, no contact form</p>")
    (legal / "terms-2026-10-20-no-contact.html.sha256").write_text(HASH_OFF + "\n")
    (legal / "terms-2026-11-01.html.part").write_text("half")  # a file being written by the deploy: left for the next night
    out = tmp_path / "backups"
    snapshot = backup.run(out, keep=5)
    assert snapshot.name.endswith(".sqlite3.gz")
    assert sorted(p.name for p in (out / "legal").iterdir()) == [
        "terms-2026-10-20-no-contact.html", "terms-2026-10-20-no-contact.html.sha256", "terms-2026-10-20.html", "terms-2026-10-20.html.sha256"]
    assert json.loads((out / backup.STATUS_FILE).read_text())["legal"] == 4
    assert oct((out / "legal" / "terms-2026-10-20.html").stat().st_mode & 0o777) == "0o600"
    backup.run(out, keep=5)  # the second night: nothing new, nothing changed
    assert json.loads((out / backup.STATUS_FILE).read_text())["legal"] == 0
    (legal / "privacy-2026-10-20.html").write_text("<p>privacy</p>")
    backup.run(out, keep=5)
    assert (out / "legal" / "privacy-2026-10-20.html").read_text() == "<p>privacy</p>"
    # an archived file that no longer matches its copy is reported (the backup fails, the owner is mailed), the copy is kept as it was
    (legal / "terms-2026-10-20.html").write_text("<p>terms, edited</p>")
    with pytest.raises(backup.BackupError, match="terms-2026-10-20.html"):
        backup.run(out, keep=5)
    assert (out / "legal" / "terms-2026-10-20.html").read_text() == "<p>terms</p>"
    assert "legal archive" in json.loads((out / backup.STATUS_FILE).read_text())["error"]


def test_the_backup_without_an_archive_is_unchanged(client, tmp_path):
    out = tmp_path / "backups"
    backup.run(out, keep=5)
    assert not (out / "legal").exists()
    assert json.loads((out / backup.STATUS_FILE).read_text())["legal"] == 0


# ------------------------------------------------------------------ deploy/legal-archive.sh against a temporary folder

NODE = shutil.which("node")
needs_web = pytest.mark.skipif(not NODE or not (ROOT / "web" / "node_modules" / "react-dom").is_dir(), reason="needs node and web/node_modules")


def _archive(*args):
    env = {**os.environ, "ACW_SERVICE_USER": "", "ACW_DELEGATE": "0"}
    return subprocess.run(["bash", str(ROOT / "deploy" / "legal-archive.sh"), *args], capture_output=True, text=True, env=env, timeout=120)


@needs_web
def test_deploy_archive_saves_each_version_once_and_stops_when_a_text_changed_under_the_same_date(tmp_path):
    pin = json.loads(PIN.read_text())
    dest = tmp_path / "data" / "legal"
    check = _archive("--dir", str(dest), "--check")
    assert check.returncode == 0, check.stderr
    assert not dest.exists()  # --check writes nothing

    first = _archive("--dir", str(dest))
    assert first.returncode == 0, first.stderr
    names = sorted(p.name for p in dest.iterdir())
    tv, pv = pin["terms"]["version"], pin["privacy"]["version"]
    t, tn, p, pn = f"terms-{tv}.html", f"terms-{tv}-no-contact.html", f"privacy-{pv}.html", f"privacy-{pv}-no-contact.html"
    assert names == sorted(f for n in (t, tn, p, pn) for f in (n, n + ".sha256"))  # both variants of both pages
    for name, doc, field in ((t, "terms", "sha256"), (tn, "terms", "sha256NoContact"), (p, "privacy", "sha256"), (pn, "privacy", "sha256NoContact")):
        assert (dest / (name + ".sha256")).read_text() == pin[doc][field] + "\n"
        assert oct((dest / name).stat().st_mode & 0o777) == "0o600"
    page = (dest / t).read_text()
    assert f'name="legal-version" content="{tv}"' in page and "Terms of Service" in page and "contact form on" in page
    assert 'href="/contact"' in page
    assert "contact form off" in (dest / tn).read_text() and 'href="/contact"' not in (dest / tn).read_text()

    # the file can be checked on its own, without React
    h = subprocess.run([NODE, str(ROOT / "web" / "scripts" / "legal-snapshot.mjs"), "--hash-of", str(dest / t)], capture_output=True, text=True)
    assert h.stdout.strip() == pin["terms"]["sha256"]
    h = subprocess.run([NODE, str(ROOT / "web" / "scripts" / "legal-snapshot.mjs"), "--hash-of", str(dest / tn)], capture_output=True, text=True)
    assert h.stdout.strip() == pin["terms"]["sha256NoContact"]

    # same text again: nothing to do, the saved file (with its render date) is not touched
    before = {n: (dest / n).read_bytes() for n in names}
    again = _archive("--dir", str(dest))
    assert again.returncode == 0 and "nothing to do" in again.stdout and "saved" not in again.stdout
    assert {n: (dest / n).read_bytes() for n in names} == before

    # a half-saved pair (the hash file lost) is completed from the page's own text
    (dest / (t + ".sha256")).unlink()
    assert _archive("--dir", str(dest)).returncode == 0
    assert (dest / (t + ".sha256")).read_text() == pin["terms"]["sha256"] + "\n"

    # the archive says something else for this version: stop, say why, change nothing
    (dest / (t + ".sha256")).write_text("0" * 64 + "\n")
    stopped = _archive("--dir", str(dest))
    assert stopped.returncode == 1
    assert "STOP" in stopped.stderr and "never overwritten" in stopped.stderr and "version date" in stopped.stderr and t in stopped.stderr
    assert (dest / t).read_bytes() == before[t] and (dest / (t + ".sha256")).read_text() == "0" * 64 + "\n"
    assert _archive("--dir", str(dest), "--check").returncode == 1  # the pre-flight check in install.sh says the same

    # the same for the other variant: the contact-off page of this version differs from what is archived
    (dest / (t + ".sha256")).write_text(pin["terms"]["sha256"] + "\n")
    (dest / (pn + ".sha256")).write_text("f" * 64 + "\n")
    stopped = _archive("--dir", str(dest))
    assert stopped.returncode == 1 and "STOP" in stopped.stderr and pn in stopped.stderr and "privacy-no-contact" in stopped.stderr


@needs_web
def test_deploy_archive_stops_before_anything_when_the_pin_is_stale(tmp_path, monkeypatch):
    stale = json.loads(PIN.read_text())
    stale["terms"]["sha256"] = "1" * 64
    f = tmp_path / "pin.json"
    f.write_text(json.dumps(stale))
    monkeypatch.setenv("ACW_LEGAL_PIN", str(f))  # (web/scripts/legal-snapshot.mjs reads the pin from here instead of web/legal-versions.json)
    r = _archive("--dir", str(tmp_path / "legal"))
    assert r.returncode == 1 and "the text changed" in r.stderr and "legal:pin" in r.stderr
    assert not (tmp_path / "legal").exists()
