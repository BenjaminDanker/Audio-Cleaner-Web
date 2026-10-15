"""Database backup: `python -m acw.admin backup DIR [--keep N]`.

Every snapshot is checked after it is written: it is unzipped to a scratch file, SQLite's own integrity check
must pass, and the money must add up (each account's balance equals the sum of its ledger entries, and what is
held equals the sum of its active holds). A snapshot that fails is moved aside as `*.bad`, the command exits
with an error (so the unit fails and the owner is mailed) and the result is written to `backup-status.json`
for the monitor.

Optional copy off this machine: set ACW_BACKUP_OFFBOX to an rsync destination (`user@host:/path/` or a local
directory), plus either ACW_BACKUP_AGE_RECIPIENT (an age public key) or ACW_BACKUP_PASSPHRASE_FILE (a file with
a passphrase for `gpg --symmetric`). The file is encrypted before it leaves; nothing happens when the variable
is unset. Settings go in ~/audio-cleaner/secrets/backup.env (read by the backup unit).
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path

from . import db

STATUS_FILE = "backup-status.json"
KEEP_BAD = 3


class BackupError(Exception):
    pass


def check_snapshot(db_file: Path) -> dict:
    """Integrity and ledger checks on an (uncompressed) snapshot. Returns {"integrity": ..., "ledger": ...}."""
    c = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    try:
        integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
        bad_balance = c.execute(
            "SELECT COUNT(*) FROM users u WHERE u.balance_micros != COALESCE((SELECT SUM(amount_micros) FROM ledger l WHERE l.user_id = u.id), 0)"
        ).fetchone()[0]
        bad_held = c.execute(
            "SELECT COUNT(*) FROM users u WHERE u.held_micros != COALESCE((SELECT SUM(amount_micros) FROM holds h"
            " WHERE h.user_id = u.id AND h.status = 'active'), 0)"
        ).fetchone()[0]
    finally:
        c.close()
    ledger = "ok" if not bad_balance and not bad_held else f"{bad_balance} balances and {bad_held} holds do not match"
    return {"integrity": integrity, "ledger": ledger}


def verify(gz: Path) -> dict:
    """Unzip `gz` to a scratch file and check it."""
    with tempfile.TemporaryDirectory(dir=gz.parent, prefix=".verify-") as tmp:
        out = Path(tmp) / "check.sqlite3"
        with gzip.open(gz, "rb") as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return check_snapshot(out)


def _write_status(out_dir: Path, status: dict) -> None:
    tmp = out_dir / (STATUS_FILE + ".tmp")
    tmp.write_text(json.dumps(status))
    os.chmod(tmp, 0o600)
    tmp.replace(out_dir / STATUS_FILE)


def encrypt(src: Path, dst: Path) -> None:
    recipient = os.environ.get("ACW_BACKUP_AGE_RECIPIENT")
    passfile = os.environ.get("ACW_BACKUP_PASSPHRASE_FILE")
    if recipient:
        cmd = ["age", "-r", recipient, "-o", str(dst), str(src)]
    elif passfile:
        if not Path(passfile).is_file():
            raise BackupError(f"ACW_BACKUP_PASSPHRASE_FILE {passfile} not found")
        cmd = ["gpg", "--batch", "--yes", "--pinentry-mode", "loopback", "--passphrase-file", passfile,
               "--symmetric", "--cipher-algo", "AES256", "-o", str(dst), str(src)]
    else:
        raise BackupError("ACW_BACKUP_OFFBOX is set, but neither ACW_BACKUP_AGE_RECIPIENT nor ACW_BACKUP_PASSPHRASE_FILE is")
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode != 0:
        raise BackupError(f"encrypting the backup failed: {r.stderr.decode(errors='replace')[-300:]}")


def copy_offbox(gz: Path, dest: str) -> str:
    """Encrypt `gz` and rsync it to `dest`. Returns the name of the copy."""
    enc = gz.with_name(gz.name + (".age" if os.environ.get("ACW_BACKUP_AGE_RECIPIENT") else ".gpg"))
    try:
        encrypt(gz, enc)
        r = subprocess.run(
            ["rsync", "-t", "--chmod=F600", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20", str(enc), dest],
            capture_output=True, timeout=1800,
        )
        if r.returncode != 0:
            raise BackupError(f"copying the backup off this machine failed: {r.stderr.decode(errors='replace')[-300:]}")
    finally:
        enc.unlink(missing_ok=True)
    return enc.name


def _prune(out_dir: Path, keep: int) -> None:
    """Old snapshots go, also when tonight's failed (for instance the off-machine copy): a backup that fails every night
    must not fill the disk. Failed snapshots (*.bad) are kept for a look, but only the newest few."""
    backups = sorted(out_dir.glob("acw-*.sqlite3.gz"))
    for old in backups[: max(0, len(backups) - keep)]:
        old.unlink(missing_ok=True)
    for old in sorted(out_dir.glob("acw-*.sqlite3.gz.bad"))[:-KEEP_BAD]:
        old.unlink(missing_ok=True)


def run(out_dir: Path, keep: int = 30) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    tmp = out_dir / f".acw-{stamp}.sqlite3"
    dst = sqlite3.connect(str(tmp))
    db.conn().backup(dst)  # online, consistent snapshot
    dst.close()
    final = out_dir / f"acw-{stamp}.sqlite3.gz"
    with open(tmp, "rb") as src, gzip.open(final, "wb", compresslevel=6) as gz:
        shutil.copyfileobj(src, gz)
    os.chmod(final, 0o600)
    tmp.unlink()
    status = {"file": final.name, "bytes": final.stat().st_size, "at": int(time.time()), "offbox": None}
    try:
        if status["bytes"] <= 0:
            raise BackupError("the snapshot is empty")
        result = verify(final)
        status.update(result)
        if result["integrity"] != "ok" or result["ledger"] != "ok":
            raise BackupError(f"snapshot failed its checks: {result}")
        dest = os.environ.get("ACW_BACKUP_OFFBOX")
        if dest:
            status["offbox"] = copy_offbox(final, dest)
    except BackupError as e:
        status["error"] = str(e)
        if status.get("integrity") != "ok" or status.get("ledger") != "ok":
            final.rename(final.with_name(final.name + ".bad"))  # kept for a look, never counted as a good backup
        _prune(out_dir, keep)
        _write_status(out_dir, status)
        raise
    _write_status(out_dir, status)
    _prune(out_dir, keep)
    return final
