"""The move of the production services to the account acw (deploy/host/move-*.sh, the wrappers, deploy/lib.sh's
delegation, install.sh's split into an owner part and a server part): nothing here touches the real services, the
real account, /etc or the real ~/audio-cleaner. Dry runs and moves work on scratch trees; the sudo hop is replaced by
a stand-in (ACW_RUN) or only printed (ACW_RUN_DRY)."""
from __future__ import annotations

import getpass
import grp
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy"
HOST = DEPLOY / "host"
ME = getpass.getuser()
WRAPPERS = ["acw-run", "acw-ctl", "acw-logs", "acw-admin", "acw-sh"]
pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")


def sh(cmd, env=None, cwd=None, check=False, stdin=None):
    e = {**os.environ, **(env or {})}
    try:
        r = subprocess.run(cmd, env=e, cwd=cwd, capture_output=True, text=True, input=stdin)
    except OSError as err:  # e.g. a shebang naming an interpreter that is not there
        r = subprocess.CompletedProcess(cmd, 126, "", str(err))
    if check:
        assert r.returncode == 0, f"{cmd} -> {r.returncode}\n{r.stdout}\n{r.stderr}"
    return r


def tree(path: Path):
    """What is in a folder tree (names, link targets, modes): to show a dry run changed nothing."""
    out = []
    for p in sorted(path.rglob("*")):
        out.append((str(p.relative_to(path)), os.readlink(p) if p.is_symlink() else None, p.lstat().st_mode))
    return out


# ------------------------------------------------------------------ every script parses

def test_all_deploy_scripts_parse():
    scripts = sorted(DEPLOY.glob("*.sh")) + sorted(HOST.glob("*.sh")) + [HOST / w for w in WRAPPERS]
    assert len(scripts) >= 15
    for s in scripts:
        r = sh(["bash", "-n", str(s)])
        assert r.returncode == 0, f"{s}: {r.stderr}"
    for w in WRAPPERS + ["move-to-acw.sh", "move-back-to-silver.sh", "fix-venv-paths.sh"]:
        assert os.access(HOST / w, os.X_OK), f"{w} is not executable"


# ------------------------------------------------------------------ the wrappers (printed, not run)

DRY = {"ACW_RUN_DRY": "1", "ACW_UID": "4242", "ACW_HOME_DIR": "/home/acw", "ACW_USER": "acw", "TERM": "xterm"}


def test_wrappers_run_the_right_thing_as_acw():
    base = "sudo -n -u acw env HOME=/home/acw USER=acw LOGNAME=acw PATH="
    ctl = sh([str(HOST / "acw-ctl"), "status", "acw-*"], DRY).stdout
    assert ctl.startswith(base) and "XDG_RUNTIME_DIR=/run/user/4242" in ctl
    assert "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/4242/bus" in ctl
    assert ctl.split("TERM=xterm ")[1].split() == ["systemctl", "--user", "status", "acw-\\*"]
    one = sh([str(HOST / "acw-logs"), "acw-web", "-n", "5"], DRY).stdout.split("TERM=xterm ")[1].split()
    assert one == ["journalctl", "--user", "-u", "acw-web", "-n", "5"]
    allu = sh([str(HOST / "acw-logs"), "--since", "-1h"], DRY).stdout.split("TERM=xterm ")[1].split()
    assert allu[:4] == ["journalctl", "--user", "-u", "acw-\\*"] and allu[4:] == ["--since", "-1h"]
    admin = sh([str(HOST / "acw-admin"), "activity"], DRY).stdout
    for needle in ("acw-worker.env", "backup.env", "acw.env", "python", "acw-admin", "activity"):
        assert needle in admin
    assert admin.index("acw-worker.env") < admin.index("acw.env")  # the web secrets are loaded last
    worker = sh([str(HOST / "acw-admin"), "--worker-env", "-m", "acw.alerts", "test"], DRY).stdout
    assert " 1 acw.alerts test" in worker.replace("\\ ", " ")  # worker-only flag, module, arguments
    shell = sh([str(HOST / "acw-sh"), "-c", "id"], DRY).stdout
    assert "exec bash -l" in shell.replace("\\ ", " ") and shell.rstrip().endswith("id")
    # as acw itself no sudo is involved; with no command or an unknown account it says so
    me = sh([str(HOST / "acw-run"), "true"], {**DRY, "ACW_USER": ME}).stdout
    assert me.startswith("env HOME=/home/acw") and "sudo" not in me
    assert sh([str(HOST / "acw-run")]).returncode == 2
    r = sh([str(HOST / "acw-run"), "true"], {"ACW_USER": "no-such-account-xyz", "ACW_RUN_DRY": ""})
    assert r.returncode == 1 and "no account" in r.stderr


# ------------------------------------------------------------------ the venv path rewrite

def _make_venv(path: Path, old_prefix: Path):
    venv.create(path, with_pip=False)
    (path / "bin" / "uvicorn").write_text(f"#!{old_prefix}/venv/bin/python\nprint('uvicorn 0.0')\n")
    (path / "bin" / "uvicorn").chmod(0o755)
    (path / "bin" / "activate").write_text(f'VIRTUAL_ENV="{old_prefix}/venv"\n')
    (path / "pyvenv.cfg").write_text((path / "pyvenv.cfg").read_text() + f"command = x -m venv {old_prefix}/venv\n")


def test_venv_paths_are_rewritten_after_the_move(tmp_path):
    old, new = tmp_path / "silver" / "audio-cleaner", tmp_path / "acw" / "audio-cleaner"
    old.mkdir(parents=True)
    _make_venv(old / "venv", old)
    probe = old / "venv" / "bin" / "uvicorn"
    assert sh([str(probe)]).returncode == 0
    new.parent.mkdir()
    shutil.move(str(old), str(new))                                        # what the migration does: a rename
    moved = new / "venv"
    assert sh([str(moved / "bin" / "uvicorn")]).returncode != 0            # the shebang names a folder that is gone
    script = str(HOST / "fix-venv-paths.sh")
    before = tree(new)
    dry = sh([script, "--dry-run", str(old), str(new), str(moved)], check=True)
    assert "would rewrite" in dry.stdout and tree(new) == before           # a dry run changes nothing
    r = sh([script, str(old), str(new), str(moved)], check=True)
    assert all(f"/{name}\n" in r.stdout or f"/{name}" in r.stdout for name in ("uvicorn", "activate", "pyvenv.cfg"))
    assert r.stdout.count("rewrote ") >= 3
    out = sh([str(moved / "bin" / "uvicorn")], check=True).stdout
    assert out.strip() == "uvicorn 0.0"
    assert str(old) not in (moved / "bin" / "activate").read_text() + (moved / "pyvenv.cfg").read_text()
    assert os.access(moved / "bin" / "uvicorn", os.X_OK)                   # the mode stays
    assert "0 file(s)" in sh([script, str(old), str(new), str(moved)], check=True).stdout   # again: nothing left to do
    python = sh([str(moved / "bin" / "python"), "-c", "import sys; print(sys.prefix)"], check=True).stdout.strip()
    assert python == str(moved)                                            # the interpreter itself never cared
    assert sh([script, str(old), str(new), str(tmp_path)]).returncode == 1   # not a venv


# ------------------------------------------------------------------ delegation in deploy/lib.sh

def _lib(env, extra=""):
    code = f'. "{DEPLOY}/lib.sh"; echo "$ACW_DELEGATE|$H"; {extra}'
    return sh(["bash", "-c", "set -euo pipefail; " + code], {"HOME": "/home/x", **env})


def test_lib_delegates_only_to_another_account(tmp_path):
    single = _lib({"ACW_SERVICE_USER": "", "ACW_DELEGATE": ""})
    assert single.stdout.strip() == "0|/home/x/audio-cleaner"
    owner = _lib({"ACW_SERVICE_USER": "someone-else", "ACW_SERVICE_HOME_DIR": "/home/acw", "ACW_DELEGATE": ""})
    assert owner.stdout.strip() == "1|/home/acw/audio-cleaner"
    as_acw = _lib({"ACW_SERVICE_USER": "someone-else", "ACW_SERVICE_HOME_DIR": "/home/acw", "ACW_DELEGATE": "",
                   "ACW_RUN": "/bin/echo"}, extra='as_acw systemctl --user status x; ctl is-active y')
    assert as_acw.stdout.splitlines()[1:] == ["systemctl --user status x", "systemctl --user is-active y"]
    itself = _lib({"ACW_SERVICE_USER": ME, "ACW_DELEGATE": "", "HOME": str(tmp_path)})
    assert itself.stdout.strip() == f"0|{tmp_path}/audio-cleaner"          # acw itself: no hop
    bad = _lib({"ACW_SERVICE_USER": "no-such-account-xyz", "ACW_DELEGATE": ""})
    assert bad.returncode != 0 and "no such account" in bad.stderr


def test_lib_refuses_while_a_move_between_accounts_is_unfinished(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "in-progress").write_text("silver -> acw\n")
    env = {"ACW_SERVICE_USER": "", "ACW_DELEGATE": "", "ACW_MOVE_STATE_DIR": str(state)}
    r = _lib(env)
    assert r.returncode != 0 and "unfinished (silver -> acw)" in r.stderr
    assert _lib({**env, "ACW_ALLOW_UNFINISHED_MOVE": "1"}).returncode == 0     # the migration's own health check
    (state / "in-progress").unlink()
    assert _lib(env).returncode == 0


def test_stage_for_acw_puts_the_commit_in_a_fresh_folder_and_keeps_four(tmp_path):
    if not shutil.which("git") or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    fake_run = tmp_path / "acw-run"
    fake_run.write_text('#!/bin/sh\nexec "$@"\n')
    fake_run.chmod(0o755)
    env = {"ACW_SERVICE_USER": "stand-in", "ACW_SERVICE_HOME_DIR": str(tmp_path / "acwhome"), "ACW_RUN": str(fake_run),
           "ACW_DELEGATE": ""}
    code = (f'REPO="{ROOT}"; . "{DEPLOY}/lib.sh"; for i in 1 2 3 4 5 6; do stage_for_acw c$i deploy/lib.sh >/dev/null; sleep 0.05; done; '
            'echo ok | put_as_acw /dev/stdin "$H/data/web-published" 600; ls "$(dirname "$H")/deploy"')
    r = sh(["bash", "-c", "set -euo pipefail; " + code], env, check=True)
    assert r.stdout.split() == ["c3", "c4", "c5", "c6"]                    # the newest four
    staged = tmp_path / "acwhome" / "deploy" / "c6"
    assert (staged / "deploy" / "lib.sh").exists() and not (staged / "server").exists()
    assert (tmp_path / "acwhome" / "audio-cleaner" / "data" / "web-published").read_text() == "ok\n"
    assert oct((tmp_path / "acwhome" / "audio-cleaner" / "data" / "web-published").stat().st_mode & 0o777) == "0o600"


# ------------------------------------------------------------------ install.sh: owner part and server part

def _staged_copy(dst: Path, with_info=True):
    """What `git archive HEAD deploy server common native` + DEPLOY_INFO gives, from the working tree (no .git)."""
    ign = shutil.ignore_patterns("__pycache__", "target", "libDF", "tract", "node_modules", "*.pyc")
    for sub in ("deploy", "native", "common/acw_common", "server/acw"):
        shutil.copytree(ROOT / sub, dst / sub, ignore=ign)
    shutil.copy(ROOT / "server" / "requirements.lock", dst / "server" / "requirements.lock")
    if with_info:
        (dst / "DEPLOY_INFO").write_text("commit=abc1234\nbranch=on-prem\nsubject=a test\n")


def _home_with_vps_env(tmp_path):
    h = tmp_path / "h"
    (h / "secrets").mkdir(parents=True)
    (h / "secrets" / "vps.env").write_text("ACW_VPS=u@203.0.113.7\n")
    return h


def test_install_server_part_plans_without_git_and_changes_nothing(tmp_path):
    stage = tmp_path / "stage"
    _staged_copy(stage)
    h = _home_with_vps_env(tmp_path)
    env = {"ACW_SERVICE_USER": "", "ACW_DELEGATE": "0", "ACW_HOME": str(h)}
    r = sh(["bash", str(stage / "deploy" / "install.sh"), "--plan"], env, check=True)
    assert r.stdout.strip() == "plan lock_changed=1 need_df=1 need_live=1"   # nothing deployed yet: everything is needed
    assert not (h / "app" / "VERSION").exists() and not (h / "bin" / "acw-live").exists()
    noinfo = tmp_path / "noinfo"
    _staged_copy(noinfo, with_info=False)
    r = sh(["bash", str(noinfo / "deploy" / "install.sh"), "--server-only"], env)
    assert r.returncode == 1 and "neither .git nor DEPLOY_INFO" in r.stderr


def test_install_refuses_server_part_started_by_the_owner_and_a_missing_hop(tmp_path):
    env = {"ACW_SERVICE_USER": "someone-else", "ACW_SERVICE_HOME_DIR": str(tmp_path / "acw"), "ACW_DELEGATE": ""}
    r = sh(["bash", str(DEPLOY / "install.sh"), "--server-only"], env)
    assert r.returncode == 2 and "is the part that runs as someone-else" in r.stderr
    # the owner's run needs to be able to act as acw before it stages or builds anything
    r = sh(["bash", str(DEPLOY / "install.sh"), "--force"], {**env, "ACW_RUN": "/bin/false"})
    assert r.returncode == 1 and "cannot run commands as someone-else" in r.stderr
    assert not (tmp_path / "acw" / "deploy").exists()


def test_vps_parts_are_the_owners_only(tmp_path):
    r = sh(["bash", str(DEPLOY / "publish-web.sh")], {"ACW_SERVICE_USER": ME, "ACW_DELEGATE": ""})
    assert r.returncode == 1 and "needs the owner's VPS login key" in r.stderr
    text = (DEPLOY / "install.sh").read_text()
    assert 'if [ "$ROLE" = server ]; then' in text and "vps_parts" in text
    # acw's run ends before the VPS parts (website, smoke test): only the owner's or a single account's run reaches them
    assert text.index('if [ "$ROLE" = server ]; then\n  [ "$STATUS" = 0 ]') < text.rindex('vps_parts "$STATUS"')


def test_drain_status_goes_through_the_service_account_helpers(tmp_path):
    h = tmp_path / "h"
    (h / "data").mkdir(parents=True)
    r = sh(["bash", str(DEPLOY / "drain.sh"), "status"], {"ACW_SERVICE_USER": "", "ACW_DELEGATE": "0", "ACW_HOME": str(h)}, check=True)
    assert r.stdout.split("\n")[:2] == ["drain off", "live sessions: 0"]
    sh(["bash", str(DEPLOY / "drain.sh"), "on"], {"ACW_SERVICE_USER": "", "ACW_DELEGATE": "0", "ACW_HOME": str(h)}, check=True)
    assert (h / "data" / "drain").exists()


# ------------------------------------------------------------------ the migration scripts, on scratch trees

def _owner_tree(tmp_path):
    """A tree shaped like the owner's side before the move, with stand-ins for everything."""
    base = tmp_path / "t"
    owner_home, acw_home = base / "silver", base / "acw"
    ac = owner_home / "audio-cleaner"
    for d in ("app/acw", "app.prev", "venv/bin", "venv-gpu/bin", "bin", "models/x", "wasm", "data", "secrets", "dev", "src"):
        (ac / d).mkdir(parents=True)
    # a python stand-in that answers `-m acw.admin activity` (the check that nothing is running)
    (ac / "venv/bin/python").write_text('#!/bin/sh\necho "file jobs: none; live sessions: 0"\n')
    (ac / "venv/bin/python").chmod(0o755)
    (ac / "venv/bin/uvicorn").write_text(f"#!{ac}/venv/bin/python\n")
    (ac / "venv/pyvenv.cfg").write_text(f"command = x -m venv {ac}/venv\n")
    (ac / "venv-gpu/pyvenv.cfg").write_text("home = /usr/bin\n")
    (ac / "bin/acw-live").write_text("x")
    (ac / "data/acw.sqlite3").write_text("db")
    for f in ("acw.env", "vps.env", "update-signing-key.pem", "github-actions-x.token"):
        (ac / "secrets" / f).write_text("K=1\n")
    (ac / "secrets/acw-worker.env").write_text(f"ACW_GITHUB_TOKEN_FILE={ac}/secrets/github-releases-read.token\n")
    (ac / "secrets/github-releases-read.token").write_text("t")
    for f in (ac / "secrets").iterdir():
        f.chmod(0o600)
    ssh = owner_home / ".ssh"
    ssh.mkdir()
    (ssh / "acw_tunnel").write_text("k")
    (ssh / "acw_tunnel.pub").write_text("p")
    if shutil.which("ssh-keygen"):
        sh(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(base / "hk")], check=True)
        (ssh / "known_hosts").write_text("vps " + " ".join((base / "hk.pub").read_text().split()[:2]) + "\n")
    ud = owner_home / ".config/systemd/user"
    (ud / "default.target.wants").mkdir(parents=True)
    (ud / "timers.target.wants").mkdir()
    for u in ("acw-web", "acw-stream", "acw-live", "acw-worker@1"):
        (ud / "default.target.wants" / f"{u}.service").symlink_to(f"../{u}.service")
    (ud / "timers.target.wants" / "acw-backup.timer").symlink_to("../acw-backup.timer")
    (base / "backups").mkdir()
    owner_home.chmod(0o750)
    return base, owner_home, acw_home


ACW_EXISTS = subprocess.run(["getent", "passwd", "acw"], capture_output=True).returncode == 0


def _move_env(base, owner_home, acw_home, **extra):
    return {"ACW_MOVE_OWNER_HOME": str(owner_home), "ACW_MOVE_SVC_HOME": str(acw_home), "ACW_BACKUP_DIR": str(base / "backups"),
            "ACW_MOVE_STATE_DIR": str(base / "state"), **extra}


needs_tools = pytest.mark.skipif(not all(shutil.which(t) for t in ("sudo", "systemctl", "ssh-keygen", "install", "stat", "getent")),
                                 reason="the dry run's checks look for these programs")


@needs_tools
def test_move_to_acw_dry_run_lists_every_step_and_changes_nothing(tmp_path):
    base, owner_home, acw_home = _owner_tree(tmp_path)
    before = tree(base)
    r = sh(["bash", str(HOST / "move-to-acw.sh"), "--dry-run", "--owner", ME], _move_env(base, owner_home, acw_home))
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    acw = str(acw_home)
    steps = [
        "== checks (nothing is changed)", "running now: file jobs: none; live sessions: 0",
        # (on a machine where the move has been made, the account exists: the step says so instead of creating it)
        ("exists (home " + acw if ACW_EXISTS else "useradd --create-home --home-dir " + acw + " --shell /usr/sbin/nologin --user-group"),
        "usermod -aG acw " + ME,
        "loginctl enable-linger acw", "visudo -cf", f"{ME} ALL=(acw) NOPASSWD: ALL",
        "/usr/local/bin/acw-run", "/usr/local/bin/acw-ctl", "/usr/local/bin/acw-logs", "/usr/local/bin/acw-admin", "/usr/local/bin/acw-sh",
        "== stop and disable the services of " + ME, "in-progress",
        "touch " + str(owner_home / "audio-cleaner/data/drain"),
        "disable --now acw-stream.service acw-live.service acw-web.service acw-worker@1.service acw-backup.timer",
        "== database snapshot with everything stopped", " backup " + str(owner_home / "audio-cleaner/data/backups"),
        "== move the folders", "== Python venvs: rewrite their absolute paths", "fix-venv-paths.sh",
        "== owner and permissions", "chmod -R u+rwX\\,g=rX\\,o= " + acw + "/audio-cleaner/venv", f"setfacl -m u:{ME}:x",
        "== the tunnel key and the VPS host key for acw", acw + "/.ssh/acw_tunnel", "ssh-keygen -F vps",
        "chown -R acw:acw " + str(base / "backups"),
        "== rebuild the live engine (acw-live)", "ACW_PYTHON=" + acw + "/audio-cleaner/venv/bin/python", "native/live/build.sh",
        "== install the units in the user manager of acw", "write acw to /etc/acw-service-user", "enable --now acw-stream.service",
        "check_running", "== smoke test", "deploy/smoke.py", "== isolation", "== done",
    ]
    pos = 0
    for needle in steps:
        i = out.find(needle, pos)
        assert i >= 0, f"missing or out of order: {needle!r}\n{out}"
        pos = i
    for item in ("app", "app.prev", "venv", "venv-gpu", "bin", "models", "wasm", "data"):
        assert f"+ mv {owner_home}/audio-cleaner/{item} {acw}/audio-cleaner/{item}" in out
    moved_secrets = sorted(Path(line.split()[-1]).name for line in out.splitlines() if line.startswith("+ mv ") and "/secrets/" in line)
    assert moved_secrets == ["acw-worker.env", "acw.env", "github-releases-read.token", "vps.env"]
    assert "update-signing-key.pem" not in "\n".join(l for l in out.splitlines() if l.startswith("+ "))   # the owner's own key stays
    assert "left in " in out and "update-signing-key.pem" in out and "github-actions-x.token" in out
    assert "id_ed25519" not in "\n".join(l for l in out.splitlines() if l.startswith("+ install"))      # no VPS login key for acw
    assert tree(base) == before                                                                   # nothing was changed


@needs_tools
def test_move_to_acw_dry_run_reports_problems_and_exits_1(tmp_path):
    base, owner_home, acw_home = _owner_tree(tmp_path)
    (owner_home / "audio-cleaner/venv/bin/python").write_text('#!/bin/sh\necho "file jobs: 2 running; live sessions: 1"\n')
    owner_home.chmod(0o755)
    r = sh(["bash", str(HOST / "move-to-acw.sh"), "--dry-run", "--owner", ME], _move_env(base, owner_home, acw_home))
    assert r.returncode == 1
    assert "work is running (file jobs: 2 running; live sessions: 1)" in r.stderr
    assert "is mode 755" in r.stderr                                                              # the owner's home must not be world-readable
    # the real run would refuse (and stops before changing anything), --force goes on
    r = sh(["bash", str(HOST / "move-to-acw.sh"), "--yes", "--owner", ME], _move_env(base, owner_home, acw_home))
    assert r.returncode != 0 and ("run it with sudo" in r.stderr or "problem(s) above; nothing was changed" in r.stderr)


@needs_tools
def test_move_back_dry_run_lists_the_reverse_steps(tmp_path):
    base, owner_home, acw_home = _owner_tree(tmp_path)
    # the state after the move: the folders at acw's, links at the owner's
    (acw_home).mkdir()
    (acw_home / "audio-cleaner").mkdir()
    ac_o, ac_a = owner_home / "audio-cleaner", acw_home / "audio-cleaner"
    for item in ("app", "app.prev", "venv", "venv-gpu", "bin", "models", "wasm", "data"):
        shutil.move(str(ac_o / item), str(ac_a / item))
    for item in ("bin", "models", "wasm", "venv", "venv-gpu"):
        (ac_o / item).symlink_to(ac_a / item)
    (ac_a / "secrets").mkdir()
    for f in ("acw.env", "acw-worker.env", "vps.env"):
        shutil.move(str(ac_o / "secrets" / f), str(ac_a / "secrets" / f))
    shutil.move(str(owner_home / ".config"), str(acw_home / ".config"))
    before = tree(base)
    env = _move_env(base, owner_home, acw_home, ACW_SERVICE_USER_NAME=ME)
    r = sh(["bash", str(HOST / "move-back-to-silver.sh"), "--dry-run", "--owner", "root"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    for needle in ("== stop and disable the services of " + ME, "disable --now acw-stream.service", "== database snapshot",
                   f"+ rm {ac_o}/bin", f"+ mv {ac_a}/venv {ac_o}/venv", f"+ mv {ac_a}/secrets/acw.env {ac_o}/secrets/acw.env",
                   "fix-venv-paths.sh " + str(ac_a) + " " + str(ac_o), "== no tunnel key left with " + ME,
                   f"chown -R root:root {base / 'backups'}", "== rebuild the live engine", "rm -f /etc/acw-service-user",
                   "enable --now acw-stream.service", "== smoke test", "== done"):
        assert needle in out, f"missing: {needle!r}\n{out}"
    assert tree(base) == before


def _can_chown_to_self():
    try:
        grp.getgrnam(ME)
        return True
    except KeyError:
        return False


def _harness(base, direction_from, direction_to, to_is_svc, tmp_path):
    h = tmp_path / "harness.sh"
    h.write_text(f'''set -euo pipefail
export ACW_SERVICE_USER_NAME={ME} ACW_MOVE_STATE_DIR={base}/state
. "{HOST}/move-lib.sh"
OWNER={ME}; set_direction {ME} {ME}
FROM_HOME={base}/{direction_from}; TO_HOME={base}/{direction_to}
FROM_H=$FROM_HOME/audio-cleaner; TO_H=$TO_HOME/audio-cleaner; TO_IS_SVC={to_is_svc}; OWNER_HOME=$FROM_HOME
m_move
''')
    return h


@pytest.mark.skipif(not _can_chown_to_self(), reason="needs a group named like the user")
def test_the_move_and_the_way_back_on_a_scratch_tree(tmp_path):
    """m_move for real, between two folders of the current user (the two accounts are the only thing replaced)."""
    base, owner_home, acw_home = _owner_tree(tmp_path)
    acw_home.mkdir()
    ac_o, ac_a = owner_home / "audio-cleaner", acw_home / "audio-cleaner"
    (ac_o / "bin/ffmpeg").write_text("#!/bin/sh\n")
    (ac_o / "bin/ffmpeg").chmod(0o775)
    (ac_o / "models").chmod(0o775)
    r = sh(["bash", str(_harness(base, "silver", "acw", 1, tmp_path))], check=True)
    # moved: the production parts and the allowlisted secrets; everything else stays
    for item in ("app", "app.prev", "venv", "venv-gpu", "bin", "models", "wasm", "data"):
        assert (ac_a / item).is_dir() and not (ac_a / item).is_symlink()
    assert sorted(p.name for p in (ac_a / "secrets").iterdir()) == ["acw-worker.env", "acw.env", "github-releases-read.token", "vps.env"]
    assert sorted(p.name for p in (ac_o / "secrets").iterdir()) == ["github-actions-x.token", "update-signing-key.pem"]
    assert (ac_o / "dev").is_dir() and (ac_o / "src").is_dir()
    assert (ac_a / "data/acw.sqlite3").read_text() == "db"
    # the owner's side: links for what tools and the web build read, nothing for app, data, secrets
    for item in ("bin", "models", "wasm", "venv", "venv-gpu"):
        assert (ac_o / item).is_symlink() and (ac_o / item).resolve() == (ac_a / item).resolve()
    assert not (ac_o / "app").exists() and not (ac_o / "data").exists()
    # paths: venv scripts, pyvenv.cfg and the env file name the new folder, never the old one
    assert (ac_a / "venv/bin/uvicorn").read_text() == f"#!{ac_a}/venv/bin/python\n"
    assert str(ac_o) not in (ac_a / "venv/pyvenv.cfg").read_text()
    assert (ac_a / "secrets/acw-worker.env").read_text() == f"ACW_GITHUB_TOKEN_FILE={ac_a}/secrets/github-releases-read.token\n"
    # permissions: acw-only for app, data, secrets and the secret files; the shared folders are not world-readable
    mode = lambda p: (p.stat().st_mode & 0o777)  # noqa: E731
    assert mode(ac_a) == 0o750
    for d in ("app", "app.prev", "data", "secrets"):
        assert mode(ac_a / d) == 0o700, d
    for f in (ac_a / "secrets").iterdir():
        assert mode(f) == 0o600, f.name
    for d in ("bin", "models", "venv"):
        assert mode(ac_a / d) & 0o007 == 0 and mode(ac_a / d) & 0o050 == 0o050, d      # group r-x, others nothing
    assert mode(ac_a / "bin/ffmpeg") & 0o110 == 0o110                                   # executables stay executable
    # and back
    sh(["bash", str(_harness(base, "acw", "silver", 0, tmp_path))], check=True)
    for item in ("app", "app.prev", "venv", "venv-gpu", "bin", "models", "wasm", "data"):
        assert (ac_o / item).is_dir() and not (ac_o / item).is_symlink(), item           # the links gave way to the folders
        assert not (ac_a / item).exists()
    assert (ac_o / "secrets/acw.env").exists() and (ac_o / "secrets/update-signing-key.pem").exists()
    assert (ac_o / "venv/bin/uvicorn").read_text() == f"#!{ac_o}/venv/bin/python\n"
    assert (ac_o / "secrets/acw-worker.env").read_text() == f"ACW_GITHUB_TOKEN_FILE={ac_o}/secrets/github-releases-read.token\n"
    assert r.returncode == 0


@pytest.mark.skipif(not _can_chown_to_self(), reason="needs a group named like the user")
def test_the_move_never_overwrites(tmp_path):
    base, owner_home, acw_home = _owner_tree(tmp_path)
    (acw_home / "audio-cleaner/app").mkdir(parents=True)
    r = sh(["bash", str(_harness(base, "silver", "acw", 1, tmp_path))])
    assert r.returncode != 0 and "already exists" in r.stderr
    assert (owner_home / "audio-cleaner/app/acw").is_dir()                                  # nothing of the source was lost


# ------------------------------------------------------------------ nothing names the owner's home any more

def test_units_examples_and_code_do_not_name_the_owners_home():
    files = list((DEPLOY / "systemd").glob("*")) + list(DEPLOY.glob("*.example")) + list((ROOT / "server" / "acw").glob("*.py"))
    for f in files:
        text = f.read_text(errors="replace")
        assert "/home/silver" not in text, f"{f.relative_to(ROOT)} names /home/silver"
    tunnels = [(DEPLOY / "systemd" / n).read_text() for n in ("acw-tunnel.service", "acw-tunnel-bulk.service", "acw-tunnel-live.service")]
    for t in tunnels:
        assert "-i %h/.ssh/acw_tunnel" in t and "-F none" in t and "HostKeyAlias=vps" in t   # the tunnel key in acw's home, never a login key
        assert "id_ed25519" not in t


def test_alert_mails_name_the_wrappers_not_the_owners_own_commands():
    for name in ("monitor.py", "alerts.py"):
        text = (ROOT / "server" / "acw" / name).read_text()
        for line in text.splitlines():
            if "Check:" in line:
                assert "journalctl --user" not in line and "systemctl --user" not in line and "python -m acw.admin" not in line, line
