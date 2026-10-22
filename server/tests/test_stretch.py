"""The editor preview's stretches (acw/stretch.py, the confined cutter acw/worker/cutter.py): the numbers, the kind of FLAC they are,
and that the cutter is confined and writes nothing. Real FFmpeg (~/audio-cleaner/bin) for the confinement; skipped when it is not there."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from acw import jobs, stretch
from acw_common import kept
from acw_common.ffmpeg import AUDIO_FORMATS

SR = 48_000
HOME = Path(os.environ.get("ACW_HOME", Path.home() / "audio-cleaner"))
pytestmark = pytest.mark.skipif(not (HOME / "bin/ffmpeg").exists(), reason="FFmpeg is not available")


def test_span_rounds_to_the_nearest_sample_caps_at_thirty_seconds_and_stops_at_the_end():
    total = 100 * SR
    assert stretch.span(0, 10, total) == (0, 10 * SR)
    assert stretch.span(None, None, total) == (0, 30 * SR)  # defaults: from the start, the longest allowed
    assert stretch.span(12.5, None, total) == (int(12.5 * SR), 30 * SR)
    assert stretch.span(1 / 3, 0.001, total) == (16000, 48)
    assert stretch.span(5, 500, total) == (5 * SR, 30 * SR)  # more than the cap is cut to it
    assert stretch.span(95, 30, total) == (95 * SR, 5 * SR)  # cut short by the end
    assert stretch.span(100 - 1 / SR, 1, total) == (total - 1, 1)
    assert stretch.span(0, 1e-9, total)[1] == 1  # never empty
    for start in (100, 100.5, 5000):
        with pytest.raises(stretch.OutOfRange):
            stretch.span(start, 1, total)


def test_the_cutter_writes_the_same_kind_of_flac_the_copies_are():
    from acw.worker import cutter

    assert cutter.FLAC24 == AUDIO_FORMATS["flac24"][1] and cutter.SAMPLE_RATE == kept.SAMPLE_RATE == stretch.SAMPLE_RATE
    assert stretch.MAX_SECONDS == jobs.STRETCH_MAX_SECONDS == 30


def test_the_cutter_is_confined_and_writes_nothing(tmp_path):
    """It is run the way stretch.cut runs it; from inside, the folder is read-only and the rest of the disk can not be read."""
    jid_dir = tmp_path / "job"
    jid_dir.mkdir()
    secret = tmp_path / "other" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text("secret")
    code = (
        "import sys, os\n"
        "from pathlib import Path\n"
        "from acw.worker import sandbox\n"
        f"sandbox.apply_landlock(sandbox.default_read_paths([{str(HOME / 'bin')!r}, {str(jid_dir)!r}]), ['/dev/null'])\n"
        "out = []\n"
        f"for p, mode in (({str(jid_dir / 'new.txt')!r}, 'w'), ({str(secret)!r}, 'r')):\n"
        "    try:\n"
        "        open(p, mode).close(); out.append('opened')\n"
        "    except PermissionError:\n"
        "        out.append('denied')\n"
        "print(','.join(out))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=stretch._environment(), cwd=str(jid_dir))
    assert r.stdout.strip() == "denied,denied", r.stderr
    assert not (jid_dir / "new.txt").exists()
    # and the program itself refuses nonsense
    for bad in ([], ["x", "relative/path.flac", "0", "10"], [str(HOME / "bin"), str(jid_dir / "a.flac"), "-1", "10"], [str(HOME / "bin"), str(jid_dir / "a.flac"), "0", "0"]):
        p = subprocess.run([sys.executable, "-m", "acw.worker.cutter", *bad], capture_output=True, env=stretch._environment(), cwd=str(jid_dir))
        assert p.returncode == 2 and p.stdout == b""
