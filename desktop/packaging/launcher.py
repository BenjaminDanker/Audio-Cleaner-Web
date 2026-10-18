"""PyInstaller entry script: the frozen equivalent of `python -m audio_cleaner_desktop`."""
import os

# One thread for numpy's own pool, before numpy is imported (see audio_cleaner_desktop/__main__.py, which does the same
# for `python -m audio_cleaner_desktop`).
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

# First of all, before any library is imported: no console window may flash up for programs that libraries start
# (Windows only; does nothing elsewhere). See audio_cleaner_desktop/quiet_popen.py.
from audio_cleaner_desktop import quiet_popen  # noqa: E402

quiet_popen.install()

import multiprocessing  # noqa: E402
import sys  # noqa: E402

from audio_cleaner_desktop.__main__ import main  # noqa: E402

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
