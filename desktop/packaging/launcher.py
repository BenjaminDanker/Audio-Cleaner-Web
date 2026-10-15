"""PyInstaller entry script: the frozen equivalent of `python -m audio_cleaner_desktop`."""
# First of all, before any library is imported: no console window may flash up for programs that libraries start
# (Windows only; does nothing elsewhere). See audio_cleaner_desktop/quiet_popen.py.
from audio_cleaner_desktop import quiet_popen

quiet_popen.install()

import multiprocessing  # noqa: E402
import sys  # noqa: E402

from audio_cleaner_desktop.__main__ import main  # noqa: E402

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
