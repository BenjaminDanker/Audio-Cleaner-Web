"""PyInstaller entry script: the frozen equivalent of `python -m audio_cleaner_desktop`."""
import multiprocessing
import sys

from audio_cleaner_desktop.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
