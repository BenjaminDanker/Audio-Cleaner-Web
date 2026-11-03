"""The window's garbage collection (MainThreadCollector in gui/app.py): Python's automatic collection is off (a worker thread that
collected would delete Qt objects outside the main thread), a timer collects on the window's thread, and what the app holds at start is set aside."""
from __future__ import annotations

import gc


def test_automatic_collection_is_off_and_the_app_set_its_start_aside(qapp):
    assert not gc.isenabled()
    assert qapp.collector._timer.isActive() and qapp.collector._timer.interval() == 50
    assert gc.get_freeze_count() > 0  # (the libraries and the app's code are not gone through by a full collection)
