"""Uploads and downloads share one byte-rate budget (audit H3)."""
import asyncio
import time

from acw import security
from acw.security import DownloadBandwidth, UploadBandwidth, transfer_budget


def _clear(monkeypatch):
    for name in ("ACW_TRANSFER_MBPS_TOTAL", "ACW_DOWNLOAD_MBPS_TOTAL", "ACW_UPLOAD_MBPS_TOTAL"):
        monkeypatch.delenv(name, raising=False)


def test_the_process_shares_one_budget_between_uploads_and_downloads():
    assert security.uploads.total is security.downloads.total
    assert security.uploads.total.rate == 5e6


def test_default_is_five_megabytes_per_second_for_both_together(monkeypatch):
    _clear(monkeypatch)
    assert transfer_budget().rate == 5e6
    monkeypatch.setenv("ACW_TRANSFER_MBPS_TOTAL", "3.5")
    assert transfer_budget().rate == 3.5e6
    monkeypatch.setenv("ACW_TRANSFER_MBPS_TOTAL", "0")
    assert transfer_budget() is None and DownloadBandwidth().total is None and UploadBandwidth().total is None


def test_the_older_variables_still_count_when_the_new_one_is_not_set(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("ACW_DOWNLOAD_MBPS_TOTAL", "6")
    monkeypatch.setenv("ACW_UPLOAD_MBPS_TOTAL", "8")
    assert transfer_budget().rate == 6e6  # the smaller
    monkeypatch.setenv("ACW_TRANSFER_MBPS_TOTAL", "4")
    assert transfer_budget().rate == 4e6  # the new one wins


def test_an_upload_and_a_download_together_stay_under_the_budget(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("ACW_TRANSFER_MBPS_TOTAL", "0.2")  # 200 kB/s
    monkeypatch.setenv("ACW_UPLOAD_MBPS_USER", "5")
    budget = transfer_budget()
    up, down = UploadBandwidth(budget), DownloadBandwidth(budget)
    user = up.for_user("u")

    async def go():
        async def upload():
            for _ in range(20):
                await up.take(user, 10_000)

        async def download():
            for _ in range(20):
                await down.take(10_000)

        await asyncio.gather(upload(), download())

    t0 = time.monotonic()
    asyncio.run(go())
    elapsed = time.monotonic() - t0
    # 400 kB through 200 kB/s with a free first second: about one second. Separate 200 kB/s caps would take none.
    assert elapsed > 0.8


def test_separate_budgets_would_not_have_added_up(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("ACW_TRANSFER_MBPS_TOTAL", "0.2")
    up, down = UploadBandwidth(), DownloadBandwidth()  # each builds its own budget

    async def go():
        async def upload():
            for _ in range(10):
                await up.take(up.for_user("u"), 10_000)

        async def download():
            for _ in range(10):
                await down.take(10_000)

        await asyncio.gather(upload(), download())

    t0 = time.monotonic()
    asyncio.run(go())
    assert time.monotonic() - t0 < 0.5  # 200 kB in total, each under its own 200 kB/s: no waiting
