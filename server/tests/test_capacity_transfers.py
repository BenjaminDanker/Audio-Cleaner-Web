"""Uploads and downloads share one byte-rate budget (audit H3). The waits run on a fake clock: they are added up, not slept."""
import asyncio
from types import SimpleNamespace

import pytest

from acw import security
from acw.security import DownloadBandwidth, UploadBandwidth, transfer_budget

VARS = ("ACW_TRANSFER_MBPS_TOTAL", "ACW_DOWNLOAD_MBPS_TOTAL", "ACW_UPLOAD_MBPS_TOTAL")


@pytest.fixture
def clock(monkeypatch):
    """security's clock: `sleep` only moves it on. Returns the list holding the time."""
    now = [0.0]

    async def sleep(seconds):
        now[0] += seconds

    for name in VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(security, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(security, "asyncio", SimpleNamespace(sleep=sleep))
    return now


@pytest.mark.parametrize("shared", [True, False])
def test_an_upload_and_a_download_together_stay_under_one_budget(clock, monkeypatch, shared):
    monkeypatch.setenv("ACW_TRANSFER_MBPS_TOTAL", "0.2")  # 200 kB/s for both together; a free first second
    monkeypatch.setenv("ACW_UPLOAD_MBPS_USER", "5")
    if shared:
        budget = transfer_budget()
        up, down = UploadBandwidth(budget), DownloadBandwidth(budget)
    else:
        up, down = UploadBandwidth(), DownloadBandwidth()  # each builds its own budget
    user = up.for_user("u")

    async def upload():
        for _ in range(20):
            await up.take(user, 10_000)

    async def download():
        for _ in range(20):
            await down.take(10_000)

    async def both():
        await asyncio.gather(upload(), download())

    asyncio.run(both())
    if shared:
        assert clock[0] > 0.8  # 400 kB through 200 kB/s with a free first second: about one second of waiting
    else:
        assert clock[0] == 0  # 200 kB each, under two caps of 200 kB/s: nothing waits


def test_uploads_and_downloads_are_one_budget_and_the_older_variables_still_count(monkeypatch):
    assert security.uploads.total is security.downloads.total
    for name in VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ACW_DOWNLOAD_MBPS_TOTAL", "6")
    monkeypatch.setenv("ACW_UPLOAD_MBPS_TOTAL", "8")
    assert transfer_budget().rate == 6e6  # the smaller of the two
    monkeypatch.setenv("ACW_TRANSFER_MBPS_TOTAL", "4")
    assert transfer_budget().rate == 4e6  # the new one wins
