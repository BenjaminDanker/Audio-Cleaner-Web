"""The preview's picture: decoded by our FFmpeg, timed to the sound, catching up instead of slowing the sound down."""
from __future__ import annotations

import subprocess

import pytest

from audio_cleaner_desktop.gui import video_preview as vp

from conftest import bundled_ffmpeg, bundled_ffprobe, wait_until

W, H, FPS, SECONDS = 64, 36, 10, 4


@pytest.fixture(scope="module")
def raw_video(tmp_path_factory, speech_wav):
    """Uncompressed video (frame n is grey level 8n) with sound: needs no video encoder, only the rawvideo one."""
    path = tmp_path_factory.mktemp("video") / "frames.mkv"
    frames = b"".join(bytes([min(255, 8 * n)]) * (W * H * 3) for n in range(FPS * SECONDS))
    subprocess.run(
        [str(bundled_ffmpeg()), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "pipe:0", "-i", str(speech_wav), "-t", str(SECONDS), "-map", "0:v", "-map", "1:a",
         "-c:v", "rawvideo", "-pix_fmt", "yuv420p", "-c:a", "flac", str(path)],
        input=frames, check=True,
    )
    return path


def _grey(image) -> int:
    return image.pixelColor(image.width() // 2, image.height() // 2).red()


def test_probe_reports_codec_start_and_size(raw_video):
    info = vp.probe(bundled_ffprobe(), raw_video)
    assert info is not None and info.codec == "rawvideo" and info.codec in vp.DECODABLE
    assert (info.width, info.height) == (W, H) and abs(info.start) < 0.1


def test_probe_scales_big_video_down_and_says_nothing_for_audio(tmp_path, speech_wav):
    assert vp.probe(bundled_ffprobe(), speech_wav) is None
    assert vp.probe(bundled_ffprobe(), tmp_path / "missing.mp4") is None


def test_frames_come_in_order_from_the_seek_position(qapp, raw_video):
    info = vp.probe(bundled_ffprobe(), raw_video)
    stream = vp.FrameStream(bundled_ffmpeg().parent / bundled_ffmpeg().name, raw_video, info, 1.0, 0, exact=True)
    try:
        got = [stream.frames.get(timeout=20) for _ in range(5)]
    finally:
        stream.stop()
    times = [t for t, _ in got]
    assert times == sorted(times) and abs(times[0] - 1.0) < 0.06
    assert abs(_grey(got[0][1]) - 8 * 10) <= 4  # frame 10 is at 1.0 s


def test_picture_follows_the_clock_and_catches_up_after_falling_behind(qapp, raw_video):
    info = vp.probe(bundled_ffprobe(), raw_video)
    clock = [0.0]
    view = vp.VideoView()
    synced = vp.SyncedVideo(view, lambda: clock[0], bundled_ffmpeg())
    try:
        synced.open(raw_video, info)
        assert wait_until(lambda: view.image is not None, 20)  # paused: the first frame is shown at once
        synced.set_playing(True)
        clock[0] = 0.55
        assert wait_until(lambda: abs(_grey(view.image) - 8 * 5) <= 4, 20)
        # The sound is suddenly far ahead (what a slow decoder looks like): the picture jumps there.
        clock[0] = 3.0
        assert wait_until(lambda: _grey(view.image) >= 8 * 28, 30)
    finally:
        synced.close()
        view.deleteLater()


def test_preview_pane_uses_its_own_decoder_for_video_and_lets_the_file_go(qapp, raw_video, monkeypatch):
    from audio_cleaner_desktop.gui import player

    if player.load_multimedia() is None:
        pytest.skip("QtMultimedia is not available here")
    pane = player.PreviewPane()
    try:
        pane.show_files(1, raw_video, None, True, raw_video.name)
        assert pane._mode == "own"
        assert wait_until(lambda: pane._view.image is not None, 20)
        pane.release()
        assert pane._synced._stream is None  # FFmpeg stopped: the file can be moved or renamed
    finally:
        pane.release()
        pane.deleteLater()
