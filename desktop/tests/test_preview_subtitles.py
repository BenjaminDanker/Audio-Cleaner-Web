"""Subtitles in the preview: SRT/WebVTT files and the track inside a video, shown in time with the sound."""
from __future__ import annotations

import subprocess

import pytest

from audio_cleaner_desktop.gui import subtitles

from conftest import bundled_ffmpeg, wait_until

SRT = """1
00:00:00,500 --> 00:00:01,800
Hello <i>there</i>

2
00:00:02,000 --> 00:00:03,250
second line
and its next line
"""

VTT = """WEBVTT

00:00.500 --> 00:01.800 align:center
Hello there

intro
00:01:02.000 --> 00:01:03.000
a minute in
"""


def test_srt_and_vtt_become_timed_cues():
    srt = subtitles.Track(subtitles.parse(SRT))
    assert srt.at(0.4) == "" and srt.at(0.5) == "Hello there" and srt.at(1.9) == ""
    assert srt.at(2.5) == "second line\nand its next line"
    vtt = subtitles.Track(subtitles.parse(VTT))
    assert vtt.at(1.0) == "Hello there" and vtt.at(62.5) == "a minute in" and not subtitles.Track([])


def test_subtitles_inside_a_video_are_read(tmp_path, speech_wav):
    (tmp_path / "s.srt").write_text(SRT, encoding="utf-8")
    video = tmp_path / "v.mkv"
    frames = bytes(64 * 36 * 3) * 40
    made = subprocess.run(
        [str(bundled_ffmpeg()), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "64x36", "-r", "10",
         "-i", "pipe:0", "-i", str(speech_wav), "-i", str(tmp_path / "s.srt"), "-t", "4", "-map", "0:v", "-map", "1:a",
         "-map", "2:s", "-c:v", "rawvideo", "-pix_fmt", "yuv420p", "-c:a", "flac", "-c:s", "srt", str(video)],
        input=frames,
    )
    if made.returncode != 0:
        pytest.skip("this FFmpeg cannot make the test video")
    track = subtitles.Track(subtitles.read_embedded(bundled_ffmpeg(), video))
    assert track.at(1.0) == "Hello there"
    assert subtitles.read_embedded(bundled_ffmpeg(), tmp_path / "missing.mkv") == []


def test_the_preview_shows_the_subtitle_of_the_moment(qapp, tmp_path, speech_wav):
    from audio_cleaner_desktop.gui import player

    if player.load_multimedia() is None:
        pytest.skip("QtMultimedia is not available here")
    srt = tmp_path / "talk.srt"
    srt.write_text(SRT, encoding="utf-8")
    pane = player.PreviewPane()
    pane.resize(600, 400)
    pane.show()
    try:
        pane.show_files(1, speech_wav, speech_wav, False, "talk", srt)
        assert pane._cc.isVisibleTo(pane)
        loaded = player.load_multimedia().QMediaPlayer.MediaStatus.LoadedMedia
        assert wait_until(lambda: pane._player.mediaStatus() == loaded, 10)
        pane._player.setPosition(600)
        assert wait_until(lambda: pane._caption.text() == "Hello there", 10)
        pane._cc.setChecked(False)  # CC off: hidden
        assert not pane._caption.isVisible()
        pane.show_files(2, speech_wav, None, False, "plain")  # no subtitles: no CC button
        assert not pane._cc.isVisibleTo(pane)
    finally:
        pane.release()
        pane.deleteLater()
