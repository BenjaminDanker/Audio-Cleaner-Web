"""The picture of the preview, decoded by our own FFmpeg and timed to the sound.

Qt's player decodes every video frame and holds the sound back when the picture is late, so a video the computer
cannot decode in real time (8K, for example) plays in slow motion, sound included. Here the sound plays in its own
QMediaPlayer and never waits; the picture comes from an FFmpeg process that decodes it at the preview's size and is
shown against the sound's clock. Frames that come too late are dropped; when the decoder stays behind it jumps ahead
to the key frame nearest the sound (as video players do on slow computers), and if it keeps falling behind it decodes
key frames only. The picture keeps up; the sound is never affected.
"""
from __future__ import annotations

import json
import logging
import queue
import re
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QRect, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)

# Video codecs the bundled FFmpeg decodes (desktop/packaging/build_ffmpeg.sh VIDEO_DECODERS). Others use Qt's player.
DECODABLE = {"rawvideo", "h264", "hevc", "vp8", "vp9", "mpeg4", "mpeg2video", "mpeg1video", "mjpeg", "prores", "dnxhd", "h263",
             "flv1", "msmpeg4v3", "wmv3", "vc1", "theora"}
MAX_W, MAX_H = 1280, 720  # the preview pane is smaller than this; decoding to a bigger picture would only cost time
# Decoder effort: every frame, or key frames only (a generic FFmpeg decoder option; h264, hevc and others honour it).
LEVELS: tuple[tuple[str, ...], ...] = ((), ("-skip_frame", "nokey"))
BEHIND_S = 0.5  # the picture is this far behind the sound ...
BEHIND_FOR_S = 1.0  # ... for this long: jump ahead to the sound
JUMPS_BEFORE_KEYFRAMES = 2  # this many jumps in a row without keeping up: decode key frames only
IN_SYNC_RESET_S = 10.0  # the picture kept up this long: the jumps are forgiven
QUEUE_FRAMES = 6

_PTS = re.compile(r"pts_time:\s*(-?[0-9.]+)")


@dataclass(frozen=True)
class VideoInfo:
    codec: str
    start: float  # the file's own start time; the sound player's positions count from it
    width: int  # as shown: rotation and pixel aspect applied, scaled to fit MAX_W x MAX_H, even
    height: int


def probe(ffprobe: Path, path: Path) -> VideoInfo | None:
    """The first video stream's codec and the size to decode it at; None without a video stream or on errors."""
    try:
        out = subprocess.run(
            [str(ffprobe), "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json",
             str(path)],
            capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL,
        )
        data = json.loads(out.stdout or "{}")
        stream = (data.get("streams") or [None])[0]
        start = float((data.get("format") or {}).get("start_time") or 0.0)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    if not stream or not stream.get("width") or not stream.get("height"):
        return None
    w, h = int(stream["width"]), int(stream["height"])
    num, _, den = str(stream.get("sample_aspect_ratio") or "1:1").partition(":")
    try:
        sar = int(num) / int(den) if int(num) > 0 and int(den) > 0 else 1.0
    except ValueError:
        sar = 1.0
    w = w * sar
    rotation = 0
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(side["rotation"])
    if abs(rotation) % 180 == 90:
        w, h = h, w
    scale = min(1.0, MAX_W / w, MAX_H / h)
    even = lambda v: max(2, int(round(v * scale / 2)) * 2)  # noqa: E731
    return VideoInfo(str(stream.get("codec_name") or ""), start, even(w), even(h))


class FrameStream:
    """One FFmpeg process decoding from `start` seconds; frames (time in seconds, QImage) arrive in `frames`.
    `exact`: the first frame is the one at `start` (a seek by the person); otherwise decoding begins at the key frame
    before `start` and the frames before it are simply late (catching up)."""

    def __init__(self, ffmpeg: Path, path: Path, info: VideoInfo, start: float, level: int, exact: bool = True):
        self.start, self.level = start, level
        self.frames: queue.Queue[tuple[float, QImage] | None] = queue.Queue(maxsize=QUEUE_FRAMES)
        self._size = info.width * info.height * 4
        self._w, self._h = info.width, info.height
        self._pts: deque[float] = deque()
        self._pts_ready = threading.Condition()
        self._stopped = False
        self._offset = info.start
        args = [
            str(ffmpeg), "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
            *LEVELS[level], *(() if exact else ("-noaccurate_seek",)), "-copyts",
            "-protocol_whitelist", "file,pipe", "-ss", f"{max(0.0, start) + info.start:.3f}", "-i", str(path),
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", f"scale={info.width}:{info.height}:flags=fast_bilinear,format=rgb0,showinfo",
            "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1",
        ]
        self._proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self._read_log, name="preview-video-log", daemon=True).start()
        threading.Thread(target=self._read_frames, name="preview-video", daemon=True).start()

    def stop(self) -> None:
        self._stopped = True
        try:
            self._proc.kill()
        except OSError:
            pass
        with self._pts_ready:
            self._pts_ready.notify_all()
        try:  # unblock a reader waiting on a full queue
            while True:
                self.frames.get_nowait()
        except queue.Empty:
            pass

    def _read_log(self) -> None:
        for raw in self._proc.stderr:  # type: ignore[union-attr]
            m = _PTS.search(raw.decode("utf-8", "replace"))
            if m:
                with self._pts_ready:
                    self._pts.append(float(m.group(1)))
                    self._pts_ready.notify_all()
        with self._pts_ready:
            self._pts_ready.notify_all()

    def _read_frames(self) -> None:
        out = self._proc.stdout
        try:
            while not self._stopped:
                data = out.read(self._size)  # type: ignore[union-attr]
                if not data or len(data) < self._size:
                    break
                with self._pts_ready:
                    deadline = time.monotonic() + 2.0
                    while not self._pts and not self._stopped and time.monotonic() < deadline:
                        self._pts_ready.wait(0.1)
                    t = self._pts.popleft() - self._offset if self._pts else None  # -copyts: the file's own times
                if t is None:
                    break
                image = QImage(data, self._w, self._h, self._w * 4, QImage.Format.Format_RGBX8888).copy()
                while not self._stopped:
                    try:
                        self.frames.put((t, image), timeout=0.2)
                        break
                    except queue.Full:
                        continue
        finally:
            if not self._stopped:
                try:
                    self.frames.put(None, timeout=1.0)  # end of the video
                except queue.Full:
                    pass


class VideoView(QWidget):
    """Draws the current picture, fitted and centred, on black."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.image: QImage | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    def show_image(self, image: QImage | None) -> None:
        self.image = image
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("black"))
        if self.image is not None and not self.image.isNull():
            size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRect(0, 0, size.width(), size.height())
            target.moveCenter(self.rect().center())
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(target, self.image)
        p.end()


class SyncedVideo(QObject):
    """Shows the picture of `path` in `view` at the time `clock()` (seconds) says, which is the sound's position."""

    def __init__(self, view: VideoView, clock: Callable[[], float], ffmpeg: Path, parent: QObject | None = None):
        super().__init__(parent)
        self.view, self._clock, self._ffmpeg = view, clock, ffmpeg
        self._path: Path | None = None
        self._info: VideoInfo | None = None
        self._stream: FrameStream | None = None
        self._next: tuple[float, QImage] | None = None  # the first frame not shown yet
        self._shown_since_seek = False
        self._playing = False
        self._level = 0
        self._behind_since: float | None = None
        self._ended = False
        self._restarted_at = 0.0
        self._waiting_first = False  # restarted, no frame yet: not "behind", just starting
        self._startup_s = 0.5  # how long the last restart took to give its first frame
        self._jumps = 0
        self._in_sync_since: float | None = None
        self._timer = QTimer(self, interval=15, timeout=self._tick)

    @property
    def level(self) -> int:
        return self._level

    def open(self, path: Path, info: VideoInfo, start: float = 0.0) -> None:
        self.close()
        self._path, self._info, self._level = path, info, 0
        self._jumps = 0
        self._restart(start)
        self._timer.start()

    def close(self) -> None:
        """Stop decoding and let go of the file (Windows locks open files)."""
        self._timer.stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream = None
        self._next = None
        self._path = None

    def seek(self, seconds: float) -> None:
        if self._path is not None:
            self._restart(seconds)

    def set_playing(self, playing: bool) -> None:
        self._playing = playing
        self._behind_since = None

    def _restart(self, start: float, exact: bool = True) -> None:
        if self._stream is not None:
            self._stream.stop()
        self._next = None
        self._shown_since_seek = False
        self._behind_since = None
        self._ended = False
        self._restarted_at = time.monotonic()
        self._waiting_first = True
        assert self._path is not None and self._info is not None
        self._stream = FrameStream(self._ffmpeg, self._path, self._info, max(0.0, start), self._level, exact)

    def _pull(self) -> tuple[float, QImage] | None:
        if self._next is None and self._stream is not None and not self._ended:
            try:
                item = self._stream.frames.get_nowait()
            except queue.Empty:
                return None
            if item is None:
                self._ended = True
                return None
            if self._waiting_first:
                self._waiting_first = False
                self._startup_s = time.monotonic() - self._restarted_at
            self._next = item
        return self._next

    def _tick(self) -> None:
        if self._stream is None:
            return
        now = self._clock()
        show: QImage | None = None
        while (item := self._pull()) is not None:
            t, image = item
            if t <= now + 0.010 or not self._shown_since_seek and not self._playing:
                show, self._next = image, None  # due (older ones are simply replaced: dropped)
                if not self._playing:
                    break
            else:
                break
        if show is not None:
            self.view.show_image(show)
            self._shown_since_seek = True
        self._keep_up(now)

    def _keep_up(self, now: float) -> None:
        """When the picture stays behind the sound, decode with less effort (and at the least, jump ahead)."""
        mono = time.monotonic()
        if not self._playing or self._ended:
            self._behind_since = None
            return
        if self._waiting_first:  # a decoder that is starting is never interrupted: its frames are shown when they come
            return
        head = self._next[0] if self._next is not None else None
        behind = head is None or head < now - BEHIND_S
        if not behind:
            self._behind_since = None
            if self._in_sync_since is None:
                self._in_sync_since = mono
            elif mono - self._in_sync_since > IN_SYNC_RESET_S:
                self._jumps = 0
            return
        self._in_sync_since = None
        if self._behind_since is None:
            self._behind_since = mono
            return
        if mono - self._behind_since < BEHIND_FOR_S:
            return
        self._jumps += 1
        if self._jumps >= JUMPS_BEFORE_KEYFRAMES and self._level < len(LEVELS) - 1:
            self._level += 1
            self._jumps = 0
            log.info("preview: this computer cannot decode every frame of this video in time; showing key frames")
        # Jump to where the sound will be once the decoder has started again (from the key frame before it).
        self._restart(now + max(0.3, 1.5 * self._startup_s), exact=False)
