"""Video conversion as its own job kind and as part of a job (options.video_out): the result's file type and picture size, whether the
picture is copied or encoded again, and what an encode costs (scaled by the picture's size and frame rate)."""
from __future__ import annotations

import math
import re

from .billing import pricing


# options.video_out: the video result's file type and the shorter side of its picture (None: the original size).
VIDEO_OUT_FORMATS = ("mp4", "mov", "mkv", "webm")
VIDEO_OUT_SIZES = (480, 720, 1080, 1440, 2160)


def encodes(params: dict) -> bool:
    """Will the job make the picture of its video again (options.video_out)? Until its file is read it may (None counts)."""
    return bool(params.get("options", {}).get("video_out")) and params.get("picture_encode") is not False


def _convert_rate(params: dict) -> int:
    """Micros a minute of the video conversion: the pinned rate times the picture's size factor (convert_factor, two decimals), in whole
    numbers (a float product can land a hair above the exact one and round up a micro)."""
    hundredths = round(float(params.get("convert_factor") or 1.0) * 100)
    return -(-int(params["convert_rate"]) * hundredths // 100)


def convert_cost(params: dict, seconds: float) -> int:
    """What the video conversion of a job costs for `seconds` (its pinned `convert_rate`, times `convert_factor`: 1 up to 1080p at 30
    frames a second, more for a bigger or faster picture): only when the picture is made again (`picture_encode`). Both are worked out
    from what the client read of the video (video_out.source) and checked when the file is read."""
    if not params.get("convert_rate") or not params.get("picture_encode"):
        return 0
    return pricing.cost_micros("convert", "standard", seconds, _convert_rate(params))


def convert_line(params: dict, seconds: float) -> dict | None:
    """The price line of a video conversion (with its size `factor`): `maybe` when it can't be told yet whether the picture is made again
    (the client did not say the video's codec; then its micros, what it would cost, are not part of the total); `copied`: the picture is
    copied as it is, free (micros 0)."""
    if not params.get("convert_rate"):
        return None
    rate = _convert_rate(params)
    micros = 0 if params.get("picture_encode") is False else pricing.cost_micros("convert", "standard", seconds, rate)
    return {"item": "convert", "label": pricing.rate("convert", "standard").label, "quantity": 1, "usdPerMinute": rate / pricing.MICROS_PER_USD,
            "factor": float(params.get("convert_factor") or 1.0), "micros": micros,
            **({"maybe": True} if params.get("picture_encode") is None else {}), **({"copied": True} if params.get("picture_encode") is False else {})}
_CODEC_RE = re.compile(r"^[a-z0-9_]{1,32}$")


def video_out_of(options: dict) -> dict | None:
    """options.video_out checked: {"format": one of VIDEO_OUT_FORMATS, "size": one of VIDEO_OUT_SIZES or None, "source": optional
    {"codec": FFmpeg's codec name, "width", "height", "fps"}: the video as the client read it, so the price is known before the upload}."""
    from . import jobs  # (jobs imports this module)

    vo = options.get("video_out")
    if vo is None:
        return None
    if not isinstance(vo, dict) or set(vo) - {"format", "size", "source"}:
        raise jobs.JobError("video_out must be {format, size, source?}")
    fmt = vo.get("format")
    if fmt not in VIDEO_OUT_FORMATS:
        raise jobs.JobError(f"video_out.format must be one of {', '.join(VIDEO_OUT_FORMATS)}")
    size = vo.get("size")
    if size is not None and (isinstance(size, bool) or size not in VIDEO_OUT_SIZES):
        raise jobs.JobError(f"video_out.size must be one of {', '.join(map(str, VIDEO_OUT_SIZES))}, or null for the original size")
    out: dict = {"format": fmt, "size": size}
    src = vo.get("source")
    if src is not None:
        if not isinstance(src, dict) or set(src) - {"codec", "width", "height", "fps"}:
            raise jobs.JobError("video_out.source must be {codec?, width?, height?, fps?}")
        clean: dict = {}
        if src.get("codec") is not None:
            if not isinstance(src["codec"], str) or not _CODEC_RE.fullmatch(src["codec"]):
                raise jobs.JobError("video_out.source.codec must be FFmpeg's codec name, like h264")
            clean["codec"] = src["codec"]
        for key, lo, hi in (("width", 16, 16384), ("height", 16, 16384), ("fps", 1, 240)):
            v = src.get(key)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
                raise jobs.JobError(f"video_out.source.{key} must be a number from {lo} to {hi}")
            clean[key] = int(v) if key != "fps" else float(v)
        if ("width" in clean) != ("height" in clean):
            raise jobs.JobError("video_out.source needs both width and height, or neither")
        out["source"] = clean
    return out


def convert_params(opts: dict, input_ext: str | None = None) -> dict:
    """What a job with a video conversion stores: the rate (pinned like every price), whether the picture is made again
    (`picture_encode`: a new size, or a file type that can't hold the video's codec; MKV holds any; None when the codec is not known)
    and the size factor (`convert_factor`), from what the client read of the video (video_out.source); the worker checks both when it
    reads the file. `input_ext`: the upload's ending (".mp4"), when known."""
    from . import jobs  # (jobs imports this module)

    from acw_common.ffmpeg import convert_factor, picture_copies, picture_size

    vo = opts.get("video_out")
    if not vo:
        return {}
    if input_ext in jobs.AUDIO_EXTS:  # a sound: its video gets a plain black picture, free
        return {"convert_rate": pricing.rate("convert", "standard").micros_per_minute, "picture_encode": False, "convert_factor": 1.0}
    src = vo.get("source") or {}
    w, h, fps = src.get("width"), src.get("height"), src.get("fps")
    resize = vo["size"] is not None and not (w and h and min(w, h) == vo["size"])
    copies = picture_copies(src.get("codec"), input_ext, "." + vo["format"])
    encode = True if resize else None if copies is None else not copies
    if w and h:
        factor = convert_factor(w, h, fps, picture_size(w, h, 1.0, 0, vo["size"]) if resize else None)
    elif vo["size"]:
        factor = convert_factor(*picture_size(16, 9, 1.0, 0, vo["size"]), fps)  # (the size of the original not said: 16:9 at the size asked)
    else:
        factor = 1.0
    return {"convert_rate": pricing.rate("convert", "standard").micros_per_minute, "picture_encode": encode, "convert_factor": factor}
