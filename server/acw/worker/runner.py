"""Sandboxed job runner: `python -m acw.worker.runner <job_dir> <stage>`.

Confines itself (see sandbox.py) before touching the upload, then works only
inside the job directory. Talks to the parent worker through files
(spec.json in, result-<stage>.json out) and JSON progress lines on stdout.
It has no access to the database, secrets or the network.

Stages:
  probe     inspect the upload, report duration/streams (for billing)
  main      make the job's timeline: the subtitle parts (locally, or chunks
            for cloud ASR), the sound steps on their parts, write outputs
  finalize  build caption files from segments.json and mux them (used after
            cloud ASR, or directly from main)
  edit      an edit run of the per-part editor: make the missing blocks of a timeline, render it into the result audio, or find
            suggestions (editrun.py; acw/editing.py describes the whole thing)
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

STAGES = ("probe", "analyse", "main", "finalize", "edit")


def emit(**kw) -> None:
    sys.stdout.write(json.dumps(kw) + "\n")
    sys.stdout.flush()


def write_result(job_dir: Path, stage: str, result: dict) -> None:
    tmp = job_dir / f"result-{stage}.json.tmp"
    tmp.write_text(json.dumps(result))
    tmp.replace(job_dir / f"result-{stage}.json")


def main() -> int:
    job_dir = Path(sys.argv[1]).resolve()
    stage = sys.argv[2]
    if stage not in STAGES:
        return 2
    spec = json.loads((job_dir / "spec.json").read_text())

    from acw.worker.sandbox import confine

    confine(
        job_dir,
        [spec["bin_dir"], spec["models_dir"], spec["lib_path"], *spec.get("gpu_read", [])],
        cpu_seconds=int(spec["cpu_seconds"]),
        address_space_bytes=int(spec["address_space_bytes"]),
        max_file_bytes=int(spec["max_file_bytes"]),
        extra_write=spec.get("gpu_devices", []),
        unix_sockets=bool(spec.get("gpu_devices")),
    )
    os.chdir(job_dir)
    os.environ["OMP_NUM_THREADS"] = str(spec.get("threads", 2))

    from acw_common.ffmpeg import MediaError

    try:
        if stage == "probe":
            result = stage_probe(job_dir, spec)
        elif stage == "analyse":
            result = stage_analyse(job_dir, spec)
        elif stage == "main":
            result = stage_main(job_dir, spec)
        elif stage == "edit":
            from acw.worker import editrun

            result = editrun.stage_edit(job_dir, spec, _ff(spec), emit)
        else:
            result = stage_finalize(job_dir, spec)
        result["ok"] = True
    except MediaError as e:
        result = {"ok": False, "user_error": str(e)}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        result = {"ok": False, "user_error": None, "error": f"{type(e).__name__}: {e}"[:500]}
    write_result(job_dir, stage, result)
    return 0


# ---------------------------------------------------------------- stages

def _ff(spec):
    from acw_common.ffmpeg import FFmpeg

    return FFmpeg(Path(spec["bin_dir"]), threads=int(spec.get("threads", 2)), force_demuxer=True)


def stage_probe(job_dir: Path, spec: dict) -> dict:
    from acw_common.ffmpeg import MediaError, VideoOut, picture_work

    ff = _ff(spec)
    p = ff.probe(job_dir / spec["input"])
    limit = spec["max_seconds"]
    # The header's length is only what the file claims; billing and limits use what actually decodes.
    # Reading stops a minute past the limit, which is enough to tell the file is too long.
    measured = ff.measure_duration(job_dir / spec["input"], p, limit + 60)
    duration = max(p.duration, measured)
    if duration > limit + 1:
        if spec.get("denoise_tier") == "rebuild":
            raise MediaError(
                f"Voice rebuilding takes files up to {limit / 60:.0f} minutes; this one is longer. "
                "Use the standard cleanup for longer files."
            )
        raise MediaError(f"This file is {duration / 60:.1f} minutes long; the limit is {limit / 60:.0f} minutes.")
    # A video asked for in another file type or size: is its picture made again (charged, by its size factor), or copied (free)?
    vo = spec["options"].get("video_out")
    picture_encode, factor = picture_work(p, "." + vo["format"], VideoOut(short_side=vo["size"])) if vo else (None, None)
    return {
        "audio_tracks": p.audio_tracks,
        "picture_encode": picture_encode,
        "convert_factor": factor,
        "duration": duration,
        "container_duration": p.duration,
        "has_video": p.has_video,
        "channels": p.audio_channels,
        "format": p.format_name,
        "audio_codec": p.audio_codec,
        "video_codec": p.video_codec,
    }


def stage_analyse(job_dir: Path, spec: dict) -> dict:
    """Measure the upload the way the website does (acw_common/analysis.py: the noise remover over about a minute of it spread over
    the file, and the browser's measurements of the same stretches), for the settings of an automatic job (automatic.py). Only the
    numbers go back; the decision is made outside the sandbox."""
    import math

    from acw_common import analysis, pipeline

    ff = _ff(spec)
    src = job_dir / spec["input"]
    probe = ff.probe(src)
    emit(stage="Listening to the file", progress=2)
    # (before any echo removal: the second track is not part of what is measured)
    x = pipeline.decode_for_clean(ff, src, probe, decode_limit(spec), work_dir=job_dir)
    stats = analysis.measure(x, spec["lib_path"], spec["dfn_model"])
    del x
    (job_dir / "decoded.f32").unlink(missing_ok=True)
    clean = {k: (v if isinstance(v, bool) or v is None or (isinstance(v, (int, float)) and math.isfinite(v)) else None)
             for k, v in stats.items() if isinstance(v, (bool, int, float)) or v is None}
    return {"stats": {k: (float(v) if isinstance(v, float) else v) for k, v in clean.items()}}


def decode_limit(spec: dict) -> float:
    """Seconds of audio any decode may read: the length that was billed (the worker's measurement), with
    a little slack for resampling, never more than the tier's limit."""
    billed = spec.get("billed_seconds")
    return spec["max_seconds"] if billed is None else min(spec["max_seconds"], billed * 1.02 + 2)


def stage_main(job_dir: Path, spec: dict) -> dict:
    """Every job is made one way, from its timeline (editrun.py): the subtitles part by part, then the sound steps on their parts (the block
    cache is filled from nothing; a later edit runs only what is missing). A conversion has nothing to do here: its video is made in
    `stage_finalize`."""
    from acw_common import editor, pipeline

    ff = _ff(spec)
    limit = decode_limit(spec)
    src = job_dir / spec["input"]
    probe = ff.probe(src)
    opts = spec["options"]
    stem = spec["output_stem"]
    outputs: list[dict] = []
    ed_spec = spec.get("editor") or {}  # the per-part editor (editrun.py): the timeline, the block cache, the recording's facts, the suggestions
    if spec["kind"] != "convert" and not ed_spec.get("timeline"):
        raise ValueError("a job without a timeline")  # (every job but a conversion is made with one: editing.timeline_from_options)
    parts_mode = ed_spec.get("subtitles") is not None  # subtitle parts (editrun.subtitles_prepare)
    sound = bool(ed_spec.get("timeline")) and ed_spec.get("audio", True)  # (a timeline of subtitles only renders no audio)
    t0 = time.time()
    editor_report: dict = {}
    needs_cloud = False
    cleaned_path = None
    notice = None
    x = None  # the decoded recording
    asr_end = 45 if sound else 90  # progress when the subtitles are done
    if parts_mode:
        from acw.worker import editrun

        emit(stage="Decoding", progress=2)
        if sound:
            x = pipeline.decode_for_clean(ff, src, probe, limit, work_dir=job_dir)  # (decoded once, for the subtitles and the sound)
        needs_cloud = editrun.subtitles_prepare(job_dir, spec, ff, probe, x, limit, emit, base=5, span=asr_end - 5)
    if sound:
        from acw.worker import editrun

        start = asr_end if parts_mode else 2
        emit(stage="Decoding", progress=start)
        # Decoded and cleaned audio live in memory-mapped files in the job directory, so RAM use doesn't grow with the length of the upload.
        if x is None:
            x = pipeline.decode_for_clean(ff, src, probe, limit, work_dir=job_dir)
        y, editor_report, notice = editrun.render(job_dir, spec, ff, probe, x, emit, base=start + 2, span=85 - start, op="first")
        editor_report["facts"] = editrun.facts_of(editor.Inputs(x, ed_spec["source_id"]), ff, probe, job_dir)
        if ed_spec.get("suggest"):
            editor_report["suggestions"] = editrun.find_suggestions(x, spec, emit, 86, 6)
        cleaned_path = ff.write_audio(pipeline.as_output(y), pipeline.SAMPLE_RATE, job_dir / f"{stem}-cleaned", opts.get("audio_format", "flac"))
        outputs.append({"file": cleaned_path.name, "kind": "audio"})
        del x, y
        for scratch in ("decoded.f32", "reference.f32", "echo-free.npy", "cleaned.npy"):
            (job_dir / scratch).unlink(missing_ok=True)
    elif ed_spec.get("timeline"):  # a timeline of subtitles only: the sound was not rendered, so the result files hold the recording as it is
        from acw_common.timeline import Timeline

        editor_report.update(rendered=Timeline(probe.duration).to_json(), billed={})
    (job_dir / "main-state.json").write_text(
        json.dumps({"outputs": outputs, "cleaned": cleaned_path.name if cleaned_path else None, "has_video": probe.has_video, "duration": probe.duration})
    )
    # Accurate subtitles and translations need the internet, which this sandbox doesn't have: they are made outside it, then `finalize`
    # writes the files.
    result = {"needs_cloud_asr": needs_cloud, "duration": probe.duration, "seconds_spent": time.time() - t0, "notice": notice,
              "noise": None, "editor": editor_report}
    if not needs_cloud:
        result.update(stage_finalize(job_dir, spec))
    return result


# Premium transcription returns plain text per request, so subtitle timing comes
# from these speech segments: short enough that spreading a segment's time over
# its words (captions.build_cues) stays close to the real timing.
ASR_SEGMENT_MAX_S = 15.0
ASR_MERGE_GAP_S = 0.4


def _write_asr_chunks(ff, job_dir: Path, x16, offset: float = 0.0, window: tuple[float, float] | None = None) -> None:
    """Cut 16 kHz audio into speech segments of at most ASR_SEGMENT_MAX_S
    (Silero VAD, bundled with faster-whisper) and encode each as mp3 (64 kbps, a few chunks per FFmpeg run).
    `offset`: the second of the recording `x16` begins at (the chunks' times are in the recording's time); `window`: (from, to) seconds
    of `x16` that belong to the part being transcribed (the rest is only context for the speech detector): the chunks are cut to it
    (a piece under a quarter of a second is dropped), so no sound outside it is sent for transcription."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    chunk_dir = job_dir / "asr_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    sr = 16000
    spans = get_speech_timestamps(
        x16,
        VadOptions(min_speech_duration_ms=250, min_silence_duration_ms=300, speech_pad_ms=150,
                   max_speech_duration_s=ASR_SEGMENT_MAX_S),
    )
    merged: list[list[int]] = []
    for span in spans:
        a, b = int(span["start"]), int(span["end"])
        if merged and a - merged[-1][1] < ASR_MERGE_GAP_S * sr and b - merged[-1][0] <= ASR_SEGMENT_MAX_S * sr:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    if window is not None:
        lo, hi = int(window[0] * sr), int(window[1] * sr)
        merged = [[max(a, lo), min(b, hi)] for a, b in merged if min(b, hi) - max(a, lo) >= sr // 4]
    names = [f"{i:04d}" for i in range(len(merged))]
    ff.write_audio_chunks(x16, sr, [(a, b) for a, b in merged], chunk_dir, names)
    chunks = [{"file": f"{n}.mp3", "offset": offset + a / sr, "seconds": (b - a) / sr} for n, (a, b) in zip(names, merged)]
    (chunk_dir / "index.json").write_text(json.dumps({"chunks": chunks}))


def stage_finalize(job_dir: Path, spec: dict) -> dict:
    from acw_common import captions
    from acw_common.ffmpeg import VideoOut

    ff = _ff(spec)
    opts = spec["options"]
    stem = spec["output_stem"]
    state = json.loads((job_dir / "main-state.json").read_text())
    outputs = list(state["outputs"])
    subtitle_file = None
    language = None
    if (spec.get("editor") or {}).get("subtitles") is not None:  # subtitles made part by part: put the parts together first
        from acw.worker import editrun

        emit(stage="Putting the subtitles together", progress=91)
        editrun.merge_subtitles(job_dir, spec)
    seg_path = job_dir / "segments.json"
    if seg_path.exists():
        emit(stage="Writing captions", progress=92)
        data = json.loads(seg_path.read_text())
        language = data.get("language")
        segs = captions.segments_from_json(data["segments"])
        cues = captions.build_cues(segs)
        srt = job_dir / f"{stem}.srt"
        srt.write_text(captions.to_srt(cues), encoding="utf-8")
        subtitle_file = srt
        formats = opts.get("caption_formats", ["srt", "vtt", "txt"])
        if "srt" in formats:
            outputs.append({"file": srt.name, "kind": "captions"})
        if "vtt" in formats:
            vtt = job_dir / f"{stem}.vtt"
            vtt.write_text(captions.to_vtt(cues), encoding="utf-8")
            outputs.append({"file": vtt.name, "kind": "captions"})
        if "txt" in formats:
            txt = job_dir / f"{stem}-transcript.txt"
            txt.write_text(captions.to_text(segs), encoding="utf-8")
            outputs.append({"file": txt.name, "kind": "transcript"})
        if "json" in formats:
            outputs.append({"file": seg_path.name, "kind": "segments"})
        for o in outputs:
            if o["kind"] in ("captions", "transcript", "segments") and language:
                o["lang"] = language[:2]
        # Translations (made outside the sandbox from segments.json): the same files, one set per language.
        for lang in opts.get("translate") or []:
            tpath = job_dir / f"segments.{lang}.json"
            if not tpath.is_file() or tpath.is_symlink():
                continue
            tsegs = captions.segments_from_json(json.loads(tpath.read_text(encoding="utf-8"))["segments"])
            tcues = captions.build_cues(tsegs, lang)
            if "srt" in formats:
                p = job_dir / f"{stem}.{lang}.srt"
                p.write_text(captions.to_srt(tcues), encoding="utf-8")
                outputs.append({"file": p.name, "kind": "captions", "lang": lang})
            if "vtt" in formats:
                p = job_dir / f"{stem}.{lang}.vtt"
                p.write_text(captions.to_vtt(tcues), encoding="utf-8")
                outputs.append({"file": p.name, "kind": "captions", "lang": lang})
            if "txt" in formats:
                p = job_dir / f"{stem}.{lang}-transcript.txt"
                p.write_text(captions.to_text(tsegs), encoding="utf-8")
                outputs.append({"file": p.name, "kind": "transcript", "lang": lang})

    # An edit run (editrun.stage_edit) makes the video again under a temporary name, which takes the old one's place when the run is complete
    # (acw/editing.py swap_in); and with only a new sound its subtitles are the job's as they are.
    staged = (spec.get("editor") or {}).get("op") not in (None, "first")
    wants_video = state.get("video", True)
    if subtitle_file is None and staged and wants_video and opts.get("embed_subtitles", True) and (job_dir / f"{stem}.srt").is_file():
        subtitle_file = job_dir / f"{stem}.srt"

    def made(out: Path) -> Path:
        return out.replace(out.with_name(f"{stem}-{out.name.split('-', 1)[1]}.run")) if staged else out

    if state["has_video"] and opts.get("output_video", True) and wants_video:
        cleaned = job_dir / state["cleaned"] if state.get("cleaned") else None
        vo = opts.get("video_out")  # another file type or picture size, asked for
        if cleaned is not None or (subtitle_file is not None and opts.get("embed_subtitles", True)) or vo:
            # (a conversion alone is all of the job's work)
            first = 95 if cleaned is not None or subtitle_file is not None else 5
            emit(stage="Making the video", progress=first)
            if vo:
                ext = "." + vo["format"]
            else:
                # Other containers (AVI, WMV, FLV, MPEG, ...) go to MKV: it holds any video codec we copy.
                ext = spec["input_ext"] if spec["input_ext"] in (".mp4", ".mov", ".m4v", ".mkv", ".webm") else ".mkv"
            out = job_dir / f"{'run' if staged else stem}-{'cleaned' if cleaned else 'captioned' if subtitle_file is not None else 'converted'}{ext}"
            ff.mux_video(
                job_dir / spec["input"],
                cleaned,
                out,
                subtitles=subtitle_file if opts.get("embed_subtitles", True) else None,
                sub_lang=_iso639_2(language),
                video=VideoOut(short_side=vo["size"], encoders=_encoders(spec, ext), fast=True) if vo else None,
                progress=(lambda f: emit(stage="Making the video", progress=int(first + (99 - first) * f))) if vo else None,
            )
            outputs.append({"file": made(out).name, "kind": "video"})
    elif not state["has_video"] and opts.get("video_out") and wants_video:
        # a sound asked for as a video: a plain black picture (one frame a second) with the sound, cleaned when it was cleaned
        vo = opts["video_out"]
        cleaned = job_dir / state["cleaned"] if state.get("cleaned") else None
        ext = "." + vo["format"]
        emit(stage="Making the video", progress=95 if cleaned is not None or subtitle_file is not None else 5)
        out = job_dir / f"{'run' if staged else stem}-{'cleaned' if cleaned else 'captioned' if subtitle_file is not None else 'converted'}{ext}"
        ff.make_video(cleaned or job_dir / spec["input"], out, float(state.get("duration") or 0.0),
                      subtitles=subtitle_file if opts.get("embed_subtitles", True) else None, sub_lang=_iso639_2(language),
                      video=VideoOut(short_side=vo["size"], encoders=_encoders(spec, ext)))
        outputs.append({"file": made(out).name, "kind": "video"})
    emit(stage="Done", progress=100)
    return {"outputs": outputs, "language": language}


def _encoders(spec: dict, ext: str) -> tuple[str, ...]:
    """The picture encoders for a video result, best first: on the card's worker its NVENC (x264 if it refuses), else x264; WebM gets VP9
    at its real-time speed (the card's encoder has no VP9)."""
    if ext == ".webm":
        return ("libvpx-vp9",)
    return ("h264_nvenc", "libx264") if spec.get("gpu_devices") else ("libx264",)


_ISO = {"en": "eng", "es": "spa", "fr": "fra", "de": "deu", "it": "ita", "pt": "por", "nl": "nld", "ja": "jpn",
        "zh": "zho", "ko": "kor", "ru": "rus", "pl": "pol", "tr": "tur", "ar": "ara", "hi": "hin", "sv": "swe"}


def _iso639_2(code: str | None) -> str | None:
    return _ISO.get((code or "")[:2]) if code else None


if __name__ == "__main__":
    sys.exit(main())
