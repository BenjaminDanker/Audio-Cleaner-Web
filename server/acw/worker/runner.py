"""Sandboxed job runner: `python -m acw.worker.runner <job_dir> <stage>`.

Confines itself (see sandbox.py) before touching the upload, then works only
inside the job directory. Talks to the parent worker through files
(spec.json in, result-<stage>.json out) and JSON progress lines on stdout.
It has no access to the database, secrets or the network.

Stages:
  probe     inspect the upload, report duration/streams (for billing)
  main      decode, denoise + clarity, transcribe locally or prepare chunks
            for cloud ASR, write outputs
  finalize  build caption files from segments.json and mux them (used after
            cloud ASR, or directly from main)
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

STAGES = ("probe", "main", "finalize")


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
        [spec["bin_dir"], spec["models_dir"], spec["lib_path"]],
        cpu_seconds=int(spec["cpu_seconds"]),
        address_space_bytes=int(spec["address_space_bytes"]),
        max_file_bytes=int(spec["max_file_bytes"]),
    )
    os.chdir(job_dir)
    os.environ["OMP_NUM_THREADS"] = str(spec.get("threads", 2))

    from acw.media.ffmpeg import MediaError

    try:
        if stage == "probe":
            result = stage_probe(job_dir, spec)
        elif stage == "main":
            result = stage_main(job_dir, spec)
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
    from acw.media.ffmpeg import FFmpeg

    return FFmpeg(Path(spec["bin_dir"]), threads=int(spec.get("threads", 2)))


def stage_probe(job_dir: Path, spec: dict) -> dict:
    p = _ff(spec).probe(job_dir / spec["input"])
    if p.duration > spec["max_seconds"] + 1:
        from acw.media.ffmpeg import MediaError

        raise MediaError(f"This file is {p.duration / 60:.1f} minutes long; the limit is {spec['max_seconds'] / 60:.0f} minutes.")
    return {
        "duration": p.duration,
        "has_video": p.has_video,
        "channels": p.audio_channels,
        "format": p.format_name,
        "audio_codec": p.audio_codec,
        "video_codec": p.video_codec,
    }


def stage_main(job_dir: Path, spec: dict) -> dict:
    import numpy as np

    from acw.dsp import clarity, dfn

    ff = _ff(spec)
    src = job_dir / spec["input"]
    probe = ff.probe(src)
    opts = spec["options"]
    stem = spec["output_stem"]
    threads = int(spec.get("threads", 2))
    outputs: list[dict] = []
    denoise = spec["kind"] in ("denoise", "denoise_transcribe")
    transcribe = spec["kind"] in ("transcribe", "denoise_transcribe")
    t0 = time.time()

    cleaned_path = None
    speech_16k = None
    if denoise:
        channels = min(probe.audio_channels, 2)
        emit(stage="decoding", progress=2)
        x = ff.decode(src, probe, dfn.SAMPLE_RATE, channels, spec["max_seconds"])
        emit(stage="denoising", progress=5)
        lib, model = spec["lib_path"], spec["dfn_model"]
        span = 70 if not transcribe else 45
        per_ch = []
        for ch in range(channels):
            def prog(f, ch=ch):
                emit(stage="denoising", progress=int(5 + span * (ch + f) / channels))

            per_ch.append(
                dfn.denoise_array(
                    x[:, ch], lib, model,
                    atten_lim_db=float(opts.get("atten_lim_db", 100.0)),
                    threads=threads, progress=prog,
                )
            )
        y = np.stack(per_ch, axis=1)
        emit(stage="polishing", progress=5 + span)
        params = clarity.ClarityParams(loudness_lufs=opts.get("loudness_lufs", -16.0))
        y = clarity.process(x, y, dfn.SAMPLE_RATE, params)
        fmt = opts.get("audio_format", "wav")
        cleaned_path = ff.write_audio(y if channels > 1 else y[:, 0], dfn.SAMPLE_RATE, job_dir / f"{stem}-cleaned", fmt)
        outputs.append({"file": cleaned_path.name, "kind": "audio"})
        if transcribe:
            from scipy.signal import resample_poly

            speech_16k = resample_poly(y.mean(axis=1), 1, 3).astype(np.float32)
        del x, y

    needs_cloud = False
    if transcribe:
        if speech_16k is None:
            emit(stage="decoding", progress=3)
            speech_16k = ff.decode(src, probe, 16000, 1, spec["max_seconds"])[:, 0]
        if spec["transcribe_tier"] == "premium":
            emit(stage="preparing transcription", progress=80)
            _write_asr_chunks(ff, job_dir, speech_16k)
            needs_cloud = True
        else:
            from acw.asr import local

            base = 50 if denoise else 5
            emit(stage="transcribing", progress=base)
            segs, lang = local.transcribe(
                speech_16k, Path(spec["models_dir"]),
                model_name=spec.get("whisper_model", local.DEFAULT_MODEL),
                language=opts.get("language"), threads=threads,
                progress=lambda f: emit(stage="transcribing", progress=int(base + (90 - base) * f)),
            )
            from acw.captions import segments_to_json

            (job_dir / "segments.json").write_text(
                json.dumps({"language": lang, "engine": "faster-whisper", "segments": segments_to_json(segs)})
            )
    (job_dir / "main-state.json").write_text(
        json.dumps({"outputs": outputs, "cleaned": cleaned_path.name if cleaned_path else None, "has_video": probe.has_video})
    )
    result = {"needs_cloud_asr": needs_cloud, "duration": probe.duration, "seconds_spent": time.time() - t0}
    if not needs_cloud:
        result.update(stage_finalize(job_dir, spec))
    return result


def _write_asr_chunks(ff, job_dir: Path, x16) -> None:
    """Split 16 kHz speech into <=10 min pieces at quiet points, encode as mp3."""
    import numpy as np

    chunk_dir = job_dir / "asr_chunks"
    chunk_dir.mkdir(exist_ok=True)
    sr, target, search = 16000, 600 * 16000, 20 * 16000
    win = int(0.3 * sr)
    cuts, pos = [0], 0
    while len(x16) - pos > target:
        lo, hi = pos + target - search, min(pos + target, len(x16) - win)
        seg = x16[lo:hi + win] ** 2
        energy = np.convolve(seg, np.ones(win, dtype=np.float32), mode="valid")[: hi - lo]
        cut = lo + int(np.argmin(energy)) + win // 2
        cuts.append(cut)
        pos = cut
    cuts.append(len(x16))
    chunks = []
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        name = f"{i:03d}.mp3"
        ff.write_audio(x16[a:b], sr, chunk_dir / f"{i:03d}", "mp3")
        chunks.append({"file": name, "offset": a / sr, "seconds": (b - a) / sr})
    (chunk_dir / "index.json").write_text(json.dumps({"chunks": chunks}))


def stage_finalize(job_dir: Path, spec: dict) -> dict:
    from acw import captions

    ff = _ff(spec)
    opts = spec["options"]
    stem = spec["output_stem"]
    state = json.loads((job_dir / "main-state.json").read_text())
    outputs = list(state["outputs"])
    subtitle_file = None
    language = None
    seg_path = job_dir / "segments.json"
    if seg_path.exists():
        emit(stage="writing captions", progress=92)
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

    if state["has_video"] and opts.get("output_video", True):
        cleaned = job_dir / state["cleaned"] if state.get("cleaned") else None
        if cleaned is not None or (subtitle_file is not None and opts.get("embed_subtitles", True)):
            emit(stage="rebuilding video", progress=95)
            ext = spec["input_ext"] if spec["input_ext"] in (".mp4", ".mov", ".m4v", ".mkv", ".webm") else ".mp4"
            out = job_dir / f"{stem}-{'cleaned' if cleaned else 'captioned'}{ext}"
            ff.mux_video(
                job_dir / spec["input"],
                cleaned,
                out,
                subtitles=subtitle_file if opts.get("embed_subtitles", True) else None,
                sub_lang=_iso639_2(language),
            )
            outputs.append({"file": out.name, "kind": "video"})
    emit(stage="done", progress=100)
    return {"outputs": outputs, "language": language}


_ISO = {"en": "eng", "es": "spa", "fr": "fra", "de": "deu", "it": "ita", "pt": "por", "nl": "nld", "ja": "jpn",
        "zh": "zho", "ko": "kor", "ru": "rus", "pl": "pol", "tr": "tur", "ar": "ara", "hi": "hin", "sv": "swe"}


def _iso639_2(code: str | None) -> str | None:
    return _ISO.get((code or "")[:2]) if code else None


if __name__ == "__main__":
    sys.exit(main())
