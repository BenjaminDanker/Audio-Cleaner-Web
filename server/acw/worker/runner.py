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
    from acw_common.ffmpeg import FFmpeg

    return FFmpeg(Path(spec["bin_dir"]), threads=int(spec.get("threads", 2)), force_demuxer=True)


def stage_probe(job_dir: Path, spec: dict) -> dict:
    from acw_common.ffmpeg import MediaError

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
    return {
        "duration": duration,
        "container_duration": p.duration,
        "has_video": p.has_video,
        "channels": p.audio_channels,
        "format": p.format_name,
        "audio_codec": p.audio_codec,
        "video_codec": p.video_codec,
    }


def decode_limit(spec: dict) -> float:
    """Seconds of audio any decode may read: the length that was billed (the worker's measurement), with
    a little slack for resampling, never more than the tier's limit."""
    billed = spec.get("billed_seconds")
    return spec["max_seconds"] if billed is None else min(spec["max_seconds"], billed * 1.02 + 2)


def stage_main(job_dir: Path, spec: dict) -> dict:
    from acw_common import dpdfnet, pipeline

    ff = _ff(spec)
    limit = decode_limit(spec)
    src = job_dir / spec["input"]
    probe = ff.probe(src)
    opts = spec["options"]
    stem = spec["output_stem"]
    threads = int(spec.get("threads", 2))
    outputs: list[dict] = []
    denoise = spec["kind"] in ("denoise", "denoise_transcribe")
    transcribe = spec["kind"] in ("transcribe", "denoise_transcribe")
    t0 = time.time()

    needs_cloud = False
    cleaned_path = None
    notice = None
    noise_used = None  # which noise remover ran ("light" | "heavy"), for the record
    asr_end = 45 if denoise else 90  # progress when transcription is done
    x = None  # the decoded audio, echo-free when the second track was used
    if opts.get("echo_reference_track"):
        if probe.audio_tracks < 2:
            notice = "This file has only one audio track, so there was no desktop sound to remove echo with. It was processed without echo removal."
        else:
            # Before anything else: both the captions and the cleaned audio should start from the echo-free voice.
            emit(stage="Removing echo", progress=2)
            mic = pipeline.decode_for_clean(ff, src, probe, limit, work_dir=job_dir)
            x = pipeline.remove_echo(
                ff, src, probe, mic, limit, work_dir=job_dir,
                progress=lambda f: emit(stage="Removing echo", progress=int(2 + 3 * f)),
            )
            del mic
    if transcribe:
        # Transcribe first, from the original audio: every recogniser we tested
        # scores worse on denoised speech (docs/research/asr.md).
        emit(stage="Decoding", progress=5 if x is not None else 2)
        if x is not None:
            speech_16k = pipeline.to_speech_16k(x)
        else:
            speech_16k = ff.decode(src, probe, 16000, 1, limit)[:, 0]
        if spec["transcribe_tier"] == "premium":
            emit(stage="Preparing", progress=5)
            _write_asr_chunks(ff, job_dir, speech_16k)
            needs_cloud = True
        else:
            from acw_common import transcribe

            base = 5
            emit(stage="Transcribing", progress=base)
            models = Path(spec["models_dir"])
            lang = opts.get("language")
            if lang is None:
                emit(stage="Detecting language", progress=base)
                lang = transcribe.detect_language(speech_16k, models / "whisper" / "small", threads)
            engine = transcribe.pick_engine(lang)
            report = lambda f: emit(stage="Transcribing", progress=int(base + (asr_end - base) * f))  # noqa: E731
            if engine.startswith("parakeet"):
                segs = transcribe.transcribe_parakeet(
                    speech_16k, models / engine, models / "silero-vad",
                    device=spec.get("asr_device", "cpu"), threads=threads, progress=report,
                )
            else:
                segs, lang = transcribe.transcribe(
                    speech_16k, models / "whisper" / spec.get("whisper_model", "large-v3-turbo"),
                    language=lang, threads=threads, progress=report,
                )
            from acw_common.captions import segments_to_json

            (job_dir / "segments.json").write_text(
                json.dumps({"language": lang, "engine": "standard", "segments": segments_to_json(segs)})
            )
    if denoise:
        start = asr_end if transcribe else (5 if x is not None else 2)
        emit(stage="Decoding", progress=start)
        # Decoded and cleaned audio live in memory-mapped files in the job
        # directory, so RAM use doesn't grow with the length of the upload.
        if x is None:
            x = pipeline.decode_for_clean(ff, src, probe, limit, work_dir=job_dir)
        span = 85 - start
        rebuild = spec.get("denoise_tier") == "rebuild"
        # The noise remover is chosen here (measured on the echo-free recording), so the worker can base its
        # time estimate on it; clean_audio then uses the same choice.
        profile = opts.get("noise", "auto") if rebuild else pipeline.choose_noise_profile(x, opts.get("noise", "auto"))
        emit(stage="Cleaning", progress=start + 2, **({} if rebuild else {"noise": profile}))
        used: dict = {}
        y = pipeline.clean_audio(
            x, spec["lib_path"], spec["dfn_model"],
            pipeline.CleanOptions(
                atten_lim_db=float(opts.get("atten_lim_db", 100.0)),
                loudness_lufs=opts.get("loudness_lufs", -16.0),
                noise=profile,
                restore_voice=rebuild,
            ),
            threads=threads,
            progress=lambda f: emit(stage="Cleaning" if f < 0.95 else "Polishing", progress=int(start + 2 + span * f)),
            out=pipeline.output_buffer(x, job_dir),
            heavy_model_path=Path(spec["models_dir"]) / "dpdfnet" / dpdfnet.MODEL_FILE,
            bwe_model_dir=Path(spec["models_dir"]) / "ap-bwe",
            restore_model_dir=Path(spec["models_dir"]) / "sidon",
            report=used,
        )
        noise_used = used.get("noise")
        if used.get("highs"):
            restored = "This sounded like a phone or low-quality recording, so its missing high frequencies were restored."
            notice = f"{notice} {restored}" if notice else restored
        fmt = opts.get("audio_format", "wav")
        cleaned_path = ff.write_audio(pipeline.as_output(y), pipeline.SAMPLE_RATE, job_dir / f"{stem}-cleaned", fmt)
        outputs.append({"file": cleaned_path.name, "kind": "audio"})
        del x, y
        for scratch in ("decoded.f32", "reference.f32", "echo-free.npy", "cleaned.npy"):
            (job_dir / scratch).unlink(missing_ok=True)

    (job_dir / "main-state.json").write_text(
        json.dumps({"outputs": outputs, "cleaned": cleaned_path.name if cleaned_path else None, "has_video": probe.has_video})
    )
    # Translations need the internet, which this sandbox doesn't have: they run outside it, then `finalize` writes the files.
    needs_cloud = needs_cloud or bool(opts.get("translate"))
    result = {"needs_cloud_asr": needs_cloud, "duration": probe.duration, "seconds_spent": time.time() - t0, "notice": notice,
              "noise": noise_used}
    if not needs_cloud:
        result.update(stage_finalize(job_dir, spec))
    return result


# Premium transcription returns plain text per request, so subtitle timing comes
# from these speech segments: short enough that spreading a segment's time over
# its words (captions.build_cues) stays close to the real timing.
ASR_SEGMENT_MAX_S = 15.0
ASR_MERGE_GAP_S = 0.4


def _write_asr_chunks(ff, job_dir: Path, x16) -> None:
    """Cut 16 kHz audio into speech segments of at most ASR_SEGMENT_MAX_S
    (Silero VAD, bundled with faster-whisper) and encode each as mp3."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    chunk_dir = job_dir / "asr_chunks"
    chunk_dir.mkdir(exist_ok=True)
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
    chunks = []
    for i, (a, b) in enumerate(merged):
        ff.write_audio(x16[a:b], sr, chunk_dir / f"{i:04d}", "mp3")
        chunks.append({"file": f"{i:04d}.mp3", "offset": a / sr, "seconds": (b - a) / sr})
    (chunk_dir / "index.json").write_text(json.dumps({"chunks": chunks}))


def stage_finalize(job_dir: Path, spec: dict) -> dict:
    from acw_common import captions

    ff = _ff(spec)
    opts = spec["options"]
    stem = spec["output_stem"]
    state = json.loads((job_dir / "main-state.json").read_text())
    outputs = list(state["outputs"])
    subtitle_file = None
    language = None
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

    if state["has_video"] and opts.get("output_video", True):
        cleaned = job_dir / state["cleaned"] if state.get("cleaned") else None
        if cleaned is not None or (subtitle_file is not None and opts.get("embed_subtitles", True)):
            emit(stage="Rebuilding video", progress=95)
            # Other containers (AVI, WMV, FLV, MPEG, ...) go to MKV: it holds any video codec we copy.
            ext = spec["input_ext"] if spec["input_ext"] in (".mp4", ".mov", ".m4v", ".mkv", ".webm") else ".mkv"
            out = job_dir / f"{stem}-{'cleaned' if cleaned else 'captioned'}{ext}"
            ff.mux_video(
                job_dir / spec["input"],
                cleaned,
                out,
                subtitles=subtitle_file if opts.get("embed_subtitles", True) else None,
                sub_lang=_iso639_2(language),
            )
            outputs.append({"file": out.name, "kind": "video"})
    emit(stage="Done", progress=100)
    return {"outputs": outputs, "language": language}


_ISO = {"en": "eng", "es": "spa", "fr": "fra", "de": "deu", "it": "ita", "pt": "por", "nl": "nld", "ja": "jpn",
        "zh": "zho", "ko": "kor", "ru": "rus", "pl": "pol", "tr": "tur", "ar": "ara", "hi": "hin", "sv": "swe"}


def _iso639_2(code: str | None) -> str | None:
    return _ISO.get((code or "")[:2]) if code else None


if __name__ == "__main__":
    sys.exit(main())
