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
from typing import Any

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


# The names of the copies a job keeps (jobs.KEPT_NOISE, jobs.KEPT_REBUILT: this module must not import the database code).
KEPT_NOISE = "kept-noise"
KEPT_REBUILT = "kept-rebuilt"
KEPT_INPUT = "kept-input"


def clean_options(spec: dict, profile: str):
    """The cleaning choices of a job (pipeline.CleanOptions) from its spec: its options and tier, with the noise remover `profile`."""
    from acw_common import pipeline

    opts = spec["options"]
    return pipeline.CleanOptions(
        atten_lim_db=float(opts.get("atten_lim_db", 100.0)),
        loudness_lufs=opts.get("loudness_lufs", -16.0),
        leveling=bool(opts.get("leveling", True)),
        noise=profile,
        restore_voice=spec.get("denoise_tier") == "rebuild",
        denoise=spec.get("denoise_tier") != "off",  # "off": Leveling on its own, no noise removal
        device=spec.get("clean_device", "cpu"),
    )


def stage_main(job_dir: Path, spec: dict) -> dict:
    from acw_common import dpdfnet, editor, pipeline
    from acw_common import kept as kept_copies

    ff = _ff(spec)
    limit = decode_limit(spec)
    src = job_dir / spec["input"]
    probe = ff.probe(src)
    opts = spec["options"]
    stem = spec["output_stem"]
    threads = int(spec.get("threads", 2))
    outputs: list[dict] = []
    ed_spec = spec.get("editor")  # the per-part editor (editrun.py): the block cache, the recording's facts, the suggestions
    # A job whose timeline has subtitle parts makes its subtitles part by part (editrun.subtitles_prepare), not for the whole file; and one
    # whose timeline has no sound parts (subtitles only) renders no audio.
    parts_mode = bool(ed_spec) and ed_spec.get("subtitles") is not None
    no_audio = bool(ed_spec) and not ed_spec.get("audio", True)
    denoise = spec["kind"] in ("denoise", "denoise_transcribe") and not no_audio
    transcribe = spec["kind"] in ("transcribe", "denoise_transcribe") and not parts_mode
    # A re-run starts from the copies the job it came from kept (rerun.py): "clean" says what to do about the sound
    # ("full": from the upload, as a first run; "denoised" / "rebuilt": from the kept noise removal / rebuilt voice;
    # "reuse": the cleaned sound of the source job is as wanted), "transcribe" whether the transcript is made again.
    rerun = spec.get("rerun")
    clean_mode = rerun["clean"] if rerun else "full"
    transcribe_run = transcribe and (rerun is None or bool(rerun["transcribe"]))
    denoise_run = denoise and clean_mode != "reuse"
    t0 = time.time()

    editor_report: dict = {}
    cache = None
    needs_cloud = False
    cleaned_path = None
    notice = None
    noise_used = None  # which noise remover ran ("light" | "heavy"), for the record
    kept_out: dict = {}  # the copies kept for a later "change and save again" (reported to the worker, which checks them)
    asr_end = 45 if denoise_run else 90  # progress when transcription is done
    x = None  # the decoded audio, echo-free when the second track was used
    echo_skipped = False  # echo removal was asked for but the file has no second track (so it is not charged)
    if opts.get("echo_reference_track") and (denoise_run or transcribe_run):
        if probe.audio_tracks < 2:
            echo_skipped = True
            notice = "This file has only one audio track, so there was no desktop sound to remove echo with. It was processed without echo removal."
        else:
            # Before anything else: both the captions and the cleaned audio should start from the echo-free voice.
            emit(stage="Removing echo", progress=2)
            mic = pipeline.decode_for_clean(ff, src, probe, limit, work_dir=job_dir)
            if ed_spec and denoise_run and clean_mode == "full" and not ed_spec.get("timeline"):
                from acw.worker import editrun

                cache = editrun.cache_for(job_dir, ff)
                if editrun.needs_facts(ed_spec, job_dir):
                    editor_report["facts"] = editrun.facts_of(editor.Inputs(mic, ed_spec["source_id"]), ff, probe, job_dir)  # (the recording as it came)
            x = pipeline.remove_echo(
                ff, src, probe, mic, limit, work_dir=job_dir,
                progress=lambda f: emit(stage="Removing echo", progress=int(2 + 3 * f)),
                **({"cache": cache, "input_id": f"{ed_spec['source_id']}~r"} if cache is not None else {}),
            )
            del mic
    if parts_mode:
        from acw.worker import editrun

        emit(stage="Decoding", progress=2)
        if x is None and denoise_run:
            x = pipeline.decode_for_clean(ff, src, probe, limit, work_dir=job_dir)
        needs_cloud = editrun.subtitles_prepare(job_dir, spec, ff, probe, x, limit, emit, base=5, span=asr_end - 5)
    if transcribe_run:
        # Transcribe first, from the original audio: every recogniser we tested
        # scores worse on denoised speech (docs/research/asr.md).
        emit(stage="Decoding", progress=5 if x is not None else 2)
        if x is None and denoise_run:
            # Cleaned too: decode once, at the cleaning rate, and make the recogniser's 16 kHz mono from that.
            x = pipeline.decode_for_clean(ff, src, probe, limit, work_dir=job_dir)
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
                lang = transcribe.detect_language(speech_16k, models / "whisper" / "base", models / "whisper" / "small", threads)
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
    if denoise_run:
        start = asr_end if transcribe_run else (5 if x is not None else 2)
        emit(stage="Decoding", progress=start)
        # Decoded and cleaned audio live in memory-mapped files in the job
        # directory, so RAM use doesn't grow with the length of the upload.
        if x is None:
            x = pipeline.decode_for_clean(ff, src, probe, limit, work_dir=job_dir)
        span = 85 - start
        rebuild = spec.get("denoise_tier") == "rebuild"
        leveling_only = spec.get("denoise_tier") == "off"  # Leveling on its own: no noise removal
        copts = lambda profile: clean_options(spec, profile)  # noqa: E731
        progress = lambda f: emit(stage="Cleaning" if f < 0.95 else "Leveling", progress=int(start + 2 + span * f))  # noqa: E731
        used: dict = {}
        echo_flag = bool(opts.get("echo_reference_track"))
        if ed_spec and ed_spec.get("timeline"):
            # A job made with a timeline: only its parts are made and charged, from an empty block cache (editrun.render; the echo
            # removal, the noise removers and Voice rebuild run on their blocks, Leveling per part)
            from acw.worker import editrun

            y, editor_report, tl_notice = editrun.render(job_dir, spec, ff, probe, x, emit, base=start + 2, span=span, op="first")
            editor_report["facts"] = editrun.facts_of(editor.Inputs(x, ed_spec["source_id"]), ff, probe, job_dir)
            notice = notice or tl_notice
        elif clean_mode in ("denoised", "rebuilt"):
            # From the copy the source job kept: no noise model runs (the blend for the strength, Leveling). The copy was made from the same recording (pipeline.redo_denoised / redo_rebuilt).
            entry = rerun["kept"]["noise" if clean_mode == "denoised" else "rebuilt"]
            emit(stage="Cleaning", progress=start + 2)
            y = pipeline.output_buffer(x, job_dir)
            y[:] = 0.0
            kept_copies.load(ff, entry, job_dir, y, job_dir / "kept-decoded.f32")
            if clean_mode == "denoised":
                plan = entry["plan"]
                if len(plan) != x.shape[1]:
                    raise ValueError("the kept copy does not fit this recording")
                y = pipeline.redo_denoised(x, y, plan, copts(entry["profile"]), progress=progress, report=used)
                used["noise"] = entry["profile"]
            else:
                y = pipeline.redo_rebuilt(x, y, copts("auto"), progress=progress, report=used)
        else:
            # The noise remover is chosen here (measured on the echo-free recording), so the worker can base its
            # time estimate on it; clean_audio then uses the same choice.
            kept: dict = {}  # a short recording's measurement doubles as its full-strength cleaning (pipeline.Measured)
            block_cache: dict = {}  # the editor's block cache is filled by this run (editrun.py): the next edit only runs what is missing
            if ed_spec and clean_mode == "full":
                from acw.worker import editrun

                echo_done = cache is not None  # (the blocks of the echo removal went there already, and the facts, from the recording as it came)
                cache = cache or editrun.cache_for(job_dir, ff)
                if not echo_done and editrun.needs_facts(ed_spec, job_dir):
                    editor_report["facts"] = editrun.facts_of(editor.Inputs(x, ed_spec["source_id"]), ff, probe, job_dir)
                if echo_done:
                    block_cache = {"cache": editrun.EchoWholeFile(cache, ed_spec["source_id"], len(x)), "input_id": editrun.EchoWholeFile.MARK}
                else:
                    block_cache = {"cache": cache, "input_id": ed_spec["source_id"]}
            if rebuild or leveling_only:
                profile = opts.get("noise", "auto")
            else:
                profile = pipeline.choose_noise_profile(x, spec["lib_path"], spec["dfn_model"], opts.get("noise", "auto"), keep=kept)
            emit(stage="Leveling" if leveling_only else "Cleaning", progress=start + 2,
                 **({} if rebuild or leveling_only else {"noise": profile}))

            def keep_copy(name: str, y: Any, cols: list[int] | None) -> None:
                """What a later "change and save again" starts from: the noise removal at full strength, or the rebuilt voice."""
                if name == "denoised":
                    entry = kept_copies.save(ff, y, cols, job_dir / KEPT_NOISE)
                    kept_out["noise"] = {**entry, "echo": echo_flag, "profile": profile, "noise_opt": opts.get("noise", "auto"),
                                         "plan": pipeline.channel_plan(x), "energy": kept_copies.energy(y, x, cols)}
                    # the recording as the noise remover heard it (echo removed), on the same sample grid, for any length: with the
                    # copy above it is what the page blends for its strength slider, exactly (web/src/media/blend.js), in short
                    # stretches (acw/stretch.py); "energy" above is what keeps the loudness of a preview even
                    kept_out["input"] = {**kept_copies.save(ff, x, cols, job_dir / KEPT_INPUT), "echo": echo_flag}
                else:
                    kept_out["rebuilt"] = {**kept_copies.save(ff, y, cols, job_dir / KEPT_REBUILT), "echo": echo_flag}

            y = pipeline.clean_audio(
                x, spec["lib_path"], spec["dfn_model"], copts(profile), threads=threads, progress=progress,
                out=pipeline.output_buffer(x, job_dir),
                heavy_model_path=Path(spec["models_dir"]) / "dpdfnet" / dpdfnet.MODEL_FILE,
                restore_model_dir=Path(spec["models_dir"]) / "sidon",
                report=used,
                measured=kept.get("measured"),
                keep=keep_copy,
                **block_cache,
            )
        if ed_spec and ed_spec.get("suggest") and clean_mode == "full":
            from acw.worker import editrun

            editor_report["suggestions"] = editrun.find_suggestions(x, spec, emit, 86, 6)
        noise_used = used.get("noise")
        fmt = opts.get("audio_format", "flac")
        cleaned_path = ff.write_audio(pipeline.as_output(y), pipeline.SAMPLE_RATE, job_dir / f"{stem}-cleaned", fmt)
        outputs.append({"file": cleaned_path.name, "kind": "audio"})
        del x, y
        for scratch in ("decoded.f32", "reference.f32", "echo-free.npy", "cleaned.npy", "kept-decoded.f32"):
            (job_dir / scratch).unlink(missing_ok=True)
    elif denoise and not no_audio:
        # The cleaned sound of the job this one came from is what is wanted (only the subtitles change): reused as it is,
        # or in the other file type that was asked for.
        from acw_common.ffmpeg import AUDIO_FORMATS

        fmt = opts.get("audio_format", "flac")
        old = job_dir / rerun["reuse_audio"]
        if old.suffix == AUDIO_FORMATS[fmt][0]:
            cleaned_path = old
        else:
            emit(stage="Cleaning", progress=60)
            p = ff.probe(old)
            audio = ff.decode(old, p, pipeline.SAMPLE_RATE, min(max(p.audio_channels, 1), 2), limit)
            cleaned_path = ff.write_audio(pipeline.as_output(audio), pipeline.SAMPLE_RATE, job_dir / f"{stem}-cleaned", fmt)
        outputs.append({"file": cleaned_path.name, "kind": "audio"})

    if no_audio:  # a timeline of subtitles only: the sound was not rendered, so the result files hold the recording as it is
        from acw_common.timeline import Timeline

        editor_report.update(rendered=Timeline(probe.duration).to_json(), billed={})
    (job_dir / "main-state.json").write_text(
        json.dumps({"outputs": outputs, "cleaned": cleaned_path.name if cleaned_path else None, "has_video": probe.has_video})
    )
    # Translations need the internet, which this sandbox doesn't have: they run outside it, then `finalize` writes the files.
    # (A re-run that kept some languages only asks for the others.)
    missing = [lang for lang in (opts.get("translate") or []) if not (job_dir / f"segments.{lang}.json").is_file()] if not parts_mode else []
    needs_cloud = needs_cloud or bool(missing)
    result = {"needs_cloud_asr": needs_cloud, "duration": probe.duration, "seconds_spent": time.time() - t0, "notice": notice,
              "echo_skipped": echo_skipped, "noise": noise_used, "kept": kept_out, "editor": editor_report}
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
