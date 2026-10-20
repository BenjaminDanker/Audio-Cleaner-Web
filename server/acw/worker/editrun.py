"""The per-part editor inside the sandboxed runner (see acw/editing.py for the whole picture).

This module runs in the confined process that handles the untrusted upload: it has no database and no network. The worker hands it a
`spec["editor"]` and reads back what it reports (`result["editor"]`), which the worker checks before it believes any of it.

    stage_edit        an edit run: decode the recording, decide the removers that are still "auto", plan the timeline against the job's
                      block cache, make the missing blocks and render the timeline into the result audio (or only find suggestions);
                      the subtitle parts that are still to make are transcribed (Standard here, Accurate in the web process) and the
                      caption files put together
    render_first      the first run of a job made with a timeline: the same, from an empty cache
    load_browser      the audio the browser cleaned for the noise parts that say `browser: true`
    subtitles_prepare / merge_subtitles
                      the timeline's subtitle parts: transcribe what the job's folder lacks (Standard with the local recogniser; Accurate
                      parts are cut into speech chunks for the web process), then put every part's segments, the translations and the
                      free parts' uploaded cues together as segments.json / segments.<lang>.json (acw_common.subtitles)
    legacy_report     the first run of a job made without one (`pipeline.clean_audio`): what the editor needs to know about it
    EchoWholeFile     lets clean_audio fill the cache after a whole-file echo removal under the editor's block identities
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SR = 48_000
SOURCE = "kept-source"
BLOCKS = "blocks"
SUBS = "subs"  # acw/editing.py: the subtitle parts' folders
BROWSER = "browser"  # ... and what the browser uploaded


def part_key(start: float, end: float) -> str:
    return f"{start:.3f}-{end:.3f}"


def cache_for(job_dir: Path, ff):
    from acw_common import blocks

    return blocks.FlacCache(job_dir / BLOCKS, ff)


class EchoWholeFile:
    """What `pipeline.clean_audio` is given as its cache after the whole file went through echo removal: it keys every block with one
    `input_id`, but the editor's identity of a block names the echo parts that touch what it read (here: all of the file, cut to the
    block's reach). `put`, `get` and `has` rewrite the key to the editor's own (editor._Identity), so a whole-file run fills the cache
    the editor plans against."""

    MARK = "\x1eecho"

    def __init__(self, cache, source_id: str, n: int):
        from acw_common import editor

        self.cache = cache
        self.n = n
        self.ident = editor._Identity(SimpleNamespace(source_id=source_id, reference_id="r"), [(0, n)])

    def _key(self, key: str) -> str:
        from acw_common import restore

        parts = key.split("|")
        if self.MARK not in parts:
            return key
        i = parts.index(self.MARK)
        if parts[1] in ("light", "heavy"):
            lo, hi = int(parts[i + 2]), int(parts[i + 5])
        else:  # Voice rebuild: the block's number and the length of the recording
            lo, hi = restore.block_input_range(int(parts[i + 3]), self.n)
        parts[i] = self.ident(lo, hi)
        return "|".join(parts)

    def get(self, key: str):
        return self.cache.get(self._key(key))

    def put(self, key: str, block) -> None:
        self.cache.put(self._key(key), block)

    def has(self, key: str) -> bool:
        return self.cache.has(self._key(key))


def facts_of(inputs, ff, probe, job_dir: Path, *, keep_source: bool = True) -> dict:
    """What the planner needs about the recording (and the recording itself as 24-bit FLAC, for the preview stretches)."""
    from acw_common import kept as kept_copies

    facts = {
        "n": inputs.n,
        "channels": inputs.channels,
        "plan": [None if c is None else int(c) for c in inputs.plan],
        "spans": {str(c): [[int(a), int(b)] for a, b in inputs.spans(c)] for c in inputs.todo},
        "rebuild_mid": bool(inputs.rebuild_mid),
        "reference": probe.audio_tracks >= 2,
    }
    if keep_source:
        facts["source"] = kept_copies.save(ff, inputs.source, None, job_dir / SOURCE)
    return facts


def needs_facts(ed: dict, job_dir: Path) -> bool:
    """The facts and the recording's FLAC are made when the job has none (a re-run's come linked from its source)."""
    return not ed.get("facts") or not (job_dir / f"{SOURCE}.flac").is_file()


def find_suggestions(x, spec: dict, emit, base: int, span: int):
    """The suggestions for the recording (acw_common.suggest), as JSON; None when they cannot be made (never fails the job)."""
    from acw_common import suggest

    ed = spec["editor"]
    if not ed.get("suggest"):
        return None
    try:
        emit(stage="Finding parts that need other settings", progress=base)
        s = suggest.suggest(x, spec["lib_path"], spec["dfn_model"], threads=int(spec.get("threads", 2)),
                            progress=lambda f: emit(stage="Finding parts that need other settings", progress=int(base + span * f)))
        return s.to_json()
    except Exception:  # noqa: BLE001  (suggestions are a courtesy)
        import traceback

        traceback.print_exc()
        return None


def runner_of(spec: dict):
    from acw_common import dpdfnet, editor

    models = Path(spec["models_dir"])
    return editor.Runner(
        lib_path=spec["lib_path"], model_path=spec["dfn_model"], heavy_model_path=models / "dpdfnet" / dpdfnet.MODEL_FILE,
        restore_model_dir=models / "sidon", device=spec.get("clean_device", "cpu"), threads=int(spec.get("threads", 2)),
    )


def resolved_timeline(tl, inputs, spec: dict):
    """The timeline with every "auto" remover decided: the ones decided before (spec["editor"]["resolved"], so a quote and its run
    agree) and the others measured on their part. Returns (timeline, {part: remover} of the parts that were "auto")."""
    from acw_common import editor

    known = spec["editor"].get("resolved") or {}
    out = tl.copy()
    autos = []
    for p in out.layers["noise"]:
        if p.settings.get("remover") == "auto":
            r = known.get(part_key(p.start, p.end))
            if r in ("light", "heavy"):
                p.settings["remover"] = r
            autos.append(part_key(p.start, p.end))
    out = editor.resolve_removers(out, inputs, spec["lib_path"], spec["dfn_model"])
    return out, {k: next(p.settings["remover"] for p in out.layers["noise"] if part_key(p.start, p.end) == k) for k in autos}


def render(job_dir: Path, spec: dict, ff, probe, x, emit, *, base: int = 5, span: int = 80, op: str = "process") -> tuple[np.ndarray, dict, str | None]:
    """Plan the timeline of the run against the job's cache and render it: (the audio, the report for the worker, a notice)."""
    from acw_common import editor
    from acw_common.timeline import Timeline

    ed = spec["editor"]
    limit = spec["max_seconds"] if spec.get("billed_seconds") is None else min(spec["max_seconds"], spec["billed_seconds"] * 1.02 + 2)
    n = len(x)
    tl = Timeline.from_json(ed["timeline"], seconds=n / SR)
    notice = None
    ref = None
    if tl.layers["echo"]:
        if probe.audio_tracks >= 2:
            emit(stage="Reading the speaker track", progress=base)
            ref = ff.decode_to_file(job_dir / spec["input"], probe, SR, 1, limit, job_dir / "reference.f32", track=1)[:, 0]
        else:
            tl.layers["echo"] = []
            notice = "This file has only one audio track, so there was no desktop sound to remove echo with. It was processed without echo removal."
    inputs = editor.Inputs(x, ed["source_id"], reference=ref, reference_id="r" if ref is not None else None)
    emit(stage="Deciding what each part needs", progress=base + 1)
    tl, resolved = resolved_timeline(tl, inputs, spec)
    cache = cache_for(job_dir, ff)
    prev = None
    if ed.get("previous"):
        try:
            prev = Timeline.from_json(ed["previous"], seconds=n / SR)
        except Exception:  # noqa: BLE001
            prev = None
    plan = editor.plan(tl, inputs, cache, prev)
    billed = {s: {"seconds": r["seconds"]} for s, r in editor.billable(plan).items()}
    summary = plan.summary()
    computed = {s: r["blocks"] for s, r in summary.items() if r["blocks"]}
    prog = lambda f: emit(stage="Cleaning" if f < 0.9 else "Leveling", progress=int(base + 2 + (span - 2) * f))  # noqa: E731
    browser = load_browser(job_dir, spec, ff, x)
    y = editor.render(tl, inputs, cache, runner_of(spec), work_dir=job_dir, progress=prog, browser=browser)
    report = {"rendered": tl.audio().to_json(), "resolved": resolved, "billed": billed, "computed": computed}
    return y, report, notice


def load_browser(job_dir: Path, spec: dict, ff, x: np.ndarray):
    """The `editor.BrowserAudio` of the noise parts the browser cleaned: each uploaded FLAC decoded (to a file, memory-mapped) to the
    recording's channels, cut or padded with silence to exactly its part. None when the timeline has no such part."""
    from acw_common import editor

    parts = spec["editor"].get("browser") or []
    if not parts:
        return None
    n, ch = len(x), x.shape[1]
    pieces = []
    for i, b in enumerate(parts):
        a, z = max(0, min(n, int(round(b["start"] * SR)))), max(0, min(n, int(round(b["end"] * SR))))
        if z <= a:
            continue
        want = z - a
        path = job_dir / b["file"]
        if want < SR // 10:  # (too short to decode as a file of its own: left as it was)
            pieces.append((a, np.array(x[a:z], dtype=np.float32)))
            continue
        raw = job_dir / f"browser-{i}.f32"
        y = ff.decode_to_file(path, ff.probe(path), SR, ch, want / SR + 1.0, raw)
        if len(y) != want:  # the file said how many samples it has; what it holds is checked here
            fixed = np.lib.format.open_memmap(job_dir / f"browser-{i}.npy", mode="w+", dtype=np.float32, shape=(want, ch))
            fixed[:] = 0.0
            m = min(want, len(y))
            fixed[:m] = y[:m]
            y = fixed
        pieces.append((a, y))
    return editor.BrowserAudio(pieces)


# ---------------------------------------------------------------------------------------------------------------------
# Subtitles per part
# ---------------------------------------------------------------------------------------------------------------------
def _ms(t: float) -> int:
    return int(round(float(t) * 1000))


def _write_json(path: Path, obj) -> None:
    import json

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def subtitles_prepare(job_dir: Path, spec: dict, ff, probe, x: np.ndarray | None, limit: float, emit, base: int = 5, span: int = 40) -> bool:
    """Transcribe the subtitle parts the job's folder does not hold yet (`subs/<key>/segments.json`), part by part, each with a little
    sound around it for context (acw_common.subtitles.padded) that is cut away again: Standard parts here with the local recogniser
    (language detected on the part when it says none); Accurate parts are cut into speech chunks (`subs/<key>/asr_chunks`, only the
    part's own speech) for the web process, which alone reaches the internet. `x`: the decoded recording (48 kHz), or None (then the
    16 kHz mono sound is decoded here). Returns True when something is left for the web process: chunks to transcribe, translations to
    make. Free parts have nothing to do here (the browser transcribed them)."""
    from acw_common import pipeline, transcribe
    from acw_common import subtitles as st

    parts = [p for p in spec["editor"]["subtitles"]["parts"] if p["quality"] != "free"]
    root = job_dir / SUBS
    models = Path(spec["models_dir"])
    threads = int(spec.get("threads", 2))
    total = len(x) / SR if x is not None else float(probe.duration)
    full16: list = []

    def speech(a_s: float, b_s: float) -> np.ndarray:
        """The 16 kHz mono sound of the recording from second a_s to b_s."""
        if x is not None:
            return pipeline.to_speech_16k(x[int(round(a_s * SR)):min(len(x), int(round(b_s * SR)))])
        if not full16:
            emit(stage="Decoding", progress=base)
            full16.append(ff.decode(job_dir / spec["input"], probe, 16000, 1, limit)[:, 0])
        return full16[0][int(round(a_s * 16000)):int(round(b_s * 16000))]

    todo = [p for p in parts if not (root / p["key"] / "segments.json").is_file() and not (root / p["key"] / "asr_chunks" / "index.json").is_file()]
    # (the parts that need the same model are made one after the other, and the model is loaded once for all of them: it is the 2.4 GB one
    # for most languages; the parts whose language is found out come last, each with whichever model its language needs)
    todo.sort(key=lambda p: transcribe.pick_engine(p["language"]) if p["language"] and p["quality"] == "standard" else "~")
    seconds_all = sum(p["end"] - p["start"] for p in todo) or 1.0
    done_s = 0.0
    asr_held: dict = {}  # the Parakeet model and the speech detector
    detect_held: dict = {}  # the small models that find out the language
    try:
        for p in todo:
            _make_part(job_dir, spec, ff, root, p, speech, st, transcribe, models, threads, emit, base, span, done_s, seconds_all, total, asr_held, detect_held)
            done_s += p["end"] - p["start"]
    finally:
        asr_held.clear()
        detect_held.clear()
        transcribe.unload()  # (the Whisper model, if one was loaded)
    return cloud_left(job_dir, spec)


def _make_part(job_dir, spec, ff, root, p, speech, st, transcribe, models, threads, emit, base, span, done_s, seconds_all, total, asr_held, detect_held) -> None:
    """One subtitle part: its text (Standard) or its speech chunks (Accurate), into its folder."""
    from acw_common.captions import segments_to_json

    d = root / p["key"]
    d.mkdir(parents=True, exist_ok=True)
    pa, pb = st.padded(p["start"], p["end"], total)
    audio = speech(pa, pb)
    first = base + span * done_s / seconds_all
    share = span * (p["end"] - p["start"]) / seconds_all
    emit(stage="Transcribing", progress=int(first))
    if p["quality"] == "accurate":
        from acw.worker import runner

        runner._write_asr_chunks(ff, d, audio, offset=pa, window=(p["start"] - pa, p["end"] - pa))
    else:
        lang = p["language"]
        if lang is None:
            emit(stage="Detecting language", progress=int(first))
            lang = transcribe.detect_language(audio, models / "whisper" / "base", models / "whisper" / "small", threads, held=detect_held)
        report = lambda f, first=first, share=share: emit(stage="Transcribing", progress=int(first + share * f))  # noqa: E731
        engine = transcribe.pick_engine(lang)
        if engine.startswith("parakeet"):
            segs = transcribe.transcribe_parakeet(audio, models / engine, models / "silero-vad", device=spec.get("asr_device", "cpu"),
                                                  threads=threads, progress=report, held=asr_held)
        else:
            whisper = models / "whisper" / spec.get("whisper_model", "large-v3-turbo")
            segs, lang = transcribe.transcribe(audio, whisper, language=lang, threads=threads, progress=report,
                                               model=transcribe.load_model(whisper, threads))  # (kept by transcribe, loaded once)
        segs = st.clip_segments(st.shift(segs, pa), p["start"], p["end"])
        _write_json(d / "segments.json", {"language": lang, "engine": "standard", "segments": segments_to_json(segs)})


def cloud_left(job_dir: Path, spec: dict) -> bool:
    """Something of the subtitle parts is left that only the web process can make: chunks waiting for Accurate transcription, or a
    translation (of a part that has its text) that is not made yet and is not into the language the part is in."""
    root = job_dir / SUBS
    for p in spec["editor"]["subtitles"]["parts"]:
        if p["quality"] == "free":
            continue
        d = root / p["key"]
        if not (d / "segments.json").is_file():
            if p["quality"] == "accurate" and (d / "asr_chunks" / "index.json").is_file():
                return True
            continue
        spoken = _read_json(d / "segments.json").get("language")
        if any(t != spoken and not (d / f"segments.{t}.json").is_file() for t in p["translate"]):
            return True
    return False


def merge_subtitles(job_dir: Path, spec: dict) -> None:
    """Put the subtitle parts together as the job's `segments.json` (the original text of every part, in time order) and
    `segments.<lang>.json` (each translated language: the translations of the parts that asked for it), which the usual caption writer
    (runner.stage_finalize) turns into the files. A free part's segments are the cues its browser uploaded. Whatever segment files the
    folder held from an earlier run are replaced; with no text at all there are none."""
    from acw_common import captions
    from acw_common import subtitles as st

    root = job_dir / SUBS
    merged = []
    for p in spec["editor"]["subtitles"]["parts"]:
        start, end = p["start"], p["end"]
        if p["quality"] == "free":
            f = job_dir / BROWSER / f"cues-{_ms(start)}-{_ms(end)}.json"
            data = _read_json(f) if f.is_file() else {}
            merged.append({"start": start, "end": end, "language": p["language"], "engine": "browser", "translations": {},
                           "segments": st.segments_from_cues(data.get("cues") or [], start, end)})
            continue
        d = root / p["key"]
        if not (d / "segments.json").is_file():
            continue
        data = _read_json(d / "segments.json")
        segs = st.clip_segments(captions.segments_from_json(data.get("segments") or []), start, end)
        tr = {}
        for lang in p["translate"]:
            tf = d / f"segments.{lang}.json"
            if tf.is_file():
                tr[lang] = st.clip_segments(captions.segments_from_json(_read_json(tf).get("segments") or []), start, end)
        merged.append({"start": start, "end": end, "language": data.get("language") or p["language"], "engine": data.get("engine"),
                       "translations": tr, "segments": segs})
    for old in job_dir.glob("segments*.json"):
        old.unlink()
    main, translated = st.combine(merged)
    if main is not None:
        _write_json(job_dir / "segments.json", main)
        for lang, payload in translated.items():
            _write_json(job_dir / f"segments.{lang}.json", payload)


def stage_edit(job_dir: Path, spec: dict, ff, emit) -> dict:
    """An edit run: process / export render the timeline into the result audio; suggest only looks for suggestions."""
    from acw_common import pipeline

    ed = spec["editor"]
    src = job_dir / spec["input"]
    probe = ff.probe(src)
    limit = spec["max_seconds"] if spec.get("billed_seconds") is None else min(spec["max_seconds"], spec["billed_seconds"] * 1.02 + 2)
    t0 = time.time()
    emit(stage="Decoding", progress=2)
    x = pipeline.decode_for_clean(ff, src, probe, limit, work_dir=job_dir)
    from acw_common import editor

    inputs = editor.Inputs(x, ed["source_id"])
    report: dict = {}
    # (facts and the recording's copy: made when the job never had them, else they stay as they are)
    if needs_facts(ed, job_dir):
        report["facts"] = facts_of(inputs, ff, probe, job_dir)
    result: dict = {"outputs": [], "notice": None}
    if ed["op"] == "suggest":
        report["suggestions"] = find_suggestions(x, spec, emit, 10, 85)
    else:
        has_subs = ed.get("subtitles") is not None
        audio = ed.get("audio", True)  # (a run that only changes the subtitles leaves the sound as it is)
        used = 0
        if has_subs:
            used = 40 if audio else 80
            result["needs_cloud_asr"] = subtitles_prepare(job_dir, spec, ff, probe, x, limit, emit, base=5, span=used)
        if audio:
            y, rep, notice = render(job_dir, spec, ff, probe, x, emit, base=5 + used, span=80 - used, op=ed["op"])
            report.update(rep)
            fmt = ed.get("audio_format") or spec["options"].get("audio_format", "flac")
            emit(stage="Writing the audio", progress=90)
            # (under a temporary name: the server puts it in the old one's place when the whole run is complete; acw/editing.py swap_in)
            from acw_common.ffmpeg import AUDIO_FORMATS

            made = ff.write_audio(pipeline.as_output(y), pipeline.SAMPLE_RATE, job_dir / "run-audio", fmt)
            final = (job_dir / f"{spec['output_stem']}-cleaned").with_suffix(AUDIO_FORMATS[fmt][0])
            out = final.with_name(final.name + ".run")
            made.replace(out)
            result["outputs"] = [{"file": out.name, "kind": "audio"}]
            result["notice"] = notice
            del y
        else:
            report["billed"] = {}
        if has_subs:
            # the caption files are put together here, or after the web process has made its part (runner.stage_finalize reads this)
            import json

            (job_dir / "main-state.json").write_text(json.dumps({"outputs": result["outputs"], "cleaned": None, "has_video": False}))
            if not result.get("needs_cloud_asr"):
                from acw.worker import runner

                result.update(runner.stage_finalize(job_dir, spec))
    emit(stage="Done", progress=100)
    result.update(editor=report, seconds_spent=time.time() - t0, duration=probe.duration)
    for scratch in ("decoded.f32", "reference.f32", "echo-free.npy", "cleaned.npy"):
        (job_dir / scratch).unlink(missing_ok=True)
    return result
