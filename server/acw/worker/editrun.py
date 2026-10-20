"""The per-part editor inside the sandboxed runner (see acw/editing.py for the whole picture).

This module runs in the confined process that handles the untrusted upload: it has no database and no network. The worker hands it a
`spec["editor"]` and reads back what it reports (`result["editor"]`), which the worker checks before it believes any of it.

    stage_edit        an edit run: decode the recording, decide the removers that are still "auto", plan the timeline against the job's
                      block cache, make the missing blocks and render the timeline into the result audio (or only find suggestions)
    render_first      the first run of a job made with a timeline: the same, from an empty cache
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
    y = editor.render(tl, inputs, cache, runner_of(spec), work_dir=job_dir, progress=prog)
    report = {"rendered": tl.to_json(), "resolved": resolved, "billed": billed, "computed": computed}
    return y, report, notice


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
        y, rep, notice = render(job_dir, spec, ff, probe, x, emit, base=5, span=80, op=ed["op"])
        report.update(rep)
        fmt = ed.get("audio_format") or spec["options"].get("audio_format", "flac")
        emit(stage="Writing the audio", progress=90)
        out = ff.write_audio(pipeline.as_output(y), pipeline.SAMPLE_RATE, job_dir / f"{spec['output_stem']}-cleaned", fmt)
        result["outputs"] = [{"file": out.name, "kind": "audio"}]
        result["notice"] = notice
        del y
    emit(stage="Done", progress=100)
    result.update(editor=report, seconds_spent=time.time() - t0, duration=probe.duration)
    for scratch in ("decoded.f32", "reference.f32", "echo-free.npy", "cleaned.npy"):
        (job_dir / scratch).unlink(missing_ok=True)
    return result
