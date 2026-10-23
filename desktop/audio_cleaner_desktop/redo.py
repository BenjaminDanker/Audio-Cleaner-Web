"""Change and save again: a finished job's sound remade from its timeline (what to do to which part, the editor's draft) and saved as a
new file. Only the blocks the job's block cache (copies.py) does not have yet are made; the rest is put together from what is there, so
a new strength or Leveling takes seconds and Voice rebuild on one part only rebuilds that part. A job made before the block cache gets its
saved copies turned into blocks first (editor_job.make_ready); one without either starts an empty cache and makes what it needs from
the original.

  subtitles the job never made, or in another language            speech recognition on the original (the transcript is kept too)
  another subtitle file from a transcript that is saved           a text file is written; nothing else runs

The original is read again for everything but the subtitle files. The blocks are only used when the original is still the file they were
made from (copies.check_original) and has the length they have. A new result never replaces a file by itself: it is written next to the
earlier one under a free name, or, when the user chose the earlier name and agreed, the earlier file goes to the Recycle Bin / Trash first.

No Qt in here.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from acw_common import pipeline
from acw_common.ffmpeg import MediaError

from . import copies, editor_job, engine, resources

log = logging.getLogger(__name__)

TEXT_KINDS = ("srt", "vtt", "txt")


@dataclass(frozen=True)
class Wanted:
    """The choices of a change: the sound (a timeline) and how it is saved."""

    subtitles: str = "off"
    language: str = "auto"
    video_size: str = "same"
    output_name: str = ""  # the whole file name, its ending says the type
    timeline: dict | None = None  # what to do to which part (acw_common.timeline JSON); None: the sound of the latest result

    def options(self) -> engine.JobOptions:
        """The engine's options for saving it (the sound itself is the timeline)."""
        return engine.JobOptions(subtitles=self.subtitles, language=self.language, video_size=self.video_size)


def wanted_from(job: copies.JobCopies, output_name: str = "") -> Wanted:
    """The choices the job's latest result was made with."""
    m = job.made
    return Wanted(
        subtitles=m.get("subtitles") if m.get("subtitles") in engine.SUBTITLE_MODES else "off",
        language=m.get("language") if m.get("language") in engine.LANGUAGES else "auto",
        video_size=m.get("video_size") if m.get("video_size") in engine.VIDEO_SIZES else "same",
        output_name=output_name,
    )


def last_result(job: copies.JobCopies) -> Path | None:
    """The first file of the latest result saved from this job."""
    for save in reversed(job.saves):
        files = save.get("files") or []
        if files:
            return Path(files[0])
    return None


def result_dir(job: copies.JobCopies, fallback: Path | None = None) -> Path:
    """Where a new result goes: next to the latest one, else next to the original (else `fallback`)."""
    last = last_result(job)
    if last is not None and last.parent.is_dir():
        return last.parent
    if job.original_path.parent.is_dir():
        return job.original_path.parent
    return fallback or job.original_path.parent


def suggest_name(job: copies.JobCopies, source: engine.Source) -> str:
    """A name for the next result that is not taken in the folder it goes to: "talk-cleaned (2).mp3" when "talk-cleaned.mp3" is there."""
    last = last_result(job)
    folder = result_dir(job)
    if last is not None:
        stem, ending = engine.split_name(last.name)
        if ending in engine.ENDING_NAMES:
            stem = _without_number(stem)
        else:
            stem, ending = job.original_path.stem + "-cleaned", engine.default_ending(source, engine.JobOptions())
    else:
        stem, ending = job.original_path.stem + "-cleaned", engine.default_ending(source, engine.JobOptions())
    n = 1
    while True:
        name = f"{stem}{'' if n == 1 else f' ({n})'}{ending}"
        if not (folder / name).exists():
            return name
        n += 1


def _without_number(stem: str) -> str:
    import re

    return re.sub(r" \(\d+\)$", "", stem)


# ---------------------------------------------------------------------------------------------------- what a change needs

def timeline_of(job: copies.JobCopies, wanted: Wanted):
    """The timeline a change stands for: the one the editor gave, else the latest result's. None when it is not valid."""
    from acw_common.timeline import Timeline, TimelineError

    seconds = editor_job.seconds_of(job)
    if wanted.timeline is None:
        return editor_job.current_timeline(job, seconds)
    try:
        return Timeline.from_json(wanted.timeline, seconds=seconds)
    except TimelineError:
        return None


def _blocks_there_for(job: copies.JobCopies, tl, step: str) -> bool:
    q = editor_job.quote_for(job, tl)
    return q is not None and not any(t.step == step for t in q.plan.misses)


@dataclass
class Plan:
    """What a change would do, in words fit for the user, and what stops it."""

    transcribe: bool = False
    slow: bool = False  # a model runs
    steps: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def plan(
    job: copies.JobCopies, wanted: Wanted, original: copies.OriginalCheck, *, source: engine.Source | None = None,
    voice_model: bool = True, speech_model: bool = True, noise_models: bool = True,
) -> Plan:
    p = Plan()
    if not original.ok:
        p.problems.append(f"{original.text} Everything but the subtitle files needs the original.")
    tl = timeline_of(job, wanted)
    q = editor_job.quote_for(job, tl) if tl is not None else None
    if tl is None:
        p.problems.append("The timeline is not valid.")
    elif editor_job.needs_migration(job):
        p.slow = True
        p.steps.append("Get this job's saved copies ready, make what changed, then put the sound together")
    elif q is None:
        p.slow = True
        p.steps.append("Make the parts that are not made yet, then put the sound together")
    elif q.nothing:
        p.steps.append("Put the sound together from what is already processed" + (", level it" if tl.layers["leveling"] else ""))
    else:
        p.slow = True
        p.steps.append(f"Process what changed ({editor_job.say_time(q.seconds)}), then put the sound together")
    if tl is not None and tl.layers["rebuild"] and not voice_model and not _blocks_there_for(job, tl, "rebuild"):
        p.problems.append("The model that rebuilds the voice is not installed. Download it in Settings.")
    if tl is not None and not noise_models and tl.layers["noise"]:
        p.problems.append("The noise removal models are not installed.")
    if wanted.subtitles != "off":
        have = job.transcript()
        usable = have is not None and (wanted.language == "auto" or wanted.language == have[1])
        if not usable:
            p.transcribe = p.slow = True
            p.steps.append("Listen to the original again to write the subtitles")
            if not speech_model:
                p.problems.append("The speech model for subtitles is not installed. Download it in Settings.")
    name = wanted.output_name.strip()
    if name and source is not None:
        _, problem = engine.check_output_name(name, source)
        if problem:
            p.problems.append(problem)
    elif not name:
        p.problems.append("Type a name for the new file.")
    p.steps.append("Save it as a new file" + (f" ({name})" if name else ""))
    return p


# ---------------------------------------------------------------------------------------------------- doing it

def save_subtitles(job: copies.JobCopies, kind: str, dest_dir: Path, stem: str) -> Path:
    """Write the saved transcript as a subtitle file (`kind`: srt, vtt or txt) named "<stem>.<kind>", or "<stem> (2).<kind>"
    when that is taken. Needs no model and not the original."""
    if kind not in TEXT_KINDS:
        raise ValueError(f"unknown subtitle file kind {kind!r}")
    saved = job.transcript()
    if saved is None:
        raise MediaError("This job has no saved transcript.")
    segments, language = saved
    ext = engine._SUBTITLE_SUFFIX[kind]
    target = engine.claim_output(dest_dir, stem, ext, suffix="")
    try:
        target.write_text(engine._subtitle_text(kind, segments, language), encoding="utf-8")
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target


def take_place(new: Path, old: Path, trash: Callable[[Path], bool]) -> Path:
    """The user chose the name of the earlier result and agreed to replace it: the earlier file goes to the Recycle Bin / Trash (so it
    can be brought back), then the new one is given its name. Where the earlier file can't be put in the bin it stays, and the new one
    keeps the name it has. Returns the path the new result ends up at."""
    if new == old or not old.exists():
        if new != old and not old.exists():
            try:
                os.replace(new, old)
                return old
            except OSError:
                log.warning("could not rename %s to %s", new, old, exc_info=True)
        return new
    if not trash(old):
        return new
    try:
        os.replace(new, old)
        return old
    except OSError:
        log.warning("could not rename %s to %s", new, old, exc_info=True)
        return new


def _check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise engine.Cancelled()


def run(
    job: copies.JobCopies,
    wanted: Wanted,
    *,
    out_dir: Path | None = None,
    progress: engine.Progress | None = None,
    cancel: threading.Event | None = None,
    threads: int | None = None,
    replace: Path | None = None,
    trash: Callable[[Path], bool] | None = None,
) -> engine.JobResult:
    """Make the result for `wanted` from the job's saved copies and write it as a new file next to the earlier one (in `out_dir` when
    given). `replace`: an earlier result whose name was chosen on purpose; it goes to the bin (`trash`) and the new file gets its name.
    Raises MediaError (fit for the user) when the original is gone or changed, or a step can't run; Cancelled when stopped. No partial
    file is left, and nothing is written over an existing file."""
    cancel = cancel or threading.Event()
    report = engine._Report(progress)
    threads = threads or engine.worker_threads()
    ff = engine.ffmpeg(threads, cancel)
    src = job.original_path

    check = job.check_original()
    if not check.ok:
        raise MediaError(check.text + " Changing and saving it again needs the original.")
    info = engine.probe(src, ff)
    source = engine.Source.of(info, src)
    options = wanted.options()
    sound_plan = plan(job, wanted, check, source=source, voice_model=resources.voice_model_dir() is not None,
                      speech_model=resources.speech_ready(), noise_models=resources.noise_ready())
    if not sound_plan.ok:
        raise MediaError(sound_plan.problems[0])
    name = engine.output_name_for(src, source, options, wanted.output_name)
    out_stem, ext = engine.split_name(name)
    dest_dir = Path(out_dir) if out_dir is not None else result_dir(job)
    tmp_out = dest_dir / f".acw-{uuid.uuid4().hex}{ext}"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp_out.touch(exist_ok=False)
    except OSError as e:
        raise MediaError(f"Cannot write to the folder {dest_dir}: {e.strerror or e}") from e

    embed = wanted.subtitles == "embed" and ext in engine._EMBED_ENDINGS
    sub_kind = "srt" if wanted.subtitles == "embed" else wanted.subtitles
    keeper = copies.Keeper.for_job(job, ff)
    x = y = None
    sub_tmp = None
    notice = None
    try:
        work = Path(tempfile.mkdtemp(
            dir=engine._work_parent(dest_dir, engine.estimated_scratch_gb(info.duration, info.audio_channels) * 1.5),
            prefix=f"{engine._WORK_PREFIX}{os.getpid()}-",
        ))
        try:
            _check_cancel(cancel)
            report(0.02)
            inputs = editor_job.decode(ff, src, info, work, reference=True, cancel=cancel, source_id=job.source_id)
            x = inputs.source
            if job.samples and len(x) != job.samples:
                raise MediaError("The original no longer matches the saved copies (it is not the same length), so it can't be changed and saved again.")
            editor_job.make_ready(job, ff, inputs, work, cancel=cancel)
            report(0.1)
            segments, language = None, None
            if wanted.subtitles != "off":
                saved = job.transcript()
                if sound_plan.transcribe:
                    segments, language = engine._transcribe(ff, src, info, work, options, threads, report.span(0.1, 0.4), cancel)
                    keeper.transcript(segments, language, job.echo)
                else:
                    segments, language = saved
                _check_cancel(cancel)
                if any(s.text.strip() for s in segments):
                    sub_tmp = work / f"subtitles{engine._SUBTITLE_SUFFIX[sub_kind]}"
                    sub_tmp.write_text(engine._subtitle_text(sub_kind, segments, language), encoding="utf-8")
            start = 0.4 if sound_plan.transcribe else 0.1
            tl = timeline_of(job, wanted)
            y, done, decided, notice, _ = editor_job.render_timeline(
                tl, inputs, job.block_cache(ff), work=work, threads=threads, known=job.editor.get("resolved"),
                progress=report.span(start, 0.9), cancel=cancel, out=pipeline.output_buffer(x, work))
            _check_cancel(cancel)
            made = _made(job, wanted, done, language, ext)
            kept_info = engine._write(ff, src, pipeline.as_output(y), info, options, tmp_out, sub_tmp if embed else None, language,
                                      work=work, progress=report.span(0.9, 0.99))
            if info.audio_tracks > 1:
                notice = engine._other_tracks_notice(info.audio_tracks - 1, kept_info.get("audio_kept", 0))
            if kept_info.get("fallback"):
                notice = "; ".join(filter(None, [notice, "this computer's own video encoder wasn't available, so the picture was "
                                                         "saved in a simpler format and the file is bigger than usual"]))
            x = y = None
            _check_cancel(cancel)
            sidecar = sub_tmp is not None and not embed
            final, sub_final = engine.place_result(dest_dir, out_stem, ext, tmp_out, sub_tmp if sidecar else None, sub_kind)
            if replace is not None and trash is not None and final.parent == replace.parent and replace.suffix.lower() == final.suffix.lower():
                final = take_place(final, replace, trash)
                if sub_final is not None:
                    sub_old = replace.with_suffix(sub_final.suffix)
                    sub_final = take_place(sub_final, sub_old, trash) if sub_old.exists() else sub_final
            job.made.update(made)
            editor_job.remember(job, done, decided)
            job.add_save([final, *([sub_final] if sub_final else [])], made)
        except BaseException as e:
            x = y = None
            traceback.clear_frames(e.__traceback__)
            raise
        finally:
            x = y = None
            try:
                shutil.rmtree(work)
            except FileNotFoundError:
                pass
            except OSError:
                threading.Thread(target=engine._remove_work_dir, args=(work,), name="scratch-cleanup", daemon=True).start()
    finally:
        tmp_out.unlink(missing_ok=True)
    report(1.0)
    return engine.JobResult(
        final, info.duration, info.has_video, sub_final, embedded=embed and sub_tmp is not None, language=language,
        notice=notice, kept_dir=job.dir, made=made,
    )


def _made(job: copies.JobCopies, wanted: Wanted, tl, language: str | None, ext: str) -> dict:
    """The choices the new result was made with (copies.JobCopies.made)."""
    return {**editor_job.made_from_timeline(tl, echo=bool(job.echo)), "noise_option": job.made.get("noise_option", "auto"),
            "audio_format": engine.AUDIO_ENDINGS.get(ext, "same"), "subtitles": wanted.subtitles, "language": language or wanted.language,
            "video_size": wanted.video_size}
