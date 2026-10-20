"""Change and save again: a finished job's result remade with other choices, starting from its saved copies (copies.py).

What a change costs depends on where the sound can start from:

  strength, Leveling, loudness, file type, name, subtitle files   from the noise removal at full strength: the strength is a blend of
                                                                   it with the original (blend.py), Leveling and the encoder run
                                                                   again. No model runs: seconds, also for a long recording.
  Voice rebuild on                                                 from the rebuilt voice when the job has one (Leveling again),
                                                                   otherwise the voice is rebuilt from the original: the slow step
  Voice rebuild off                                                from the saved noise removal, or noise removal from the original
                                                                   when the job never had one
  subtitles the job never made, or in another language            speech recognition on the original (the transcript is kept too)
  another subtitle file from a transcript that is saved           a text file is written; nothing else runs

The original is read again for everything but the subtitle files: Leveling measures the result against it, and the strength blends it
in. The copies are only used when the original is still the file they were made from (copies.check_original) and has the length they
have. A new result never replaces a file by itself: it is written next to the earlier one under a free name, or, when the user
chose the earlier name and agreed, the earlier file goes to the Recycle Bin / Trash first.

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

import numpy as np

from acw_common import kept, pipeline
from acw_common.ffmpeg import MediaError

from . import blend, copies, editor_job, engine, resources

log = logging.getLogger(__name__)

SOUND_BLEND, SOUND_LEVEL_REBUILT, SOUND_REBUILD, SOUND_NOISE = "blend", "level_rebuilt", "rebuild", "noise"
SOUND_EDITOR = "editor"  # a job that keeps blocks (editor_job.py): the sound is made from its timeline
TEXT_KINDS = ("srt", "vtt", "txt")


@dataclass(frozen=True)
class Wanted:
    """The choices of a change."""

    atten_lim_db: float = 100.0  # strength; ignored while the voice is rebuilt
    rebuild_voice: bool = False
    leveling: bool = True
    loudness: str = engine.DEFAULT_LOUDNESS
    subtitles: str = "off"
    language: str = "auto"
    video_size: str = "same"
    output_name: str = ""  # the whole file name, its ending says the type
    timeline: dict | None = None  # a job that keeps blocks: what to do to which part (acw_common.timeline JSON); None: the choices above, whole file

    def options(self) -> engine.JobOptions:
        """The same choices as the engine's options (the strength is the nearest name; what is used is `atten_lim_db`)."""
        key = blend.strength_key(self.atten_lim_db) or min(engine.STRENGTHS, key=lambda k: abs(engine.STRENGTHS[k] - self.atten_lim_db))
        return engine.JobOptions(
            strength=key, leveling=self.leveling, loudness=self.loudness, subtitles=self.subtitles, language=self.language,
            rebuild_voice=self.rebuild_voice, video_size=self.video_size,
        )


def wanted_from(job: copies.JobCopies, output_name: str = "") -> Wanted:
    """The choices the job's latest result was made with."""
    m = job.made
    atten = m.get("atten_lim_db")
    return Wanted(
        atten_lim_db=float(atten) if isinstance(atten, (int, float)) and not isinstance(atten, bool) else 100.0,
        rebuild_voice=bool(m.get("rebuilt")),
        leveling=bool(m.get("leveling", True)),
        loudness=m.get("loudness") if m.get("loudness") in engine.LOUDNESS_TARGETS else engine.DEFAULT_LOUDNESS,
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
    """The timeline a change of a job that keeps blocks stands for: the one the editor gave, else the choices on the whole file. None when
    it is not valid."""
    from acw_common.timeline import Timeline, TimelineError

    seconds = editor_job.seconds_of(job)
    try:
        if wanted.timeline is not None:
            return Timeline.from_json(wanted.timeline, seconds=seconds)
        remover = job.made.get("noise_option") if job.made.get("noise_option") in ("light", "heavy") else (
            job.made.get("noise") if job.made.get("noise") in ("light", "heavy") else "auto")
        return Timeline.whole_file(
            seconds, echo=bool(job.echo), noise=False if wanted.rebuild_voice else {"strength_db": min(wanted.atten_lim_db, 100.0), "remover": remover},
            rebuild=wanted.rebuild_voice, leveling={"loudness_lufs": engine.LOUDNESS_TARGETS[wanted.loudness]} if wanted.leveling else False)
    except TimelineError:
        return None


def _blocks_there_for(job: copies.JobCopies, tl, step: str) -> bool:
    q = editor_job.quote_for(job, tl)
    return q is not None and not any(t.step == step for t in q.plan.misses)


@dataclass
class Plan:
    """What a change would do, in words fit for the user, and what stops it."""

    sound: str = SOUND_BLEND
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
    if job.has_blocks:
        tl = timeline_of(job, wanted)
        p.sound = SOUND_EDITOR
        q = editor_job.quote_for(job, tl) if tl is not None else None
        if q is None:
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
    elif wanted.rebuild_voice:
        if job.has(copies.REBUILT):
            p.sound = SOUND_LEVEL_REBUILT
            p.steps.append("Use the rebuilt voice that was saved" + (", level it" if wanted.leveling else ""))
        else:
            p.sound, p.slow = SOUND_REBUILD, True
            p.steps.append("Rebuild the voice from the original (this takes a while, about as long as cleaning it did)")
            if not voice_model:
                p.problems.append("The model that rebuilds the voice is not installed. Download it in Settings.")
    else:
        if job.has(copies.NOISE):
            p.sound = SOUND_BLEND
            p.steps.append("Mix the original back in to the chosen strength" + (", level it" if wanted.leveling else ""))
        else:
            p.sound, p.slow = SOUND_NOISE, True
            p.steps.append("Remove the noise again (this job has no saved noise removal; it takes about as long as cleaning it did)")
            if not noise_models:
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
            inputs = None
            if sound_plan.sound == SOUND_EDITOR:
                inputs = editor_job.decode(ff, src, info, work, reference=True, cancel=cancel, source_id=job.source_id)
                x = inputs.source
                if job.samples and len(x) != job.samples:
                    raise MediaError("The original no longer matches the saved copies (it is not the same length), so it can't be changed and saved again.")
            else:
                x = _recording(job, ff, src, info, work, cancel)
            report(0.1)
            segments, language = None, None
            if wanted.subtitles != "off":
                saved = job.transcript()
                if sound_plan.transcribe:
                    segments, language = engine._transcribe(
                        ff, src, info, work, options, threads, report.span(0.1, 0.4), cancel,
                        echo_free=x if job.echo and inputs is None else None,
                    )
                    keeper.transcript(segments, language, job.echo)
                else:
                    segments, language = saved
                _check_cancel(cancel)
                if any(s.text.strip() for s in segments):
                    sub_tmp = work / f"subtitles{engine._SUBTITLE_SUFFIX[sub_kind]}"
                    sub_tmp.write_text(engine._subtitle_text(sub_kind, segments, language), encoding="utf-8")
            start = 0.4 if sound_plan.transcribe else 0.1
            used: dict = {}
            y = _sound(job, wanted, sound_plan, ff, x, work, cancel, threads, report.span(start, 0.9), used, keeper, inputs)
            _check_cancel(cancel)
            made = _made(job, wanted, sound_plan.sound, used, language, ext)
            if used.get("timeline") is not None:
                notice = "; ".join(filter(None, [notice, used.get("notice")])) or None
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
            if used.get("timeline") is not None:
                editor_job.remember(job, used["timeline"], used.get("decided"))
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


def _made(job: copies.JobCopies, wanted: Wanted, sound: str, used: dict, language: str | None, ext: str) -> dict:
    """The choices the new result was made with (copies.JobCopies.made)."""
    if sound == SOUND_EDITOR:
        return {**editor_job.made_from_timeline(used["timeline"], echo=bool(job.echo)), "noise_option": job.made.get("noise_option", "auto"),
                "audio_format": engine.AUDIO_ENDINGS.get(ext, "same"), "subtitles": wanted.subtitles, "language": language or wanted.language,
                "video_size": wanted.video_size}
    rebuilt = sound in (SOUND_REBUILD, SOUND_LEVEL_REBUILT)
    atten = None if rebuilt else float(wanted.atten_lim_db)
    return {
        "atten_lim_db": atten, "strength": blend.strength_key(atten) if atten is not None else None,
        "noise": "rebuilt" if rebuilt else (used.get("noise") or job.made.get("noise")), "rebuilt": rebuilt,
        "noise_option": job.made.get("noise_option", "auto"),
        "leveling": wanted.leveling, "loudness": wanted.loudness, "audio_format": engine.AUDIO_ENDINGS.get(ext, "same"),
        "subtitles": wanted.subtitles, "language": language or wanted.language, "video_size": wanted.video_size, "echo": job.echo,
    }


def _recording(job: copies.JobCopies, ff, src: Path, info, work: Path, cancel: threading.Event) -> np.ndarray:
    """The original as the pipeline lined it up for this job (decoded at 48 kHz, one or two channels; with the speaker echo taken out
    when that was done), checked to be as long as the copies are."""
    if job.echo:
        entry = job.entry(copies.INPUT)
        if entry is not None:
            mic = pipeline.decode_for_clean(ff, src, info, engine.MAX_SECONDS, work_dir=work)
            x = pipeline.output_buffer(mic, work)
            x[:] = 0.0
            del mic
            kept.load(ff, entry, job.dir, x, work / "input-decoded.f32")
        elif info.audio_tracks >= 2:  # the echo-free recording is made again (no model runs)
            mic = pipeline.decode_for_clean(ff, src, info, engine.MAX_SECONDS, work_dir=work)
            x = pipeline.remove_echo(ff, src, info, mic, engine.MAX_SECONDS, work_dir=work, cancel=cancel)
            del mic
        else:
            raise MediaError("The original no longer has the second audio track the speaker echo was removed with.")
    else:
        x = pipeline.decode_for_clean(ff, src, info, engine.MAX_SECONDS, work_dir=work)
    if job.samples and len(x) != job.samples:
        raise MediaError("The original no longer matches the saved copies (it is not the same length), so it can't be changed and saved again.")
    return x


def _sound(
    job: copies.JobCopies, wanted: Wanted, p: Plan, ff, x: np.ndarray, work: Path, cancel: threading.Event, threads: int,
    progress: engine.Progress, used: dict, keeper: copies.Keeper, inputs=None,
) -> np.ndarray:
    """The cleaned sound for `wanted`, (samples, channels) float32."""
    if p.sound == SOUND_EDITOR:
        tl = timeline_of(job, wanted)
        if tl is None:
            raise MediaError("The timeline is not valid.")
        y, done, decided, notice, _ = editor_job.render_timeline(
            tl, inputs, job.block_cache(ff), work=work, threads=threads, known=job.editor.get("resolved"), progress=progress, cancel=cancel,
            out=pipeline.output_buffer(x, work))
        used.update(timeline=done, decided=decided, notice=notice, noise=None)
        return y
    opts = pipeline.CleanOptions(
        atten_lim_db=wanted.atten_lim_db, leveling=wanted.leveling, loudness_lufs=engine.LOUDNESS_TARGETS[wanted.loudness],
    )
    if p.sound in (SOUND_BLEND, SOUND_LEVEL_REBUILT):
        kind = copies.NOISE if p.sound == SOUND_BLEND else copies.REBUILT
        entry = job.entry(kind)
        if entry is None:
            raise MediaError("The saved copy is not there any more.")
        y = pipeline.output_buffer(x, work)
        y[:] = 0.0
        kept.load(ff, entry, job.dir, y, work / "kept-decoded.f32")
        if p.sound == SOUND_BLEND:
            plan_ = entry.get("plan")
            if not isinstance(plan_, list) or len(plan_) != x.shape[1]:
                raise MediaError("The saved copy does not fit the original (a different number of channels).")
            used["noise"] = entry.get("profile")
            return pipeline.redo_denoised(x, y, [None if c is None else int(c) for c in plan_], opts, progress=progress, report=used)
        return pipeline.redo_rebuilt(x, y, opts, progress=progress, report=used)
    # a model has to run again (Voice rebuild without a saved rebuilt voice; noise removal in a job that has none)
    rebuild = p.sound == SOUND_REBUILD
    noise = next((n for n in (job.made.get("noise_option"), job.made.get("noise")) if n in ("light", "heavy")), "auto")
    opts = pipeline.CleanOptions(
        atten_lim_db=wanted.atten_lim_db, leveling=wanted.leveling, loudness_lufs=engine.LOUDNESS_TARGETS[wanted.loudness],
        noise=noise, restore_voice=rebuild,
    )
    return pipeline.clean_audio(
        x, resources.libdf_path(), resources.model_path(), opts, threads=threads, progress=progress, cancel=cancel,
        out=pipeline.output_buffer(x, work), heavy_model_path=resources.heavy_noise_model_path(),
        restore_model_dir=resources.voice_model_dir(), report=used, keep=engine.keep_copies(keeper, x, used, job.echo),
    )


@dataclass
class Prepared:
    """The sound of a job made ready for live listening: the saved copy and the original lined up in scratch files."""

    stream: blend.BlendStream
    work: Path

    def close(self) -> None:
        """Let go of the scratch files (the stream must not be used afterwards)."""
        self.stream = None  # type: ignore[assignment]
        try:
            shutil.rmtree(self.work)
        except FileNotFoundError:
            pass
        except OSError:
            threading.Thread(target=engine._remove_work_dir, args=(self.work,), name="scratch-cleanup", daemon=True).start()


def can_slide(job: copies.JobCopies) -> bool:
    """The job has what the strength slider blends (a saved noise removal), apart from the original itself being unchanged."""
    return not job.has_blocks and job.has(copies.NOISE) and not (job.echo and not job.has(copies.INPUT))


def prepare_stream(
    job: copies.JobCopies, *, cancel: threading.Event | None = None, scratch_parent: Path | None = None, exact: bool = False,
    threads: int = 2,
) -> Prepared:
    """Decode the original and the saved noise removal into scratch files and line them up for `blend.BlendStream` (the same way the
    pipeline lined them up: same decoding, same length; the speaker echo taken out when it was). Raises MediaError (fit for the user)
    when the original is gone or changed, the copy is missing, or the disk is too full."""
    cancel = cancel or threading.Event()
    check = job.check_original()
    if not check.ok:
        raise MediaError(check.text)
    entry = job.entry(copies.NOISE)
    if entry is None or not can_slide(job):
        raise MediaError("This job has no saved noise removal to mix at other strengths.")
    ff = engine.ffmpeg(threads, cancel)
    src = job.original_path
    info = engine.probe(src, ff)
    folder = scratch_parent or job.dir.parent
    folder.mkdir(parents=True, exist_ok=True)
    parent = engine._work_parent(folder, engine.estimated_scratch_gb(info.duration, info.audio_channels))
    need_gb = engine.estimated_scratch_gb(info.duration, info.audio_channels) * engine.SCRATCH_SAFETY
    try:
        free_gb = shutil.disk_usage(parent).free / 1e9
    except OSError:
        free_gb = need_gb
    if free_gb < need_gb:
        raise MediaError(f"There is not enough free disk space to get the sound ready for trying other strengths (about {need_gb:.1f} GB "
                         f"is needed, {free_gb:.1f} GB is free). Saving at another strength still works.")
    work = Path(tempfile.mkdtemp(dir=parent, prefix=f"{engine._WORK_PREFIX}{os.getpid()}-"))
    try:
        x = _recording(job, ff, src, info, work, cancel)
        _check_cancel(cancel)
        plan_ = entry.get("plan")
        if not isinstance(plan_, list) or len(plan_) != x.shape[1]:
            raise MediaError("The saved copy does not fit the original (a different number of channels).")
        plan_ = [None if c is None else int(c) for c in plan_]
        full = pipeline.output_buffer(x, work)
        full[:] = 0.0
        kept.load(ff, entry, job.dir, full, work / "kept-decoded.f32")
        _check_cancel(cancel)
        return Prepared(blend.BlendStream(full, x, plan_, exact=exact), work)
    except BaseException:
        x = full = None  # noqa: F841 - the mappings must be let go before the folder can be removed (Windows)
        try:
            shutil.rmtree(work)
        except OSError:
            threading.Thread(target=engine._remove_work_dir, args=(work,), name="scratch-cleanup", daemon=True).start()
        raise
