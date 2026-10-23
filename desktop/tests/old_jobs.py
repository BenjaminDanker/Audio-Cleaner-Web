"""Jobs as the app made them before the block cache: whole saved copies (noise.flac at full strength, rebuilt.flac, input.flac without the
speaker echo) instead of blocks. Built here with the same pieces the old engine used (kept.save + JobCopies.put_audio): the noise removal at full
strength is the editor's own render of a timeline with only the noise layer at 100 dB, which is what the old copy held."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from acw_common import editor, kept, pipeline
from acw_common.timeline import Timeline
from audio_cleaner_desktop import copies, editor_job, engine

SR = pipeline.SAMPLE_RATE


def decode_original(path: Path) -> np.ndarray:
    ff = engine.ffmpeg()
    return pipeline.decode_for_clean(ff, path, ff.probe(path), 600)


def render_whole(x: np.ndarray, **layers) -> np.ndarray:
    """The whole file `x` (samples, channels) rendered with these layers (Timeline.whole_file keywords), nothing cached."""
    inputs = editor.Inputs(x, "old-job-test")
    return editor.render(Timeline.whole_file(inputs.seconds, **layers), inputs, runner=editor_job.runner(2))


def old_job(clip: Path, tmp_path: Path, *, noise: bool = True, rebuilt: np.ndarray | None = None, remover: str = "light",
            made: dict | None = None) -> tuple[copies.JobCopies, np.ndarray, np.ndarray | None]:
    """An old job of `clip`: its manifest, a noise copy (when `noise`) and a rebuilt copy (`rebuilt`: the samples it holds). Returns
    (the job, the recording, the noise copy's samples). The job has no blocks."""
    ff = engine.ffmpeg()
    x = decode_original(clip)
    folder = copies.new_job_dir(tmp_path / "jobs")
    job = copies.JobCopies.create(folder, clip, copies.fingerprint(clip), engine.media_facts(ff.probe(clip)))
    job.data["samples"] = len(x)
    full = None
    if noise:
        full = render_whole(x, noise={"strength_db": 100.0, "remover": remover}, leveling=False)
        job.put_audio(copies.NOISE, kept.save(ff, full, list(range(x.shape[1])), folder / "noise"), plan=list(range(x.shape[1])),
                      profile=remover, echo=False)
    if rebuilt is not None:
        job.put_audio(copies.REBUILT, kept.save(ff, rebuilt, list(range(x.shape[1])), folder / "rebuilt"), plan=list(range(x.shape[1])), echo=False)
    job.made.update({"atten_lim_db": 100.0, "rebuilt": rebuilt is not None and not noise, "leveling": False, "loudness": "podcast",
                     "noise": remover, "echo": False, **(made or {})})
    job.save()
    return job, x, full
