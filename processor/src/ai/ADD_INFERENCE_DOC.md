# AI Task Plugin System

This directory contains the pluggable inference task framework.

## Core Concepts
- **MediaTask** (`base.py`): Abstract base class. Implement `process(input_path, ctx, progress_cb)` and return the path to a produced output artifact.
- **MediaTaskContext**: Supplies a per-job temporary working directory plus optional parameters (e.g. attenuation_db).
- **Registry**: Global mapping from a `processing_type` string (lowercase) to a task factory. Jobs specify `processing_type` and the processor dispatches accordingly.

## Built-in Tasks
| processing_type | File                     | Description                                                      |
|-----------------|--------------------------|------------------------------------------------------------------|
| `denoise`       | `audio_denoise_dfnet.py` | DeepFilterNet3 denoising (extraction + enhancement + remux).     |
| `passthrough`   | `audio_passthrough.py`   | Copies input media unchanged (reference / health-check task).    |

## Adding a New Task
1. Create a new file (e.g. `audio_transcribe.py`).
2. Implement a subclass of `MediaTask`:
   ```python
   from ai.base import MediaTask, MediaTaskContext, registry
   class TranscribeTask(MediaTask):
       kind = "audio"
       def process(self, input_path: str, ctx: MediaTaskContext, progress_cb=None) -> str:
           # ... do work ...
           return output_path
   registry.register("transcribe", lambda: TranscribeTask())
   ```
3. Import the module in `ai/__init__.py` (or ensure it is imported elsewhere) so registration runs.
4. Ensure jobs set `processing_type` to the new key (e.g. `transcribe`).

## Progress Reporting
Tasks may call `progress_cb(pct)` (0–100). Keep it sparse: a few meaningful milestones (e.g. 10 after decode, 30 after enhancement, 85 before finalize). The `processor_main` maps task-level percentages to job progress (see `progress_milestones` dict). If you add new tasks, emit similar coarse checkpoints so jobs display consistent progress behavior.

## Model Loading & Reuse
The DeepFilterNet task uses a module-level singleton so large weights load only once (see lazy init in `audio_denoise_dfnet.py`). Pattern:
```python
_MODEL = None
def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = HeavyModel.load(...)
    return _MODEL
```
Call `_get_model()` inside `process()` to ensure first-use initialization only.

### Why `deepfilternet.py` & task file are separate
`deepfilternet.py` wraps raw model lifecycle; `audio_denoise_dfnet.py` focuses on pipeline (extraction, encode/remux, registry). This separation lets future tasks reuse the model wrapper without duplicating logic. If you prefer a single file, you can merge them—just keep singleton initialization.

## Guidelines
- Never mutate the original input file. Write outputs inside `ctx.work_dir`.
- Keep external dependencies minimal; reuse shared utilities (`media_extractor`, ffmpeg logic already inside tasks where needed).
- Fail fast with clear exceptions; upstream processor marks job failed.
- Register early (import side-effect) so the worker sees the task when processing starts.
- For heavy GPU / model init, delay load until first call or provide an explicit warmup method.

## Orchestrator Entry Point
`media_processor.py` (formerly `video_handler.py`) picks a task by `processing_type` and invokes `task.process()`. It is intentionally thin; new behavior should live in tasks, not in the orchestrator.

## Listing Available Tasks
From Python:
```python
from ai.base import registry
print(registry.available())
```

## Removing / Replacing a Task
Re-register with `overwrite=True` or delete the file and remove its import. Restart the worker process to clear it from the in-memory registry.

## Quick Checklist for a New Task
1. Create file `<name>.py`.
2. Implement subclass of `MediaTask`.
3. Use singleton pattern for big models.
4. Emit a few `progress_cb` milestones.
5. `registry.register("<key>", lambda: YourTask())`.
6. Import in `ai/__init__.py`.
7. Submit a job with `processing_type` set to your key.
