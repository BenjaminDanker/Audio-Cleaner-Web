"""Base abstractions for pluggable audio / video AI inference tasks.

Design goals:
  * Minimal, stable interface (process() returning output path).
  * Support both single-step (denoise) and future multi-step composite pipelines.
  * Allow lazy / shared model initialization (cache-heavy models loaded once).

Key concepts:
  * MediaTask: atomic unit that accepts standardized inputs (paths) and returns an artifact path.
  * TaskFactory: creates or returns a (possibly cached) MediaTask instance.
  * Registry: maps processing_type strings to factories.

Extension guideline:
  1. Implement a subclass of MediaTask.
  2. Register it in ai.registry at import time (side-effect registration) or via an init hook.
  3. JobRecord.processing_type should match the registered key (case-insensitive).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Any
import logging
import tempfile

logger = logging.getLogger(__name__)


class ProgressCallback(Protocol):
    async def __call__(self, pct: int) -> None:  # pragma: no cover - structural protocol only
        ...


@dataclass
class MediaTaskContext:
    """Context supplied to task execution.

    work_dir: dedicated temp directory for this job (task may create subdirs).
    attenuation_db: optional user-provided attenuation parameter (legacy for denoise tasks).
    extra: future-proof dictionary for additional parameters (e.g., language for ASR, model size, etc.).
    """

    work_dir: str
    attenuation_db: Optional[int] = None
    extra: dict[str, Any] | None = None


class MediaTask(ABC):
    """Atomic inference unit.

    Contract:
      * Implement process(input_path, ctx, progress_cb) -> output_path (string path to produced file).
      * MUST NOT mutate input file in-place; always write new artifact.
      * Should periodically invoke progress_cb (coarse 0..100) when meaningful.
    """

    kind: str = "generic"  # Override with 'audio' or 'video' if desired.

    @abstractmethod
    def process(
        self,
        input_path: str,
        ctx: MediaTaskContext,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> str:  # return output artifact path
        raise NotImplementedError

    # Optional hook for heavy model warm-up.
    def warmup(self) -> None:  # pragma: no cover - default no-op
        return


TaskFactory = Callable[[], MediaTask]


class TaskRegistrationError(RuntimeError):
    pass


class InferenceRegistry:
    """Simple in-memory registry for MediaTasks (singleton usage expected)."""

    def __init__(self):
        self._factories: dict[str, TaskFactory] = {}

    def register(self, name: str, factory: TaskFactory, overwrite: bool = False) -> None:
        key = name.lower().strip()
        if not overwrite and key in self._factories:
            raise TaskRegistrationError(f"Task '{key}' already registered")
        self._factories[key] = factory
        logger.debug("Registered inference task '%s'", key)

    def create(self, name: str) -> MediaTask:
        key = name.lower().strip()
        if key not in self._factories:
            raise TaskRegistrationError(f"Unknown inference task '{name}'")
        task = self._factories[key]()
        return task

    def available(self) -> list[str]:  # pragma: no cover - trivial
        return sorted(self._factories.keys())


# Global default registry instance
registry = InferenceRegistry()


def run_single_task(
    processing_type: str,
    input_file: str,
    attenuation_db: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> str:
    """Helper to execute a single registered task with auto temp dir management.

    Returns path to produced artifact.
    """
    task = registry.create(processing_type)
    with tempfile.TemporaryDirectory(prefix=f"aitask_{processing_type}_") as td:
        ctx = MediaTaskContext(work_dir=td, attenuation_db=attenuation_db, extra=extra)
        return task.process(input_file, ctx, progress_cb)
