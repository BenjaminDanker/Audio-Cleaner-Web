import asyncio
import logging
import random
import time
from typing import Optional
from datetime import datetime, timezone
from job_models import JobRecord

logger = logging.getLogger(__name__)

class JobStore:
    """Combined job persistence + progress throttling.

    Exposes async methods but internally uses thread executor for sync Cosmos SDK.
    """

    def __init__(self, container, loop: Optional[asyncio.AbstractEventLoop] = None, progress_step: int = 5):
        self._container = container
        self._loop = loop or asyncio.get_event_loop()
        self._progress_step = progress_step
        self._last_progress = {}  # (job_id,user_id) -> last pct

    # ---- Internal sync helpers ----
    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

    def _sync_read(self, job_id: str, partition_key: str):
        return self._container.read_item(job_id, partition_key)

    def _sync_update(self, job_id: str, partition_key: str, status: str, fields: dict):
        job = self._container.read_item(job_id, partition_key)
        job['status'] = status
        job['updatedAt'] = self._now_iso()
        if status == 'processing' and 'startedAt' not in job:
            job['startedAt'] = self._now_iso()
        elif status in ('completed', 'failed'):
            job['completedAt'] = self._now_iso()
        for k, v in fields.items():
            job[k] = v
        self._container.upsert_item(job)

    def _sync_update_with_retry(self, job_id: str, partition_key: str, status: str, fields: dict, max_attempts: int = 4):
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                self._sync_update(job_id, partition_key, status, fields)
                return
            except Exception as e:  # Broad except intentionally: Cosmos transient errors vary
                last_err = e
                # Simple jittered backoff
                delay = min(0.2 * 2 ** (attempt - 1), 3.0) + random.random() * 0.1
                logging.warning(f"Retry {attempt}/{max_attempts} updating job {job_id} due to error: {e} (sleep {delay:.2f}s)")
                time.sleep(delay)
        raise last_err  # propagate after exhausting retries

    # ---- Public job API ----
    async def get(self, job_id: str, user_id: str):
        item = await self._loop.run_in_executor(None, self._sync_read, job_id, user_id)
        return JobRecord.from_cosmos(item)

    async def update_status(self, job_id: str, status: str, user_id: Optional[str], **fields):
        if user_id is None:
            logger.error(f"update_status called without user_id for job {job_id}; operation aborted to avoid wrong partition")
            return
        partition_key = user_id
        try:
            await self._loop.run_in_executor(None, self._sync_update_with_retry, job_id, partition_key, status, fields)
        except Exception as e:
            logger.error(f"Failed to update job {job_id} -> {status}: {e}")

    async def is_final(self, job_id: str, user_id: str) -> bool:
        try:
            item = await self._loop.run_in_executor(None, self._sync_read, job_id, user_id)
            return item.get('status') in ('completed', 'failed')
        except Exception:
            return False

    # ---- Progress throttling ----
    async def set_progress(self, job_id: str, user_id: str, pct: int):
        # Clamp progress to 0..100 to avoid accidental out-of-range values
        pct = max(0, min(100, int(pct)))
        key = (job_id, user_id)
        prev = self._last_progress.get(key, -self._progress_step)
        if pct - prev >= self._progress_step or pct >= 100:
            self._last_progress[key] = pct
            await self.update_status(job_id, 'processing', user_id, progress=pct)
