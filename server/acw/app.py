"""FastAPI application: `uvicorn acw.app:app --host 127.0.0.1 --port 8790`.

The process binds to loopback only. Public traffic reaches it through Caddy on
the VPS and an SSH reverse tunnel, so nothing listens on the home LAN.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from . import alerts, auth, db, edit_subtitles, httpstack, jobs, retention, webhooks
from . import health as health_checks
from .api import auth_routes, billing_routes, contact_routes, downloads_routes, jobs_routes, keys_routes
from . import translate
from .asr import openai_asr
from .billing import ledger, stripe_service
from .config import get_settings

log = logging.getLogger("acw")


PREMIUM_JOBS_AT_ONCE = 2  # extra-accurate transcriptions (long OpenAI calls per file)
TRANSLATION_JOBS_AT_ONCE = 2  # file translations: places of their own, so a 4 hour talk in three languages (40 minutes) can't hold up
#                               every transcription and a transcription backlog can't hold up translations
_cloud_tasks: dict[asyncio.Task, str] = {}  # running task -> its phase ("asr" or "translate")
_CLOUD_STAGES = {"asr": "Transcribing", "translate": "Translating"}


def _needs_transcription(job: dict) -> bool:
    """The job's cloud work starts with extra-accurate transcription: some subtitle part's speech chunks wait for it (every job makes its
    subtitles part by part)."""
    params = json.loads(job["params_json"])
    return edit_subtitles.has_subtitle_layer(params) and edit_subtitles.subtitle_cloud_phase(job["id"], params) == "asr"


def _claim_cloud_jobs(limit: int, phase: str | None = None) -> list[dict]:
    """Take up to `limit` waiting cloud jobs (oldest first) that are in `phase` ("asr": transcription still to do;
    "translate": only translations are left), any phase if None. Each is marked as being worked on and carries `_phase`."""
    claimed = []
    for job in db.all_rows(
        "SELECT * FROM jobs WHERE status = 'cloud_asr' AND stage NOT IN ('Transcribing', 'Translating') ORDER BY queued_at LIMIT 100"
    ):
        if len(claimed) >= limit:
            break
        job = dict(job)
        job_phase = "asr" if _needs_transcription(job) else "translate"
        if phase is not None and job_phase != phase:
            continue
        n = db.execute(
            "UPDATE jobs SET stage = ?, heartbeat_at = ? WHERE id = ? AND status = 'cloud_asr' AND stage NOT IN ('Transcribing', 'Translating')",
            (_CLOUD_STAGES[job_phase], db.now(), job["id"]),
        ).rowcount
        if n:
            claimed.append({**job, "_phase": job_phase})
    return claimed


def _cloud_done(job: dict) -> None:
    db.execute(
        "UPDATE jobs SET status = 'queued', next_stage = 'finalize', stage = 'Finishing', progress = 90,"
        " queued_at = ? WHERE id = ? AND status = 'cloud_asr'",
        (db.now(), job["id"]),
    )


def _beat(job_id: str) -> None:
    db.execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ? AND status = 'cloud_asr'", (db.now(), job_id))


def _set_stage(job_id: str, stage: str) -> None:
    db.execute("UPDATE jobs SET stage = ?, heartbeat_at = ? WHERE id = ? AND status = 'cloud_asr'", (stage, db.now(), job_id))


def _clip_part(part_dir, e: dict) -> None:
    """The text of an Accurate part, which was made from its speech chunks, cut to the part (editing.py)."""
    from acw_common import captions
    from acw_common import subtitles as st

    path = part_dir / "segments.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = st.clip_segments(captions.segments_from_json(data.get("segments") or []), e["start"], e["end"])
    path.write_text(json.dumps({**data, "segments": captions.segments_to_json(segs)}), encoding="utf-8")


async def _run_subtitle_parts(job: dict, params: dict) -> None:
    """The cloud work of a job whose subtitles are made part by part: the Accurate parts' transcription (one phase), then the
    translations of every part (the other phase), each into the part's own folder (editing.subtitle_cloud_work). A translation that can't
    be made is left out, said in the notice and not charged."""
    work = await asyncio.to_thread(edit_subtitles.subtitle_cloud_work, job["id"], params)
    base = jobs.job_dir(job["id"]) / edit_subtitles.SUBS
    if work["asr"]:
        for e in work["asr"]:
            d = base / e["key"]
            await openai_asr.run_for_job(d, e["language"], (e["end"] - e["start"]) + 30)
            await asyncio.to_thread(_clip_part, d, e)
        again = await asyncio.to_thread(edit_subtitles.subtitle_cloud_work, job["id"], params)
        if again["translate"]:
            await asyncio.to_thread(_set_stage, job["id"], "Waiting to translate")
            return
    else:
        await asyncio.to_thread(_set_stage, job["id"], "Translating")
        failed: list[str] = []
        for e, targets in work["translate"]:
            done = await translate.translate_job(base / e["key"], targets)
            failed += [t for t in targets if t not in done]
        if failed:
            what = " and ".join(translate.LANGUAGE_NAMES.get(t, t) for t in dict.fromkeys(failed))
            await asyncio.to_thread(jobs.set_notice, job["id"], f"The {what} translation couldn’t be made this time for some parts, so you weren’t charged for it.")
    await asyncio.to_thread(_cloud_done, job)


async def _run_cloud_job(job: dict) -> None:
    """The steps of a job that need the internet: extra-accurate transcription, then translations. Work already
    done is kept on disk, so a restart or a failed attempt doesn't pay for it twice. One call does one phase:
    after the transcription a job that also wants translations goes back to waiting ("Waiting to translate") and is
    taken up again in the translation places, so the two kinds of work never wait for each other's places."""

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(60)
            await asyncio.to_thread(_beat, job["id"])

    beat = asyncio.create_task(heartbeat())
    try:
        params = json.loads(job["params_json"])
        if not edit_subtitles.has_subtitle_layer(params):
            raise ValueError("a cloud job without subtitle parts")  # (every job makes its subtitles part by part)
        await _run_subtitle_parts(job, params)
    except Exception:  # noqa: BLE001
        log.exception("cloud ASR failed for %s", job["id"])
        await asyncio.to_thread(jobs.finish_failed, job, "Extra accurate transcription failed; you were not charged.")
    finally:
        beat.cancel()


async def _cloud_asr_loop() -> None:
    while True:
        await asyncio.sleep(3)
        try:
            for t in [t for t in _cloud_tasks if t.done()]:
                del _cloud_tasks[t]
            for phase, places in (("asr", PREMIUM_JOBS_AT_ONCE), ("translate", TRANSLATION_JOBS_AT_ONCE)):
                free = places - sum(1 for p in _cloud_tasks.values() if p == phase)
                if free > 0:
                    for job in await asyncio.to_thread(_claim_cloud_jobs, free, phase):
                        _cloud_tasks[asyncio.create_task(_run_cloud_job(job))] = phase
        except Exception:  # noqa: BLE001
            log.exception("cloud ASR loop failed")


def _housekeeping() -> None:
    steps = (
        auth.purge_expired,
        # Cloud ASR that never finished (process restart mid-request): retry.
        lambda: db.execute(
            "UPDATE jobs SET stage = CASE stage WHEN 'Translating' THEN 'Waiting to translate' ELSE 'Waiting to transcribe' END"
            " WHERE status = 'cloud_asr' AND stage IN ('Transcribing', 'Translating') AND heartbeat_at < ?",
            (db.now() - 600,),
        ),
        ledger.reap_stale_holds,  # money reserved by things that are over (live sessions: the live process, acw.live_app)
        retention.sweep,  # privacy retention (docs/operations.md)
        # Expired outputs, abandoned uploads, orphan directories, jobs whose worker died. The workers also call it, but
        # only between jobs: while every worker is busy with a long file nothing else would run it. Safe to run from
        # several processes at once: every step is a guarded UPDATE or an idempotent transaction (finish_failed).
        jobs.sweep,
    )
    for step in steps:
        try:
            step()
        except Exception:  # noqa: BLE001
            log.exception("housekeeping step failed")


async def _housekeeping_loop() -> None:
    while True:
        await asyncio.sleep(30)
        try:
            await asyncio.to_thread(_housekeeping)
            if stripe_service.enabled():
                await asyncio.to_thread(stripe_service.reconcile)  # payments whose webhook never came
                for user_id in await asyncio.to_thread(stripe_service.users_needing_reload):
                    await asyncio.to_thread(stripe_service.run_auto_reload, user_id)
                for payment_id in await asyncio.to_thread(stripe_service.pending_tax_transactions):
                    await asyncio.to_thread(stripe_service._record_tax_transaction, payment_id)
        except Exception:  # noqa: BLE001
            log.exception("housekeeping failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # A previous process died while transcribing: pick those jobs up again right away (finished segments are kept).
    db.execute(
        "UPDATE jobs SET stage = CASE stage WHEN 'Translating' THEN 'Waiting to translate' ELSE 'Waiting to transcribe' END"
        " WHERE status = 'cloud_asr' AND stage IN ('Transcribing', 'Translating')"
    )
    try:
        await asyncio.to_thread(alerts.web_started)  # mails if the last run ended uncleanly
    except Exception:  # noqa: BLE001
        log.exception("alert start-up check failed")
    # Job callbacks (docs/api.md) go out from this process only: the workers and the bulk process have no network.
    tasks = [asyncio.create_task(_cloud_asr_loop()), asyncio.create_task(_housekeeping_loop()), asyncio.create_task(webhooks.run_loop())]
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        try:
            alerts.web_stopping()  # clean-stop marker: its absence at next start means a crash
        except Exception:  # noqa: BLE001
            log.exception("could not record clean stop")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Audio Cleaner",
        lifespan=lifespan,
        docs_url=None if settings.production else "/api/docs",
        redoc_url=None,
        openapi_url=None if settings.production else "/api/openapi.json",
        root_path=settings.base_path,
    )

    httpstack.install(app, settings)

    @app.get("/api/health")
    def health(request: Request, deep: int = 0, principal: auth.Principal | None = Depends(auth.optional_principal)) -> dict:
        db.one("SELECT 1")
        if not deep:
            return {"ok": True}
        # Details about the machine: from this machine itself (no proxy header: not via the tunnel) or an admin.
        local = request.client is not None and request.client.host in ("127.0.0.1", "::1") and "x-forwarded-for" not in request.headers
        if not (local or (principal is not None and principal.is_admin and principal.via == "session")):
            return {"ok": True}
        return health_checks.deep()

    for r in (
        auth_routes.router,
        auth_routes.me_router,
        billing_routes.router,
        billing_routes.webhook_router,
        contact_routes.router,
        downloads_routes.router,
        jobs_routes.router,
        keys_routes.router,
    ):
        app.include_router(r)

    if settings.web_dist and settings.web_dist.is_dir():
        _mount_spa(app, settings.web_dist)
    return app


def _mount_spa(app: FastAPI, dist: Path) -> None:
    """Dev convenience: serve the built frontend. In production Caddy on the
    VPS serves it so the free browser tier never touches the home link."""
    dist = dist.resolve()
    index = dist / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(status_code=404)
        candidate = (dist / path).resolve()
        if path and candidate.is_file() and dist in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


app = create_app()
