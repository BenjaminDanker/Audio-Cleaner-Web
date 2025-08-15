from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import JSONResponse
from typing import Dict, Optional, List
import asyncio
import json
import os
import sys
import numpy as np
import time
import io
import soundfile as sf
from datetime import datetime, timezone

# Make processor/src importable
CUR_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CUR_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from ai.audio_clarity_pipeline import process_stream_chunk, StreamState  # type: ignore
from ai.asr_pipeline import _get_openai_client, _cleanup_segments_with_llm, _translate_segments, SubtitleSegment  # type: ignore
from captions.caption_encoder import write_srt, write_vtt, Segment as CapSegment  # type: ignore

app = FastAPI(title="Audio Cleaner Streaming Service", version="0.2.0")

class SessionState:
    def __init__(self, session_id: str):
        self.id = session_id
        self.ws = None  # type: Optional[WebSocket]
        self.sr = 16000
        self.langs = ["en"]  # type: List[str]
        self.user_id = None  # set from init message
        self.proc_state = None  # type: Optional[StreamState]
        self.processed_seconds = 0.0
        self.last_deducted_seconds = 0.0
        self.credits_cents_spent = 0
        self._buf = bytearray()
        self.token = None
        # ASR state
        self._pcm_ring = np.zeros(1, dtype=np.float32)
        self._pcm_samples = 0
        self._asr_running = False
        self._asr_last_run = 0.0
        self._asr_last_end_by_lang = {}
        self._asr_max_seconds = float(os.getenv("STREAM_ASR_BUFFER_SECONDS", "6"))
        self._asr_stride_seconds = float(os.getenv("STREAM_ASR_STRIDE_SECONDS", "2"))
        # Credit policy
        self._base_cents_per_min = float(os.getenv("STREAM_BASE_CENTS_PER_MINUTE", "10"))  # $0.10/min default
        self._extra_lang_cents_per_min = float(os.getenv("STREAM_EXTRA_LANG_CENTS_PER_MINUTE", "5"))
        self._low_credits_grace_sec = float(os.getenv("STREAM_LOW_CREDITS_GRACE_SECONDS", "8"))
        self._low_sent = False
        self._stop_sent = False
        # Transcript accumulation
        self._segments_by_lang = {}
        self._started_at = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

sessions: Dict[str, SessionState] = {}

@app.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok"})

@app.post("/stream/stop")
async def stop_stream(payload: dict = Body(...)):
    sid = str(payload.get("sessionId", ""))
    st = sessions.pop(sid, None)
    if st and st.ws:
        try:
            await st.ws.close()
        except Exception:
            pass
    # Persist transcript SRT/VTT per language to Blob Storage if configured
    urls = {}
    try:
        conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container = os.getenv("PROCESSED_CONTAINER_NAME", "processed-videos")
        if conn and st and st._segments_by_lang:
            from azure.storage.blob import BlobServiceClient  # type: ignore
            bsc = BlobServiceClient.from_connection_string(conn)
            cont = bsc.get_container_client(container)
            try:
                cont.create_container()
            except Exception:
                pass
            base = f"streams/{sid}/{int(time.time())}"
            # Write temp files in-memory
            for lang, segs in st._segments_by_lang.items():
                caps = [CapSegment(s.start, s.end, s.text) for s in segs]
                srt_buf = io.StringIO()
                vtt_buf = io.StringIO()
                # Use helpers to generate files on disk-like buffers
                # Workaround: encoder writes to file paths; use temp files
                import tempfile
                with tempfile.TemporaryDirectory() as td:
                    srt_path = os.path.join(td, f"{lang}.srt")
                    vtt_path = os.path.join(td, f"{lang}.vtt")
                    write_srt(caps, srt_path)
                    write_vtt(caps, vtt_path)
                    with open(srt_path, "rb") as f:
                        cont.upload_blob(f"{base}_{lang}.srt", f.read(), overwrite=True)
                    with open(vtt_path, "rb") as f:
                        cont.upload_blob(f"{base}_{lang}.vtt", f.read(), overwrite=True)
                    sas_srt = cont.get_blob_client(f"{base}_{lang}.srt").url
                    sas_vtt = cont.get_blob_client(f"{base}_{lang}.vtt").url
                    urls[lang] = {"srt": sas_srt, "vtt": sas_vtt}
    except Exception:
        urls = {}
    return JSONResponse({
        "sessionId": sid,
        "processedSeconds": round(st.processed_seconds, 2) if st else 0.0,
        "subtitles": urls,
    })

@app.websocket("/stream/{session_id}")
async def stream_ws(websocket: WebSocket, session_id: str):
    # Require API key header for non-browser clients
    api_key = websocket.headers.get("x-api-key") or websocket.headers.get("authorization", "").removeprefix("Bearer ")
    env_keys = set(filter(None, [os.getenv("STREAMING_API_KEY", "")] + [k.strip() for k in os.getenv("STREAMING_API_KEYS", "").split(",") if k.strip()]))
    if not api_key or (env_keys and api_key not in env_keys):
        # If no env keys configured, allow for local dev; otherwise enforce
        if env_keys:
            await websocket.close(code=4401)
            return
    # Basic session token check (query param t)
    token = websocket.query_params.get("t")
    if not token:
        # Allow during early local dev
        pass
    await websocket.accept()
    st = sessions.get(session_id) or SessionState(session_id)
    st.ws = websocket
    st.token = token
    sessions[session_id] = st
    try:
        # Protocol: first message should be text JSON { type: 'init', sr, languages[] }
        init_msg = await websocket.receive_text()
        try:
            init = json.loads(init_msg)
        except Exception:
            init = {}
        if isinstance(init, dict) and init.get("type") == "init":
            if isinstance(init.get("sr"), int):
                st.sr = int(init["sr"]) or 16000
            if isinstance(init.get("languages"), list):
                st.langs = [str(x) for x in init.get("languages") if isinstance(x, (str,))] or ["en"]
            if init.get("userId") and isinstance(init.get("userId"), str):
                st.user_id = init.get("userId")
        await websocket.send_text(json.dumps({"type": "ready", "sr": st.sr, "languages": st.langs}))

        # Receive binary float32 mono chunks; return processed audio (binary) and caption deltas (text)
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                raw = msg["bytes"]
                # Interpret as float32 little-endian mono PCM
                try:
                    x = np.frombuffer(raw, dtype=np.float32)
                except Exception:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Bad PCM chunk"}))
                    continue
                y, st.proc_state = process_stream_chunk(x, st.sr, st.proc_state, params=None)
                st.processed_seconds += float(y.shape[0]) / float(st.sr)
                # Send processed audio back as binary
                await websocket.send_bytes(y.tobytes())
                # Credit ticking and low-credit signaling (best-effort; requires COSMOS_CONNECTION_STRING)
                await _tick_credits_and_maybe_signal(st)
                # Append to ASR buffer
                st._pcm_ring = np.concatenate([st._pcm_ring, x]) if st._pcm_samples else x.copy()
                st._pcm_samples = st._pcm_ring.shape[0]
                # Trim ring buffer to max seconds
                max_samples = int(st._asr_max_seconds * st.sr)
                if st._pcm_samples > max_samples:
                    st._pcm_ring = st._pcm_ring[-max_samples:]
                    st._pcm_samples = st._pcm_ring.shape[0]
                # Launch ASR task every stride if not already running
                now = time.time()
                if (not st._asr_running) and (now - st._asr_last_run >= st._asr_stride_seconds) and st._pcm_samples >= int(1.0 * st.sr):
                    st._asr_running = True
                    st._asr_last_run = now
                    asyncio.create_task(_run_asr_and_emit(st))
            elif "text" in msg and msg["text"] is not None:
                # Handle control messages (noop for now)
                t = msg["text"]
                if t == "ping":
                    await websocket.send_text("pong")
                else:
                    # ignore or future commands
                    pass
            else:
                await asyncio.sleep(0)
    except WebSocketDisconnect:
        pass
    finally:
        # Keep session so /stop can report; or clean up if idle
        sessions[session_id] = st


async def _run_asr_and_emit(st: SessionState):
    try:
        # Snapshot buffer
        pcm = st._pcm_ring.copy()
        buf_sec = float(pcm.shape[0]) / float(st.sr)
        if pcm.shape[0] == 0:
            return
        # Write in-memory WAV
        bio = io.BytesIO()
        sf.write(bio, pcm, st.sr, format="WAV", subtype="PCM_16")
        bio.seek(0)
        # Call Azure OpenAI Whisper
        client = _get_openai_client()
        whisper_depl = os.getenv("AZURE_OPENAI_WHISPER_DEPLOYMENT", "whisper-1")
        resp = client.audio.transcriptions.create(
            model=whisper_depl,
            file=("chunk.wav", bio.getvalue()),
            response_format="verbose_json",
            temperature=0,
        )
        data = resp if isinstance(resp, dict) else json.loads(resp.model_dump_json())
        segs: List[SubtitleSegment] = []
        for s in data.get("segments", []):
            segs.append(SubtitleSegment(start=float(s.get("start", 0.0)), end=float(s.get("end", 0.0)), text=s.get("text", "").strip()))
        if not segs:
            return
        # Cleanup
        cleaned = _cleanup_segments_with_llm(segs)
        # Time offset so segments match session timeline
        offset = max(0.0, st.processed_seconds - buf_sec)
        def _offset_segments(ss: List[SubtitleSegment]) -> List[SubtitleSegment]:
            return [SubtitleSegment(start=s.start + offset, end=s.end + offset, text=s.text) for s in ss]
        base = _offset_segments(cleaned)
        # Build per-language
        by_lang: Dict[str, List[dict]] = {}
        # Primary (detected language unknown here; treat as 'orig')
        primary_lang = "orig"
        last_end = st._asr_last_end_by_lang.get(primary_lang, 0.0)
        new_primary = [s for s in base if s.end > last_end + 0.05]
        by_lang[primary_lang] = [{"start": s.start, "end": s.end, "text": s.text} for s in new_primary]
        if new_primary:
            st._asr_last_end_by_lang[primary_lang] = max(s.end for s in new_primary)
        # Translations
        for lang in st.langs:
            if lang == primary_lang:
                continue
            try:
                translated = _translate_segments(cleaned, lang)
                translated = _offset_segments(translated)
            except Exception:
                translated = base
            last_end_l = st._asr_last_end_by_lang.get(lang, 0.0)
            new_tr = [s for s in translated if s.end > last_end_l + 0.05]
            by_lang[lang] = [{"start": s.start, "end": s.end, "text": s.text} for s in new_tr]
            if new_tr:
                st._asr_last_end_by_lang[lang] = max(s.end for s in new_tr)
        # Send delta if any
        any_new = any(len(v) > 0 for v in by_lang.values())
        if any_new and st.ws:
            # Accumulate for final persistence
            for lang, items in by_lang.items():
                st._segments_by_lang.setdefault(lang, [])
                for it in items:
                    st._segments_by_lang[lang].append(SubtitleSegment(start=it["start"], end=it["end"], text=it["text"]))
            await st.ws.send_text(json.dumps({"type": "delta", "segmentsByLang": by_lang, "isFinal": False}))
    except Exception as e:
        try:
            if st.ws:
                await st.ws.send_text(json.dumps({"type": "error", "message": f"asr_failed: {str(e)}"}))
        except Exception:
            pass
    finally:
        st._asr_running = False


async def _tick_credits_and_maybe_signal(st: SessionState):
    try:
        # Effective rate per minute = base + extra per additional language
        add_langs = max(0, len([l for l in st.langs if l != 'orig']) - 0)
        rate_cpm = st._base_cents_per_min + (add_langs * st._extra_lang_cents_per_min)
        # Deduct every ~5s of processed audio to reduce write pressure
        if st.processed_seconds - st.last_deducted_seconds >= 5.0:
            inc_sec = st.processed_seconds - st.last_deducted_seconds
            inc_cents = int(round((inc_sec / 60.0) * rate_cpm))
            st.last_deducted_seconds = st.processed_seconds
            if inc_cents > 0 and st.user_id:
                conn = os.getenv("COSMOS_CONNECTION_STRING")
                db_name = os.getenv("COSMOS_DB_NAME", "AudioCleanerDB")
                if conn:
                    from azure.cosmos import CosmosClient  # type: ignore
                    cli = CosmosClient.from_connection_string(conn)
                    db = cli.get_database_client(db_name)
                    accounts = db.get_container_client('accounts')
                    txns = db.get_container_client('transactions')
                    try:
                        acc = accounts.read_item(st.user_id, st.user_id)
                        bal = int(acc.get('balance', 0))
                        new_bal = max(0, bal - inc_cents)
                        acc['balance'] = new_bal
                        acc['updatedAt'] = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
                        accounts.upsert_item(acc)
                        st.credits_cents_spent += inc_cents
                        tx = {
                            'id': f"txn_stream_{st.id}_{int(time.time())}",
                            'userId': st.user_id,
                            'type': 'streaming-tick',
                            'amount': inc_cents,
                            'description': f'streaming {inc_sec:.1f}s @ {rate_cpm} cpm',
                            'sessionId': st.id,
                            'createdAt': datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                        }
                        txns.upsert_item(tx)
                        # Low/stop signaling based on zero balance
                        if new_bal <= 0 and not st._low_sent and st.ws:
                            st._low_sent = True
                            await st.ws.send_text(json.dumps({"type": "LOW_CREDITS", "remainingCents": 0}))
                        if new_bal <= 0 and not st._stop_sent and st.ws:
                            st._stop_sent = True
                            await st.ws.send_text(json.dumps({"type": "STOP"}))
                            try:
                                await st.ws.close()
                            except Exception:
                                pass
                    except Exception:
                        # If deduction fails, do not crash stream; optionally log
                        pass
    except Exception:
        pass
