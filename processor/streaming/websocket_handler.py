"""WebSocket handler for streaming audio processing with security enhancements."""
from __future__ import annotations

import asyncio
import json
import os
import time
import io
import numpy as np
import soundfile as sf
import logging
from typing import List, Dict
from fastapi import WebSocket, WebSocketDisconnect

from session import SessionState, session_manager
from auth import verify_session_token
from billing import tick_credits_and_maybe_signal, probe_billing_and_update_state

# Import security utilities
import sys
from pathlib import Path
security_path = Path(__file__).parent
if str(security_path) not in sys.path:
    sys.path.append(str(security_path))
    
from security import (
    get_client_ip, 
    check_connection_limit, 
    track_connection,
    is_origin_allowed,
    validate_audio_data,
    log_security_event
)

# Import AI pipeline components
shared_dir = Path(__file__).parent.parent / "shared"
if str(shared_dir) not in sys.path:
    sys.path.append(str(shared_dir))

from ai.audio_clarity_pipeline import process_stream_chunk  # type: ignore
from ai.asr_pipeline import _get_speech_services_config, _cleanup_segments_with_llm, _translate_segments, SubtitleSegment  # type: ignore
from pricing import streaming_transcription_charge_cents, streaming_language_charge_cents  # type: ignore

logger = logging.getLogger(__name__)


async def handle_websocket_connection(websocket: WebSocket, session_id: str) -> None:
    """Handle a WebSocket connection for streaming audio processing with security."""
    client_ip = "unknown"
    
    try:
        # Extract client IP from WebSocket connection
        client_ip = websocket.client.host if websocket.client else "unknown"
        
        # Check connection limits per IP
        if not check_connection_limit(client_ip):
            logger.warning(f"Connection limit exceeded for IP: {client_ip}")
            await websocket.close(code=4429)  # Too Many Requests
            return
        
        # Validate origin if present (CORS-like check for WebSockets)
        origin = websocket.headers.get("origin")
        if origin and not is_origin_allowed(origin):
            logger.warning(f"Disallowed origin: {origin} from IP: {client_ip}")
            await websocket.close(code=4403)  # Forbidden
            return
        
        # Validate session token
        token = websocket.headers.get("x-session-token") or websocket.query_params.get("t")
        token_payload = verify_session_token(token, expected_session=session_id)
        if not token_payload:
            logger.warning(f"Invalid token for session {session_id} from IP: {client_ip}")
            await websocket.close(code=4401)  # Unauthorized
            return
            
        # Track connection
        track_connection(client_ip, connected=True)
        
        await websocket.accept()
        
        # Log successful connection
        user_id = token_payload.get("userId") if isinstance(token_payload, dict) else None
        logger.info(f"WebSocket connected: session={session_id}, user={user_id}, ip={client_ip}")
        
        # Get or create session
        st = session_manager.create_session(session_id)
        st.ws = websocket
        st.token = token
        
        # Extract userId from validated token for billing
        if isinstance(token_payload, dict):
            st.user_id = token_payload.get("userId")
        
        # Protocol: first message should be text JSON { type: 'init', sr, languages[] }
        init_msg = await websocket.receive_text()
        try:
            init = json.loads(init_msg)
        except Exception:
            init = {}
            
        if isinstance(init, dict) and init.get("type") == "init":
            # Validate sample rate
            sample_rate = init.get("sr")
            if isinstance(sample_rate, int) and 8000 <= sample_rate <= 48000:
                st.sr = sample_rate
            else:
                st.sr = 16000  # Safe default
                
            # Validate languages list
            languages = init.get("languages")
            if isinstance(languages, list) and len(languages) <= 5:  # Limit language count
                st.langs = [str(x) for x in languages if isinstance(x, str)][:5]  # Truncate to max 5
            else:
                st.langs = ["en"]  # Safe default
                
            if init.get("userId") and isinstance(init.get("userId"), str):
                st.user_id = init.get("userId")
                
        await websocket.send_text(json.dumps({
            "type": "ready", 
            "sr": st.sr, 
            "languages": st.langs,
            "maxChunkSize": 1024 * 1024  # Inform client of limits
        }))

        # Main processing loop
        while True:
            msg = await websocket.receive()
            
            if "bytes" in msg and msg["bytes"] is not None:
                await _process_audio_chunk(st, msg["bytes"])
            elif "text" in msg and msg["text"] is not None:
                await _handle_text_message(st, msg["text"])
            else:
                await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session={session_id}, ip={client_ip}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        try:
            if websocket.client_state.value < 3:  # Not closed
                await websocket.close(code=1011)  # Internal Error
        except Exception:
            pass
    finally:
        # Always clean up connection tracking
        track_connection(client_ip, connected=False)


async def _process_audio_chunk(st: SessionState, audio_data: bytes) -> None:
    """Process incoming audio data chunk with security validation."""
    try:
        # Validate audio data size and format
        validate_audio_data(audio_data, st.sr)
        
        # Interpret as float32 little-endian mono PCM
        x = np.frombuffer(audio_data, dtype=np.float32)
        
        # Basic data quality check - float32 PCM should be in [-1.0, 1.0] range
        if len(x) > 0:
            max_amplitude = np.max(np.abs(x))
            if max_amplitude > 2.0:  # Way outside normal range, likely data corruption
                logger.debug(f"Unusual audio amplitude {max_amplitude} in session {st.id} - possible data corruption")
                if st.ws:
                    await st.ws.send_text(json.dumps({
                        "type": "warning", 
                        "message": "Audio data appears corrupted"
                    }))
                return
                
    except Exception as e:
        if st.ws:
            await st.ws.send_text(json.dumps({
                "type": "error", 
                "message": f"Invalid audio data: {str(e)}"
            }))
        return
        
    # Probe billing availability and pause/resume if needed
    await probe_billing_and_update_state(st)
    if st.paused_due_to_billing or st.paused_by_client:
        return
        
    # Process audio through clarity pipeline
    y, st.proc_state = process_stream_chunk(x, st.sr, st.proc_state, params=None)
    st.processed_seconds += float(y.shape[0]) / float(st.sr)
    
    # Send processed audio back as binary
    if st.ws:
        await st.ws.send_bytes(y.tobytes())
        
    # Credit ticking and low-credit signaling
    await tick_credits_and_maybe_signal(st)
    
    # Update ASR buffer
    st._pcm_ring = np.concatenate([st._pcm_ring, x]) if st._pcm_samples else x.copy()
    st._pcm_samples = st._pcm_ring.shape[0]
    
    # Trim ring buffer to max seconds
    max_samples = int(st._asr_max_seconds * st.sr)
    if st._pcm_samples > max_samples:
        st._pcm_ring = st._pcm_ring[-max_samples:]
        st._pcm_samples = st._pcm_ring.shape[0]
        
    # Launch ASR task every stride if not already running
    now = time.time()
    if (not st._asr_running and 
        (now - st._asr_last_run >= st._asr_stride_seconds) and 
        st._pcm_samples >= int(1.0 * st.sr)):
        st._asr_running = True
        st._asr_last_run = now
        asyncio.create_task(_run_asr_and_emit(st))


async def _handle_text_message(st: SessionState, text: str) -> None:
    """Handle text control messages with input validation."""
    # Limit message size to prevent abuse
    if len(text) > 1024:
        if st.ws:
            await st.ws.send_text(json.dumps({
                "type": "error",
                "message": "Text message too long"
            }))
        return
    
    # Simple ping/pong for connection testing
    if text == "ping" and st.ws:
        await st.ws.send_text("pong")
        return
    
    # Try to parse as JSON for control messages
    try:
        msg = json.loads(text)
        if isinstance(msg, dict):
            msg_type = msg.get("type")
            
            # Handle specific control message types
            if msg_type == "heartbeat":
                if st.ws:
                    await st.ws.send_text(json.dumps({
                        "type": "heartbeat_ack",
                        "timestamp": time.time()
                    }))
            elif msg_type == "pause":
                st.paused_by_client = True
            elif msg_type == "resume":
                st.paused_by_client = False
            else:
                # Log unknown message types for security monitoring
                logger.warning(f"Unknown message type '{msg_type}' from session {st.id}")
                
    except json.JSONDecodeError:
        # Non-JSON text messages are ignored
        logger.debug(f"Non-JSON text message in session {st.id}: {text[:100]}")
        pass


async def _run_asr_and_emit(st: SessionState) -> None:
    """Run ASR on current buffer and emit transcription deltas."""
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
        
        segs: List[SubtitleSegment] = []
        
        # Use Azure AI Speech Services for real-time transcription
        import azure.cognitiveservices.speech as speechsdk  # type: ignore
        import tempfile
        
        speech_config = _get_speech_services_config()
        speech_config.output_format = speechsdk.OutputFormat.Detailed
        speech_config.request_word_level_timestamps()
        
        # Create temporary file for Speech Services
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, pcm, st.sr, format="WAV", subtype="PCM_16")
            
            audio_config = speechsdk.audio.AudioConfig(filename=tmp_file.name)
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            
            result = recognizer.recognize_once()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                # Parse detailed result for segments
                if hasattr(result, 'json') and result.json:
                    json_result = json.loads(result.json)
                    if 'NBest' in json_result and json_result['NBest']:
                        words = json_result['NBest'][0].get('Words', [])
                        for i, word in enumerate(words):
                            word_start = word.get('Offset', 0) / 10000000
                            word_end = (word.get('Offset', 0) + word.get('Duration', 0)) / 10000000
                            segs.append(SubtitleSegment(
                                start=word_start,
                                end=word_end,
                                text=word.get('Word', '')
                            ))
                    else:
                        # Fallback: single segment
                        segs.append(SubtitleSegment(
                            start=0.0,
                            end=buf_sec,
                            text=result.text
                        ))
                else:
                    # Fallback: single segment
                    segs.append(SubtitleSegment(
                        start=0.0,
                        end=buf_sec,
                        text=result.text
                    ))
            
            # Clean up temp file
            import os
            os.unlink(tmp_file.name)
            
        if not segs:
            return
            
        # Cleanup segments
        cleaned = _cleanup_segments_with_llm(segs)
        
        # Time offset so segments match session timeline
        offset = max(0.0, st.processed_seconds - buf_sec)
        def _offset_segments(ss: List[SubtitleSegment]) -> List[SubtitleSegment]:
            return [SubtitleSegment(start=s.start + offset, end=s.end + offset, text=s.text) for s in ss]
            
        base = _offset_segments(cleaned)
        
        # Build per-language segments
        by_lang: Dict[str, List[dict]] = {}
        
        # Primary language (detected language unknown; treat as 'orig')
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
                
        # Send delta if any new segments
        any_new = any(len(v) > 0 for v in by_lang.values())
        if any_new and st.ws:
            await st.ws.send_text(json.dumps({
                "type": "delta", 
                "segmentsByLang": by_lang, 
                "isFinal": False
            }))
            
    except Exception as e:
        try:
            if st.ws:
                await st.ws.send_text(json.dumps({
                    "type": "error", 
                    "message": f"asr_failed: {str(e)}"
                }))
        except Exception:
            pass
    finally:
        st._asr_running = False
