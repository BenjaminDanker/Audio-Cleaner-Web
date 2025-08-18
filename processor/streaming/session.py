"""Session state management for streaming audio processing."""
from __future__ import annotations

import os
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional, List
from fastapi import WebSocket

from ai.audio_clarity_pipeline import StreamState  # type: ignore


class SessionState:
    """Manages state for a single streaming session."""
    
    def __init__(self, session_id: str):
        self.id = session_id
        self.ws: Optional[WebSocket] = None
        self.sr = 16000
        self.langs: List[str] = ["en"]
        self.user_id: Optional[str] = None
        self.proc_state: Optional[StreamState] = None
        self.processed_seconds = 0.0
        self.last_deducted_seconds = 0.0
        self.credits_cents_spent = 0
        self._buf = bytearray()
        self.token: Optional[str] = None
        
        # ASR state
        self._pcm_ring = np.zeros(1, dtype=np.float32)
        self._pcm_samples = 0
        self._asr_running = False
        self._asr_last_run = 0.0
        self._asr_last_end_by_lang: Dict[str, float] = {}
        self._asr_max_seconds = float(os.getenv("STREAM_ASR_BUFFER_SECONDS", "6"))
        self._asr_stride_seconds = float(os.getenv("STREAM_ASR_STRIDE_SECONDS", "2"))
        
        # Credit policy
        self._base_cents_per_min = float(os.getenv("STREAM_BASE_CENTS_PER_MINUTE", "10"))
        self._extra_lang_cents_per_min = float(os.getenv("STREAM_EXTRA_LANG_CENTS_PER_MINUTE", "5"))
        self._low_credits_grace_sec = float(os.getenv("STREAM_LOW_CREDITS_GRACE_SECONDS", "8"))
        self._low_sent = False
        self._stop_sent = False
        
        # Transcript accumulation (disabled by default)
        accumulate = bool(int(os.getenv("STREAM_ACCUMULATE_CAPTIONS", "0")))
        self._segments_by_lang = {} if accumulate else None
        self._started_at = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        
        # Billing availability control
        self.paused_due_to_billing = False
        self._pause_notified = False
        self._resume_notified = False
        self._billing_last_probe = 0.0
        
        # Client-side pause control
        self.paused_by_client = False


class SessionManager:
    """Manages active streaming sessions."""
    
    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get existing session by ID."""
        return self._sessions.get(session_id)
    
    def create_session(self, session_id: str) -> SessionState:
        """Create new session or return existing one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id)
        return self._sessions[session_id]
    
    def remove_session(self, session_id: str) -> Optional[SessionState]:
        """Remove and return session if it exists."""
        return self._sessions.pop(session_id, None)
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())


# Global session manager instance
session_manager = SessionManager()
