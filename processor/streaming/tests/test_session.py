"""Tests for session state management."""
import pytest
import numpy as np
import os
import sys
from datetime import datetime, timezone
from unittest.mock import Mock

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from session import SessionState, SessionManager


class TestSessionState:
    """Test SessionState class functionality."""
    
    def test_session_initialization(self):
        """Test that sessions initialize with correct defaults."""
        session = SessionState("test-session-123")
        
        assert session.id == "test-session-123"
        assert session.ws is None
        assert session.sr == 16000
        assert session.langs == ["en"]
        assert session.user_id is None
        assert session.processed_seconds == 0.0
        assert session.last_deducted_seconds == 0.0
        assert session.credits_cents_spent == 0
        assert session.token is None
        assert not session.paused_due_to_billing
        assert not session.paused_by_client
        
        # ASR state
        assert isinstance(session._pcm_ring, np.ndarray)
        assert session._pcm_ring.dtype == np.float32
        assert session._pcm_samples == 0
        assert not session._asr_running
        assert session._asr_last_run == 0.0
        assert isinstance(session._asr_last_end_by_lang, dict)
        assert len(session._asr_last_end_by_lang) == 0
    
    def test_session_configuration_from_env(self):
        """Test that session picks up configuration from environment."""
        session = SessionState("test-session")
        
        # These should match values from test __init__.py
        assert session._asr_max_seconds == 6.0
        assert session._asr_stride_seconds == 2.0
        assert session._base_cents_per_min == 10.0
        assert session._extra_lang_cents_per_min == 5.0
        assert session._low_credits_grace_sec == 8.0
    
    def test_session_audio_buffer_management(self):
        """Test PCM audio buffer management."""
        session = SessionState("test-session")
        
        # Add some audio data
        audio_chunk = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        session._pcm_ring = audio_chunk.copy()
        session._pcm_samples = len(audio_chunk)
        
        assert session._pcm_samples == 4
        assert np.array_equal(session._pcm_ring, audio_chunk)
        
        # Add more audio
        more_audio = np.array([0.5, 0.6], dtype=np.float32)
        session._pcm_ring = np.concatenate([session._pcm_ring, more_audio])
        session._pcm_samples = len(session._pcm_ring)
        
        assert session._pcm_samples == 6
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        assert np.array_equal(session._pcm_ring, expected)
    
    def test_session_language_tracking(self):
        """Test language-specific ASR tracking."""
        session = SessionState("test-session")
        session.langs = ["en", "es", "fr"]
        
        # Track different end times for different languages
        session._asr_last_end_by_lang["en"] = 10.5
        session._asr_last_end_by_lang["es"] = 8.2
        session._asr_last_end_by_lang["fr"] = 12.1
        
        assert session._asr_last_end_by_lang["en"] == 10.5
        assert session._asr_last_end_by_lang["es"] == 8.2
        assert session._asr_last_end_by_lang["fr"] == 12.1
    
    def test_session_billing_state(self):
        """Test billing-related state management."""
        session = SessionState("test-session")
        
        # Test initial billing state
        assert not session._low_sent
        assert not session._stop_sent
        assert not session._pause_notified
        assert not session._resume_notified
        
        # Test billing pause/resume logic
        session.paused_due_to_billing = True
        session._pause_notified = True
        assert session.paused_due_to_billing
        assert session._pause_notified
        
        session.paused_due_to_billing = False
        session._resume_notified = True
        session._pause_notified = False
        assert not session.paused_due_to_billing
        assert session._resume_notified
    
    def test_session_client_pause_control(self):
        """Test client-side pause functionality."""
        session = SessionState("test-session")
        
        # Initially not paused
        assert not session.paused_by_client
        
        # Client can pause
        session.paused_by_client = True
        assert session.paused_by_client
        
        # Client can resume
        session.paused_by_client = False
        assert not session.paused_by_client


class TestSessionManager:
    """Test SessionManager functionality."""
    
    def test_session_manager_initialization(self):
        """Test that session manager initializes correctly."""
        manager = SessionManager()
        assert isinstance(manager._sessions, dict)
        assert len(manager._sessions) == 0
    
    def test_create_new_session(self):
        """Test creating new sessions."""
        manager = SessionManager()
        
        session = manager.create_session("new-session-123")
        assert isinstance(session, SessionState)
        assert session.id == "new-session-123"
        assert len(manager._sessions) == 1
        assert "new-session-123" in manager._sessions
    
    def test_get_existing_session(self):
        """Test retrieving existing sessions."""
        manager = SessionManager()
        
        # Create a session
        original = manager.create_session("existing-session")
        
        # Get the same session
        retrieved = manager.get_session("existing-session")
        assert retrieved is original
        assert retrieved.id == "existing-session"
    
    def test_get_nonexistent_session(self):
        """Test retrieving non-existent sessions."""
        manager = SessionManager()
        
        result = manager.get_session("does-not-exist")
        assert result is None
    
    def test_create_session_idempotent(self):
        """Test that creating existing session returns the same instance."""
        manager = SessionManager()
        
        # Create session twice
        session1 = manager.create_session("same-session-id")
        session2 = manager.create_session("same-session-id")
        
        # Should be the same instance
        assert session1 is session2
        assert len(manager._sessions) == 1
    
    def test_remove_session(self):
        """Test removing sessions."""
        manager = SessionManager()
        
        # Create and remove session
        session = manager.create_session("to-remove")
        removed = manager.remove_session("to-remove")
        
        assert removed is session
        assert len(manager._sessions) == 0
        assert "to-remove" not in manager._sessions
    
    def test_remove_nonexistent_session(self):
        """Test removing non-existent sessions."""
        manager = SessionManager()
        
        result = manager.remove_session("does-not-exist")
        assert result is None
    
    def test_list_sessions(self):
        """Test listing all session IDs."""
        manager = SessionManager()
        
        # Initially empty
        sessions = manager.list_sessions()
        assert sessions == []
        
        # Add sessions
        manager.create_session("session-1")
        manager.create_session("session-2")
        manager.create_session("session-3")
        
        sessions = manager.list_sessions()
        assert len(sessions) == 3
        assert "session-1" in sessions
        assert "session-2" in sessions
        assert "session-3" in sessions
    
    def test_session_manager_multiple_operations(self):
        """Test complex session management operations."""
        manager = SessionManager()
        
        # Create multiple sessions
        s1 = manager.create_session("session-1")
        s2 = manager.create_session("session-2")
        s3 = manager.create_session("session-3")
        
        # Verify they're all different
        assert s1 is not s2
        assert s2 is not s3
        assert s1 is not s3
        
        # Modify session state
        s1.processed_seconds = 10.5
        s2.langs = ["en", "es"]
        s3.user_id = "user123"
        
        # Verify states are independent
        assert manager.get_session("session-1").processed_seconds == 10.5
        assert manager.get_session("session-2").langs == ["en", "es"]
        assert manager.get_session("session-3").user_id == "user123"
        
        # Remove middle session
        removed = manager.remove_session("session-2")
        assert removed is s2
        assert len(manager.list_sessions()) == 2
        assert "session-2" not in manager.list_sessions()
        
        # Other sessions should still exist
        assert manager.get_session("session-1") is s1
        assert manager.get_session("session-3") is s3


class TestSessionWebSocketIntegration:
    """Test session integration with WebSocket mocks."""
    
    def test_session_websocket_assignment(self):
        """Test assigning WebSocket to session."""
        session = SessionState("ws-session")
        mock_ws = Mock()
        
        session.ws = mock_ws
        assert session.ws is mock_ws
    
    def test_session_token_assignment(self):
        """Test assigning token to session."""
        session = SessionState("token-session")
        test_token = "test.token.here"
        
        session.token = test_token
        assert session.token == test_token
    
    def test_session_user_id_assignment(self):
        """Test assigning user ID to session."""
        session = SessionState("user-session")
        
        session.user_id = "user12345"
        assert session.user_id == "user12345"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
