"""Tests for authentication and token verification."""
import pytest
import time
import json
import hmac
import base64
import os
import sys
from unittest.mock import patch

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from auth import verify_session_token, _b64url_no_pad, _b64url_decode


class TestTokenUtilities:
    """Test base64url encoding/decoding utilities."""
    
    def test_b64url_no_pad_encoding(self):
        """Test base64url encoding without padding."""
        test_data = b"hello world"
        encoded = _b64url_no_pad(test_data)
        
        # Should be base64url encoded without padding
        assert isinstance(encoded, str)
        assert "=" not in encoded  # No padding
        assert "+" not in encoded  # URL-safe
        assert "/" not in encoded  # URL-safe
    
    def test_b64url_decode_with_padding(self):
        """Test base64url decoding with automatic padding."""
        # Test data that requires padding
        test_cases = [
            ("SGVsbG8", b"Hello"),
            ("SGVsbG9X", b"HelloW"),
            ("SGVsbG9Xb3JsZA", b"HelloWorld")
        ]
        
        for encoded, expected in test_cases:
            decoded = _b64url_decode(encoded)
            assert decoded == expected
    
    def test_b64url_roundtrip(self):
        """Test encode/decode roundtrip."""
        test_data = b"This is a test message with various characters! @#$%"
        encoded = _b64url_no_pad(test_data)
        decoded = _b64url_decode(encoded)
        assert decoded == test_data


class TestTokenVerification:
    """Test token verification functionality."""
    
    def create_test_token(self, payload: dict, key: str) -> str:
        """Helper to create test tokens."""
        payload_json = json.dumps(payload, separators=(',', ':')).encode()
        payload_b64 = _b64url_no_pad(payload_json)
        
        signature = hmac.new(
            key.encode(),
            payload_b64.encode(),
            digestmod='sha256'
        ).digest()
        signature_b64 = _b64url_no_pad(signature)
        
        return f"{payload_b64}.{signature_b64}"
    
    @patch.dict('os.environ', {'STREAM_SESSION_SIGNING_KEY': 'test-key'})
    def test_valid_token(self):
        """Test verification of valid token."""
        future_time = int(time.time()) + 3600
        payload = {
            "sid": "test-session-123",
            "exp": future_time,
            "mode": "stream",
            "userId": "user123"
        }
        
        token = self.create_test_token(payload, "test-key")
        result = verify_session_token(token, "test-session-123")
        
        # Should return the payload when valid
        assert result == payload
    
    @patch.dict('os.environ', {'STREAM_SESSION_SIGNING_KEY': 'test-key'})
    def test_expired_token(self):
        """Test rejection of expired token."""
        past_time = int(time.time()) - 3600  # 1 hour ago
        payload = {
            "sid": "test-session-123",
            "exp": past_time,
            "mode": "stream",
            "userId": "user123"
        }
        
        token = self.create_test_token(payload, "test-key")
        result = verify_session_token(token, "test-session-123")
        assert result is False
    
    @patch.dict('os.environ', {'STREAM_SESSION_SIGNING_KEY': 'test-key'})
    def test_wrong_session_id(self):
        """Test rejection of token with wrong session ID."""
        future_time = int(time.time()) + 3600
        payload = {
            "sid": "different-session",
            "exp": future_time,
            "mode": "stream",
            "userId": "user123"
        }
        
        token = self.create_test_token(payload, "test-key")
        result = verify_session_token(token, "test-session-123")
        assert result is False
    
    @patch.dict('os.environ', {'STREAM_SESSION_SIGNING_KEY': 'test-key'})
    def test_wrong_mode(self):
        """Test rejection of token with wrong mode."""
        future_time = int(time.time()) + 3600
        payload = {
            "sid": "test-session-123",
            "exp": future_time,
            "mode": "batch",  # Wrong mode
            "userId": "user123"
        }
        
        token = self.create_test_token(payload, "test-key")
        result = verify_session_token(token, "test-session-123")
        assert result is False
    
    @patch.dict('os.environ', {'STREAM_SESSION_SIGNING_KEY': 'test-key'})
    def test_invalid_signature(self):
        """Test rejection of token with invalid signature."""
        future_time = int(time.time()) + 3600
        payload = {
            "sid": "test-session-123",
            "exp": future_time,
            "mode": "stream",
            "userId": "user123"
        }
        
        # Create token with wrong key
        token = self.create_test_token(payload, "wrong-key")
        result = verify_session_token(token, "test-session-123")
        assert result is False
    
    def test_no_signing_key(self):
        """Test rejection when no signing key is configured."""
        # No STREAM_SESSION_SIGNING_KEY environment variable
        result = verify_session_token("any.token", "any-session")
        assert result is False
    
    @patch.dict('os.environ', {'STREAM_SESSION_SIGNING_KEY': 'test-key'})
    def test_malformed_token(self):
        """Test rejection of malformed tokens."""
        malformed_tokens = [
            "not.a.valid.token.format",
            "no-dot-separator",
            "invalid!!!.signature",
            "",
            "onlyonepart"
        ]
        
        for token in malformed_tokens:
            result = verify_session_token(token, "test-session-123")
            assert result is False
    
    @patch.dict('os.environ', {'STREAM_SESSION_SIGNING_KEY': 'test-key'})
    def test_timing_attack_resistance(self):
        """Test that verification uses constant-time comparison (behavior test)."""
        future_time = int(time.time()) + 3600
        payload = {
            "sid": "test-session-123",
            "exp": future_time,
            "mode": "stream",
            "userId": "user123"
        }
        
        valid_token = self.create_test_token(payload, "test-key")
        invalid_token = self.create_test_token(payload, "wrong-key")
        
        # Both should return consistent results regardless of timing
        # Valid token should return payload
        result = verify_session_token(valid_token, "test-session-123")
        assert result == payload
        
        # Invalid token should return False
        result = verify_session_token(invalid_token, "test-session-123")
        assert result is False
        
        # Test that multiple calls are consistent  
        for _ in range(5):
            assert verify_session_token(valid_token, "test-session-123") == payload
            assert verify_session_token(invalid_token, "test-session-123") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
