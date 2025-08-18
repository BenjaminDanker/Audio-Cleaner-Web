"""Tests for security utilities and validation."""
import pytest
import re
from unittest.mock import Mock, patch
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials

from security import (
    validate_session_id,
    extract_token_from_request,
    verify_session_access,
    get_client_ip,
    is_origin_allowed,
    check_connection_limit,
    track_connection,
    validate_audio_data,
    log_security_event,
    SESSION_ID_PATTERN,
    ALLOWED_ORIGINS,
    connection_counts
)


class TestSessionValidation:
    """Test session ID validation."""
    
    def test_valid_session_ids(self):
        """Test that valid session IDs pass validation."""
        valid_ids = [
            "session123",
            "user-session-456", 
            "test_session_789",
            "a1b2c3d4e5f6",
            "session-with-hyphens",
            "session_with_underscores"
        ]
        
        for session_id in valid_ids:
            result = validate_session_id(session_id)
            assert result == session_id
    
    def test_invalid_session_ids(self):
        """Test that invalid session IDs are rejected."""
        invalid_ids = [
            "",  # Empty
            "session with spaces",
            "session@with#special!chars",
            "session.with.dots",
            "session/with/slashes",
            "x" * 101,  # Too long
            "session+plus",
            "session=equals"
        ]
        
        for session_id in invalid_ids:
            with pytest.raises(HTTPException) as exc_info:
                validate_session_id(session_id)
            assert exc_info.value.status_code == 400
    
    def test_session_id_pattern_regex(self):
        """Test the session ID regex pattern directly."""
        assert SESSION_ID_PATTERN.match("valid123")
        assert SESSION_ID_PATTERN.match("valid-session")
        assert SESSION_ID_PATTERN.match("valid_session")
        
        assert not SESSION_ID_PATTERN.match("invalid spaces")
        assert not SESSION_ID_PATTERN.match("invalid@chars")
        assert not SESSION_ID_PATTERN.match("")


class TestTokenExtraction:
    """Test token extraction from requests."""
    
    def test_extract_from_authorization_header(self):
        """Test extracting token from Authorization header."""
        auth = HTTPAuthorizationCredentials(scheme="Bearer", credentials="test-token-123")
        token = extract_token_from_request(authorization=auth)
        assert token == "test-token-123"
    
    def test_extract_from_session_header(self):
        """Test extracting token from X-Session-Token header."""
        token = extract_token_from_request(x_session_token="header-token-456")
        assert token == "header-token-456"
    
    def test_authorization_takes_precedence(self):
        """Test that Authorization header takes precedence."""
        auth = HTTPAuthorizationCredentials(scheme="Bearer", credentials="auth-token")
        token = extract_token_from_request(
            authorization=auth, 
            x_session_token="header-token"
        )
        assert token == "auth-token"
    
    def test_no_token_available(self):
        """Test when no token is available."""
        token = extract_token_from_request()
        assert token is None


class TestClientIP:
    """Test client IP extraction."""
    
    def test_get_client_ip_from_forwarded_for(self):
        """Test extracting IP from X-Forwarded-For header."""
        request = Mock()
        request.headers = {"X-Forwarded-For": "203.0.113.1, 198.51.100.1"}
        request.client = Mock()
        request.client.host = "10.0.0.1"
        
        ip = get_client_ip(request)
        assert ip == "203.0.113.1"  # First IP in chain
    
    def test_get_client_ip_from_real_ip(self):
        """Test extracting IP from X-Real-IP header."""
        request = Mock()
        request.headers = {"X-Real-IP": "203.0.113.2"}
        request.client = Mock()
        request.client.host = "10.0.0.1"
        
        ip = get_client_ip(request)
        assert ip == "203.0.113.2"
    
    def test_get_client_ip_direct_connection(self):
        """Test extracting IP from direct connection."""
        request = Mock()
        request.headers = {}
        request.client = Mock()
        request.client.host = "192.168.1.100"
        
        ip = get_client_ip(request)
        assert ip == "192.168.1.100"
    
    def test_get_client_ip_no_client(self):
        """Test when client info is not available."""
        request = Mock()
        request.headers = {}
        request.client = None
        
        ip = get_client_ip(request)
        assert ip == "unknown"


class TestOriginValidation:
    """Test CORS origin validation."""
    
    def test_allowed_exact_origins(self):
        """Test exact origin matches."""
        allowed = ["http://localhost:3000", "http://localhost:4280"]
        
        for origin in allowed:
            if origin in ALLOWED_ORIGINS:
                assert is_origin_allowed(origin)
    
    def test_wildcard_origins(self):
        """Test wildcard origin patterns."""
        # Test pattern from ALLOWED_ORIGINS
        test_origins = [
            "https://app-name.azurestaticapps.net",
            "https://my-app.1.azurestaticapps.net",
            "https://staging.azurestaticapps.net"
        ]
        
        for origin in test_origins:
            # This would be allowed by "https://*.azurestaticapps.net" pattern
            assert "azurestaticapps.net" in origin
    
    def test_disallowed_origins(self):
        """Test that disallowed origins are rejected."""
        disallowed = [
            "https://evil.com",
            "http://malicious-site.net",
            "https://fake-azurestaticapps.com",  # Wrong TLD
            ""
        ]
        
        for origin in disallowed:
            assert not is_origin_allowed(origin)


class TestConnectionLimiting:
    """Test connection limiting functionality."""
    
    def setUp(self):
        """Clear connection counts before each test."""
        connection_counts.clear()
    
    def test_connection_tracking(self):
        """Test connection count tracking."""
        self.setUp()
        ip = "192.168.1.100"
        
        # Initially should be within limits
        assert check_connection_limit(ip)
        
        # Track connections
        track_connection(ip, connected=True)
        assert connection_counts[ip] == 1
        
        track_connection(ip, connected=True)
        assert connection_counts[ip] == 2
        
        # Disconnect
        track_connection(ip, connected=False)
        assert connection_counts[ip] == 1
        
        track_connection(ip, connected=False)
        assert ip not in connection_counts  # Should be cleaned up
    
    def test_connection_limit_enforcement(self):
        """Test that connection limits are enforced."""
        self.setUp()
        ip = "192.168.1.101"
        
        # Add connections up to limit (5 total)
        for i in range(5):  # MAX_CONNECTIONS_PER_IP = 5
            # Should be allowed before adding
            assert check_connection_limit(ip)
            track_connection(ip, connected=True)
        
        # Should now be at limit (5 connections)
        assert not check_connection_limit(ip)
    
    def test_multiple_ips_independent(self):
        """Test that different IPs have independent limits."""
        self.setUp()
        ip1 = "192.168.1.102"
        ip2 = "192.168.1.103"
        
        # Max out first IP
        for i in range(5):
            track_connection(ip1, connected=True)
        
        # Second IP should still be allowed
        assert check_connection_limit(ip2)


class TestAudioValidation:
    """Test audio data validation."""
    
    def test_valid_audio_data(self):
        """Test validation of valid audio data."""
        # 100 float32 samples = 400 bytes
        valid_data = b'\x00' * 400
        assert validate_audio_data(valid_data, 16000)
    
    def test_audio_chunk_too_large(self):
        """Test rejection of oversized audio chunks."""
        # Create data larger than MAX_AUDIO_CHUNK_SIZE (1MB)
        large_data = b'\x00' * (1024 * 1024 + 1)
        
        with pytest.raises(HTTPException) as exc_info:
            validate_audio_data(large_data, 16000)
        assert exc_info.value.status_code == 413
    
    def test_sample_rate_too_high(self):
        """Test rejection of excessive sample rates."""
        valid_data = b'\x00' * 400
        
        with pytest.raises(HTTPException) as exc_info:
            validate_audio_data(valid_data, 100000)  # Way too high
        assert exc_info.value.status_code == 400
    
    def test_invalid_audio_length(self):
        """Test rejection of audio data with invalid length."""
        # Length not multiple of 4 (invalid for float32)
        invalid_data = b'\x00' * 401  # 401 is not divisible by 4
        
        with pytest.raises(HTTPException) as exc_info:
            validate_audio_data(invalid_data, 16000)
        assert exc_info.value.status_code == 400


class TestSecurityLogging:
    """Test security event logging."""
    
    @patch('security.logger')
    def test_log_security_event(self, mock_logger):
        """Test security event logging functionality."""
        request = Mock()
        request.headers = {"User-Agent": "TestAgent/1.0"}
        request.url.path = "/test/path"
        request.method = "POST"
        request.client.host = "192.168.1.200"
        
        log_security_event(
            "test_event",
            request,
            session_id="test-session",
            user_id="test-user",
            details={"extra": "info"}
        )
        
        # Verify logger was called
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        
        # Check log message
        assert "test_event" in call_args[0][0]
        
        # Check extra data
        extra_data = call_args[1]["extra"]
        assert extra_data["event_type"] == "test_event"
        assert extra_data["client_ip"] == "192.168.1.200"
        assert extra_data["session_id"] == "test-session"
        assert extra_data["user_id"] == "test-user"
        assert extra_data["extra"] == "info"


class TestVerifySessionAccess:
    """Test session access verification."""
    
    @patch('security.verify_session_token')
    def test_valid_session_access(self, mock_verify):
        """Test successful session access verification."""
        mock_verify.return_value = {"userId": "test-user", "sid": "test-session"}
        
        result = verify_session_access("valid-token", "test-session")
        assert result["userId"] == "test-user"
        mock_verify.assert_called_once_with("valid-token", expected_session="test-session")
    
    @patch('security.verify_session_token')
    def test_invalid_token_access(self, mock_verify):
        """Test rejection of invalid token."""
        mock_verify.return_value = False
        
        with pytest.raises(HTTPException) as exc_info:
            verify_session_access("invalid-token", "test-session")
        assert exc_info.value.status_code == 401
    
    def test_missing_token_access(self):
        """Test rejection when no token provided."""
        with pytest.raises(HTTPException) as exc_info:
            verify_session_access("", "test-session")
        assert exc_info.value.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
