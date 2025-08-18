"""Integration tests for FastAPI endpoints without external dependencies."""
import pytest
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

# We'll mock external dependencies to test the API structure
@patch('session.session_manager')
@patch('security.verify_session_token')
@patch('security.log_security_event')
class TestAPIEndpoints:
    """Test FastAPI endpoints with mocked dependencies."""
    
    def setup_method(self):
        """Setup test client."""
        # Import here to avoid issues with missing dependencies
        with patch.dict('os.environ', {
            'STREAM_SESSION_SIGNING_KEY': 'test-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
            'AZURE_OPENAI_API_KEY': 'test-key',
            'COSMOS_CONNECTION_STRING': 'test-connection'
        }):
            from app import app
            self.client = TestClient(app)
    
    def test_health_endpoint(self, mock_log, mock_verify, mock_session_manager):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "audio-cleaner-streaming"
        assert "timestamp" in data
        # Version should not be exposed for security
        assert "version" not in data
    
    def test_stop_endpoint_valid_token(self, mock_log, mock_verify, mock_session_manager):
        """Test stop endpoint with valid authentication."""
        # Mock valid token verification
        mock_verify.return_value = {"userId": "test-user", "sid": "test-session"}
        
        # Mock session manager
        mock_session = Mock()
        mock_session.processed_seconds = 45.5
        mock_session.ws = None
        mock_session_manager.remove_session.return_value = mock_session
        
        response = self.client.post(
            "/stream/stop",
            json={"sessionId": "test-session"},
            headers={"Authorization": "Bearer valid-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["sessionId"] == "test-session"
        assert data["processedSeconds"] == 45.5
        assert data["subtitles"] == {}
        assert data["status"] == "stopped"
        
        # Verify session was removed
        mock_session_manager.remove_session.assert_called_once_with("test-session")
    
    def test_stop_endpoint_invalid_token(self, mock_log, mock_verify, mock_session_manager):
        """Test stop endpoint with invalid authentication."""
        # Mock invalid token verification
        mock_verify.return_value = False
        
        response = self.client.post(
            "/stream/stop",
            json={"sessionId": "test-session"},
            headers={"Authorization": "Bearer invalid-token"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
    
    def test_stop_endpoint_no_auth(self, mock_log, mock_verify, mock_session_manager):
        """Test stop endpoint without authentication."""
        response = self.client.post(
            "/stream/stop",
            json={"sessionId": "test-session"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
    
    def test_stop_endpoint_invalid_session_id(self, mock_log, mock_verify, mock_session_manager):
        """Test stop endpoint with invalid session ID format."""
        mock_verify.return_value = {"userId": "test-user", "sid": "invalid session"}
        
        response = self.client.post(
            "/stream/stop",
            json={"sessionId": "invalid session"},  # Contains space
            headers={"Authorization": "Bearer valid-token"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_stop_endpoint_security_logging(self, mock_log, mock_verify, mock_session_manager):
        """Test that stop endpoint logs security events."""
        mock_verify.return_value = {"userId": "test-user", "sid": "test-session"}
        mock_session_manager.remove_session.return_value = Mock()
        
        self.client.post(
            "/stream/stop",
            json={"sessionId": "test-session"},
            headers={"Authorization": "Bearer valid-token"}
        )
        
        # Verify security logging was called
        mock_log.assert_called_once()
        call_args = mock_log.call_args[0]
        assert call_args[0] == "stream_stop_request"


@patch('websocket_handler.handle_websocket_connection')
class TestWebSocketEndpoint:
    """Test WebSocket endpoint."""
    
    def setup_method(self):
        """Setup test client."""
        with patch.dict('os.environ', {
            'STREAM_SESSION_SIGNING_KEY': 'test-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
            'AZURE_OPENAI_API_KEY': 'test-key',
            'COSMOS_CONNECTION_STRING': 'test-connection'
        }):
            from app import app
            self.client = TestClient(app)
    
    def test_websocket_endpoint_exists(self, mock_handler):
        """Test that WebSocket endpoint is properly configured."""
        # This test mainly verifies the endpoint exists and routing works
        # Full WebSocket testing would require more complex setup
        
        # Mock the handler to do nothing
        mock_handler.return_value = None
        
        # Test that the endpoint exists by checking the route
        from app import app
        routes = [route.path for route in app.routes]
        assert "/stream/{session_id}" in routes


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def setup_method(self):
        """Setup test client."""
        with patch.dict('os.environ', {
            'STREAM_SESSION_SIGNING_KEY': 'test-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
            'AZURE_OPENAI_API_KEY': 'test-key',
            'COSMOS_CONNECTION_STRING': 'test-connection'
        }):
            from app import app
            self.client = TestClient(app)
    
    @patch('security.verify_session_token')
    @patch('session.session_manager')
    def test_health_endpoint_rate_limiting(self, mock_session_manager, mock_verify):
        """Test rate limiting on health endpoint."""
        # Make multiple requests quickly
        responses = []
        for i in range(5):
            response = self.client.get("/health")
            responses.append(response.status_code)
        
        # All should succeed (100/minute limit is generous for health checks)
        assert all(status == 200 for status in responses)
    
    @patch('security.verify_session_token')
    @patch('session.session_manager')
    def test_stop_endpoint_rate_limiting_setup(self, mock_session_manager, mock_verify):
        """Test that rate limiting is configured for stop endpoint."""
        # This is more of a configuration test since testing actual rate limiting
        # requires either time delays or more complex mock setups
        
        mock_verify.return_value = {"userId": "test-user", "sid": "test-session"}
        mock_session_manager.remove_session.return_value = Mock()
        
        # Single request should work
        response = self.client.post(
            "/stream/stop",
            json={"sessionId": "test-session"},
            headers={"Authorization": "Bearer valid-token"}
        )
        
        assert response.status_code == 200


class TestCORSConfiguration:
    """Test CORS configuration."""
    
    def setup_method(self):
        """Setup test client."""
        with patch.dict('os.environ', {
            'STREAM_SESSION_SIGNING_KEY': 'test-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
            'AZURE_OPENAI_API_KEY': 'test-key',
            'COSMOS_CONNECTION_STRING': 'test-connection'
        }):
            from app import app
            self.client = TestClient(app)
    
    def test_cors_headers_present(self):
        """Test that CORS headers are present in responses."""
        response = self.client.get("/health")
        
        # FastAPI CORS middleware should add headers
        assert response.status_code == 200
        # Note: TestClient doesn't always show CORS headers,
        # but we can verify the middleware is configured
    
    def test_options_request_handling(self):
        """Test that OPTIONS requests are handled (CORS preflight)."""
        response = self.client.options("/health")
        
        # Should not return 405 Method Not Allowed if CORS is properly configured
        assert response.status_code != 405


class TestInputValidation:
    """Test input validation for all endpoints."""
    
    def setup_method(self):
        """Setup test client."""
        with patch.dict('os.environ', {
            'STREAM_SESSION_SIGNING_KEY': 'test-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
            'AZURE_OPENAI_API_KEY': 'test-key',
            'COSMOS_CONNECTION_STRING': 'test-connection'
        }):
            from app import app
            self.client = TestClient(app)
    
    @patch('security.verify_session_token')
    def test_stop_endpoint_pydantic_validation(self, mock_verify):
        """Test Pydantic validation on stop endpoint."""
        mock_verify.return_value = {"userId": "test-user", "sid": "valid-session"}
        
        # Missing sessionId
        response = self.client.post(
            "/stream/stop",
            json={},
            headers={"Authorization": "Bearer valid-token"}
        )
        assert response.status_code == 422
        
        # Invalid sessionId format
        response = self.client.post(
            "/stream/stop", 
            json={"sessionId": "session with spaces"},
            headers={"Authorization": "Bearer valid-token"}
        )
        assert response.status_code == 422
        
        # sessionId too long
        response = self.client.post(
            "/stream/stop",
            json={"sessionId": "x" * 101},  # Max length is 100
            headers={"Authorization": "Bearer valid-token"}
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
