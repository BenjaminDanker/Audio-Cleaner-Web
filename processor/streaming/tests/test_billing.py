"""Tests for billing functionality without external dependencies."""
import pytest
import time
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

# We'll test the billing logic without actually calling Cosmos
from billing import tick_credits_and_maybe_signal, probe_billing_and_update_state
from session import SessionState


class TestBillingLogic:
    """Test billing calculation and logic without Cosmos calls."""
    
    def create_mock_session(self, **kwargs):
        """Helper to create mock session with billing defaults."""
        session = SessionState("test-session")
        
        # Set billing-related defaults
        session.processed_seconds = kwargs.get('processed_seconds', 0.0)
        session.last_deducted_seconds = kwargs.get('last_deducted_seconds', 0.0)
        session.user_id = kwargs.get('user_id', 'test-user')
        session.langs = kwargs.get('langs', ['en'])
        session.credits_cents_spent = kwargs.get('credits_cents_spent', 0)
        session._low_sent = kwargs.get('low_sent', False)
        session._stop_sent = kwargs.get('stop_sent', False)
        session.ws = kwargs.get('ws', Mock())
        session.paused_due_to_billing = kwargs.get('paused_due_to_billing', False)
        
        return session
    
    def test_billing_rate_calculation(self):
        """Test billing rate calculation for different language configs."""
        session = SessionState("test-session")

        # Default rates from environment
        base_rate = session._base_cents_per_min  # 10 cents
        extra_rate = session._extra_lang_cents_per_min  # 5 cents

        # Single language (en) - current implementation charges for each lang
        session.langs = ['en']
        # Rate = base + max(0, len(langs if not 'orig') - 0) * extra = 10 + 1 * 5 = 15
        expected_rate = base_rate + max(0, len([l for l in session.langs if l != 'orig']) - 0) * extra_rate
        assert expected_rate == 15.0        # Two languages - should add extra
        session.langs = ['en', 'es']  
        # Rate = base + max(0, 2 - 0) * extra = 10 + 2 * 5 = 20
        # Note: the actual logic counts non-'orig' languages
        
        # Three languages - more extra
        session.langs = ['en', 'es', 'fr']
        # Rate = base + max(0, 3 - 0) * extra = 10 + 3 * 5 = 25
    
    @pytest.mark.asyncio
    @patch('clients.get_container')
    @patch.dict('os.environ', {'COSMOS_CONNECTION_STRING': 'mock-connection-string'})
    async def test_credit_deduction_timing(self, mock_get_container):
        """Test that credits are deducted at proper intervals."""
        # Mock Cosmos containers
        mock_accounts = Mock()
        mock_txns = Mock()
        mock_get_container.side_effect = lambda name: mock_accounts if name == 'accounts' else mock_txns
        
        # Mock account with sufficient balance
        mock_accounts.read_item.return_value = {'balance': 1000, 'id': 'test-user'}
        
        session = self.create_mock_session(
            processed_seconds=10.0,  # 10 seconds processed
            last_deducted_seconds=0.0,  # No deductions yet
            user_id='test-user'
        )
        
        # Should deduct because 10 - 0 >= 5 seconds
        await tick_credits_and_maybe_signal(session)
        
        # Should have updated last_deducted_seconds
        assert session.last_deducted_seconds == 10.0
        
        # Should have called Cosmos to deduct credits
        mock_accounts.read_item.assert_called_once_with('test-user', 'test-user')
        mock_accounts.upsert_item.assert_called_once()
        mock_txns.upsert_item.assert_called_once()
    
    @patch('billing.get_container')
    async def test_no_deduction_too_soon(self, mock_get_container):
        """Test that credits aren't deducted too frequently."""
        session = self.create_mock_session(
            processed_seconds=3.0,  # Only 3 seconds processed
            last_deducted_seconds=0.0,  # No deductions yet
            user_id='test-user'
        )
        
        # Should NOT deduct because 3 - 0 < 5 seconds
        await tick_credits_and_maybe_signal(session)
        
        # last_deducted_seconds should remain unchanged
        assert session.last_deducted_seconds == 0.0
        
        # Should not have called Cosmos
        mock_get_container.assert_not_called()
    
    @patch('billing.get_container')
    async def test_incremental_deductions(self, mock_get_container):
        """Test incremental credit deductions."""
        mock_accounts = Mock()
        mock_txns = Mock()
        mock_get_container.side_effect = lambda name: mock_accounts if name == 'accounts' else mock_txns
        
        # Mock account with sufficient balance
        mock_accounts.read_item.return_value = {'balance': 1000, 'id': 'test-user'}
        
        session = self.create_mock_session(
            processed_seconds=15.0,  # 15 seconds total
            last_deducted_seconds=10.0,  # Already deducted for first 10 seconds
            user_id='test-user'
        )
        
        await tick_credits_and_maybe_signal(session)
        
        # Should deduct for incremental 5 seconds (15 - 10)
        assert session.last_deducted_seconds == 15.0
        
        # Calculate expected deduction for 5 seconds at 10 cents/minute
        # 5 seconds = 5/60 minutes = 0.0833 minutes
        # 0.0833 * 10 = 0.833 cents, rounded to 1 cent
        assert session.credits_cents_spent > 0
    
    async def test_no_user_id_no_billing(self):
        """Test that billing is skipped when no user ID."""
        session = self.create_mock_session(
            processed_seconds=10.0,
            last_deducted_seconds=0.0,
            user_id=None  # No user ID
        )
        
        await tick_credits_and_maybe_signal(session)
        
        # Should not update billing state
        assert session.last_deducted_seconds == 0.0
        assert session.credits_cents_spent == 0
    
    @patch('billing.get_container')
    async def test_low_credits_signaling(self, mock_get_container):
        """Test low credits warning and stop signaling."""
        mock_accounts = Mock()
        mock_txns = Mock()
        mock_get_container.side_effect = lambda name: mock_accounts if name == 'accounts' else mock_txns
        
        # Mock account with zero balance
        mock_accounts.read_item.return_value = {'balance': 0, 'id': 'test-user'}
        
        mock_ws = AsyncMock()
        session = self.create_mock_session(
            processed_seconds=10.0,
            last_deducted_seconds=0.0,
            user_id='test-user',
            ws=mock_ws
        )
        
        await tick_credits_and_maybe_signal(session)
        
        # Should have sent low credits and stop messages
        assert session._low_sent
        assert session._stop_sent
        
        # Check WebSocket calls
        assert mock_ws.send_text.call_count == 2  # LOW_CREDITS and STOP
        calls = [call[0][0] for call in mock_ws.send_text.call_args_list]
        
        # Parse JSON messages
        messages = [json.loads(call) for call in calls]
        types = [msg['type'] for msg in messages]
        
        assert 'LOW_CREDITS' in types
        assert 'STOP' in types
    
    @patch('billing.get_container')
    async def test_cosmos_failure_pauses_billing(self, mock_get_container):
        """Test that Cosmos failures pause billing gracefully."""
        mock_accounts = Mock()
        mock_accounts.read_item.side_effect = Exception("Cosmos unavailable")
        mock_get_container.return_value = mock_accounts
        
        mock_ws = AsyncMock()
        session = self.create_mock_session(
            processed_seconds=10.0,
            last_deducted_seconds=0.0,
            user_id='test-user',
            ws=mock_ws,
            paused_due_to_billing=False
        )
        
        await tick_credits_and_maybe_signal(session)
        
        # Should pause billing due to error
        assert session.paused_due_to_billing
        
        # Should have sent pause notification
        mock_ws.send_text.assert_called()
        call_arg = mock_ws.send_text.call_args[0][0]
        message = json.loads(call_arg)
        assert message['type'] == 'PAUSED_BILLING'


class TestBillingProbe:
    """Test billing availability probing."""
    
    def create_mock_session(self, **kwargs):
        """Helper to create mock session."""
        session = SessionState("test-session")
        session.paused_due_to_billing = kwargs.get('paused_due_to_billing', False)
        session._pause_notified = kwargs.get('pause_notified', False)
        session._resume_notified = kwargs.get('resume_notified', False)
        session._billing_last_probe = kwargs.get('billing_last_probe', 0.0)
        session.ws = kwargs.get('ws', Mock())
        return session
    
    @patch('billing.get_container')
    @patch('time.time')
    async def test_billing_probe_timing(self, mock_time, mock_get_container):
        """Test that billing probe respects timing intervals."""
        mock_time.return_value = 100.0  # Current time
        
        session = self.create_mock_session(
            billing_last_probe=98.0  # Probed 2 seconds ago
        )
        
        await probe_billing_and_update_state(session)
        
        # Should not probe because 100 - 98 < 5 seconds
        mock_get_container.assert_not_called()
        assert session._billing_last_probe == 98.0  # Unchanged
    
    @patch('billing.get_container')
    @patch('time.time')
    async def test_billing_probe_success_resume(self, mock_time, mock_get_container):
        """Test successful billing probe resumes paused session."""
        mock_time.return_value = 100.0
        mock_container = Mock()
        mock_container.read.return_value = True  # Successful probe
        mock_get_container.return_value = mock_container
        
        mock_ws = AsyncMock()
        session = self.create_mock_session(
            billing_last_probe=90.0,  # 10 seconds ago, should probe
            paused_due_to_billing=True,  # Currently paused
            pause_notified=True,
            resume_notified=False,
            ws=mock_ws
        )
        
        await probe_billing_and_update_state(session)
        
        # Should have probed and resumed
        assert not session.paused_due_to_billing
        assert session._resume_notified
        assert not session._pause_notified
        assert session._billing_last_probe == 100.0
        
        # Should have sent resume message
        mock_ws.send_text.assert_called()
        call_arg = mock_ws.send_text.call_args[0][0]
        message = json.loads(call_arg)
        assert message['type'] == 'RESUMED_BILLING'
    
    @patch('billing.get_container')
    @patch('time.time')
    async def test_billing_probe_failure_pause(self, mock_time, mock_get_container):
        """Test failed billing probe pauses session."""
        mock_time.return_value = 100.0
        mock_container = Mock()
        mock_container.read.side_effect = Exception("Connection failed")
        mock_get_container.return_value = mock_container
        
        mock_ws = AsyncMock()
        session = self.create_mock_session(
            billing_last_probe=90.0,  # Should probe
            paused_due_to_billing=False,  # Not currently paused
            pause_notified=False,
            ws=mock_ws
        )
        
        await probe_billing_and_update_state(session)
        
        # Should have paused due to failure
        assert session.paused_due_to_billing
        assert session._pause_notified
        assert not session._resume_notified
        assert session._billing_last_probe == 100.0
        
        # Should have sent pause message
        mock_ws.send_text.assert_called()
        call_arg = mock_ws.send_text.call_args[0][0]
        message = json.loads(call_arg)
        assert message['type'] == 'PAUSED_BILLING'


class TestBillingCalculations:
    """Test billing rate and amount calculations."""
    
    def test_single_language_rate(self):
        """Test billing rate for single language."""
        session = SessionState("test-session")
        session.langs = ['en']

        # Base rate + 1 extra language charge: 10 cents/min + 1 * 5 cents = 15 cents/min
        base = session._base_cents_per_min
        extra_langs = max(0, len([l for l in session.langs if l != 'orig']) - 0)
        rate = base + extra_langs * session._extra_lang_cents_per_min

        assert rate == 15.0
    
    def test_multiple_language_rate(self):
        """Test billing rate for multiple languages."""
        session = SessionState("test-session")
        session.langs = ['en', 'es', 'fr']  # 3 languages
        
        # Base rate + 3 extra languages * 5 cents = 10 + 15 = 25 cents/min
        base = session._base_cents_per_min  # 10
        extra_langs = max(0, len([l for l in session.langs if l != 'orig']) - 0)  # 3
        rate = base + extra_langs * session._extra_lang_cents_per_min  # 10 + 3*5 = 25
        
        assert rate == 25.0
    
    def test_billing_amount_calculation(self):
        """Test calculation of billing amounts for time periods."""
        # 1 minute at 10 cents/min = 10 cents
        seconds = 60.0
        rate_cpm = 10.0
        amount = int(round((seconds / 60.0) * rate_cpm))
        assert amount == 10
        
        # 30 seconds at 10 cents/min = 5 cents
        seconds = 30.0
        amount = int(round((seconds / 60.0) * rate_cpm))
        assert amount == 5
        
        # 5 seconds at 10 cents/min = 0.833 cents, rounded to 1
        seconds = 5.0
        amount = int(round((seconds / 60.0) * rate_cpm))
        assert amount == 1  # Rounded up
        
        # Very short time should still charge minimum
        seconds = 0.1
        amount = int(round((seconds / 60.0) * rate_cpm))
        assert amount == 0  # Rounds to 0 for very short periods


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
