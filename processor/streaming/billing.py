"""Billing and credit management for streaming sessions."""
from __future__ import annotations

import os
import time
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from session import SessionState

# Import shared client at runtime to avoid circular imports
def _get_cosmos_containers():
    """Get Cosmos containers for billing operations."""
    import sys
    from pathlib import Path
    
    # Add shared directory to path if not already there
    shared_dir = Path(__file__).parent.parent / "shared"
    if str(shared_dir) not in sys.path:
        sys.path.append(str(shared_dir))
    
    from clients import get_container
    return get_container('accounts'), get_container('transactions')


async def tick_credits_and_maybe_signal(st: 'SessionState') -> None:
    """Deduct credits for processed audio and signal low/zero balance."""
    try:
        # Calculate effective rate per minute
        add_langs = max(0, len([l for l in st.langs if l != 'orig']) - 0)
        rate_cpm = st._base_cents_per_min + (add_langs * st._extra_lang_cents_per_min)
        
        # Deduct every ~5s of processed audio to reduce write pressure
        if st.processed_seconds - st.last_deducted_seconds >= 5.0:
            inc_sec = st.processed_seconds - st.last_deducted_seconds
            inc_cents = int(round((inc_sec / 60.0) * rate_cpm))
            st.last_deducted_seconds = st.processed_seconds
            
            if inc_cents > 0 and st.user_id:
                conn = os.getenv("COSMOS_CONNECTION_STRING")
                if conn:
                    accounts, txns = _get_cosmos_containers()
                    try:
                        # Update account balance
                        acc = accounts.read_item(st.user_id, st.user_id)
                        bal = int(acc.get('balance', 0))
                        new_bal = max(0, bal - inc_cents)
                        acc['balance'] = new_bal
                        acc['updatedAt'] = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
                        accounts.upsert_item(acc)
                        st.credits_cents_spent += inc_cents
                        
                        # Create transaction record
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
                        
                        # Signal low/zero balance
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
                        # If deduction fails, pause processing but keep session open
                        st.paused_due_to_billing = True
                        if st.ws and not st._pause_notified:
                            st._pause_notified = True
                            st._resume_notified = False
                            await st.ws.send_text(json.dumps({"type": "PAUSED_BILLING"}))
    except Exception:
        pass


async def probe_billing_and_update_state(st: 'SessionState') -> None:
    """Probe billing system availability and update session state."""
    try:
        now = time.time()
        # Probe at most every 5 seconds to avoid hammering
        if now - float(st._billing_last_probe or 0.0) < 5.0:
            return
            
        st._billing_last_probe = now
        conn = os.getenv("COSMOS_CONNECTION_STRING")
        
        if not conn:
            st.paused_due_to_billing = True
            if st.ws and not st._pause_notified:
                st._pause_notified = True
                st._resume_notified = False
                await st.ws.send_text(json.dumps({"type": "PAUSED_BILLING"}))
            return
            
        # Try a lightweight read to verify availability
        accounts, _ = _get_cosmos_containers()
        try:
            _ = accounts.read()
        except Exception:
            _ = accounts.read_container()
            
        # If we reach here, billing is available
        if st.paused_due_to_billing:
            st.paused_due_to_billing = False
            if st.ws and not st._resume_notified:
                st._resume_notified = True
                st._pause_notified = False
                await st.ws.send_text(json.dumps({"type": "RESUMED_BILLING"}))
                
    except Exception:
        # Treat any failure as pause state
        st.paused_due_to_billing = True
        if st.ws and not st._pause_notified:
            st._pause_notified = True
            st._resume_notified = False
            try:
                await st.ws.send_text(json.dumps({"type": "PAUSED_BILLING"}))
            except Exception:
                pass
