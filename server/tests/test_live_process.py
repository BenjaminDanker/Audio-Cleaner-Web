"""The live process (acw/live_app.py): it serves only the live WebSocket and its health check, each process closes only
its own orphaned sessions, and the cores both processes use are read together."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse

from acw import auth, broker, db
from acw.stream import routes

from .conftest import ORIGIN


@pytest.fixture
def role(monkeypatch):
    """Whichever process role a test sets is undone afterwards (routes._role is a module global)."""
    monkeypatch.setattr(routes, "_role", "web")
    return routes.set_role


def test_the_live_app_serves_only_the_websocket_and_its_health_check(role):
    from acw.live_app import create_live_app

    app = create_live_app()
    assert set(app.openapi()["paths"]) == {"/api/health"}  # the HTTP routes (the WebSocket is not in the schema)
    with TestClient(app, base_url=ORIGIN) as c:
        assert routes._role == "live"  # started as the live process
        assert c.get("/api/health").json() == {"ok": True, "process": "live"}
        assert all(c.get(p).status_code == 404 for p in ("/api/jobs", "/api/auth/me", "/docs", "/api/downloads/obs-plugin/latest/windows"))
        with pytest.raises(WebSocketDenialResponse) as denied:  # the live WebSocket is there: refused for want of a sign-in
            c.websocket_connect("/api/stream/ws", headers={"origin": ORIGIN}).__enter__()
        assert denied.value.status_code == 401


def test_each_process_closes_only_its_own_orphaned_sessions(role):
    uid = auth.get_or_create_user("a@example.com")["id"]
    for sid, owner in (("s-web", "web"), ("s-live", "live"), ("s-old", None)):  # s-old: from before owners were recorded
        db.execute("INSERT INTO stream_sessions(id, user_id, denoise_tier, caption_tier, status, started_at, owner)"
                   " VALUES (?,?,'light','local','active',?,?)", (sid, uid, db.now() - 600, owner))
    active = lambda: {r["id"] for r in db.all_rows("SELECT id FROM stream_sessions WHERE status = 'active'")}  # noqa: E731
    role("live")
    assert routes.close_orphans() == 1 and active() == {"s-web", "s-old"}
    role("web")
    assert routes.close_orphans() == 2 and active() == set()
    db.execute("INSERT INTO stream_sessions(id, user_id, denoise_tier, caption_tier, status, started_at, owner)"
               " VALUES ('s-live2',?,'light','local','active',?,'live')", (uid, db.now() - 600))
    assert routes.close_orphans() == 0 and active() == {"s-live2"}  # the web process leaves the live one's sessions alone


def test_live_cores_of_both_processes_are_read_together(tmp_path):
    broker.publish_live_cores([1, 2], tmp_path)
    broker.publish_live_cores([2, 5], tmp_path, role="live")
    assert broker.live_cores(tmp_path) == [1, 2, 5]
    broker.publish_live_cores([], tmp_path, role="live")
    assert broker.live_cores(tmp_path) == [1, 2]
    assert broker.live_cores(tmp_path / "nowhere") == []


def test_the_unit_the_tunnel_and_the_vps_route_agree_on_the_live_ports():
    """acw-live listens on 8795 (loopback), its tunnel carries it to the VPS's 18795, and the VPS sends the live WebSocket
    there first and to the main process (18790) when it is down. The free tools may fetch models from Hugging Face."""
    deploy = Path(__file__).resolve().parents[2] / "deploy"
    unit = (deploy / "systemd/acw-live.service").read_text()
    assert "acw.live_app:app --host 127.0.0.1 --port 8795" in unit and "0.0.0.0" not in unit
    assert "-R 127.0.0.1:18795:127.0.0.1:8795" in (deploy / "systemd/acw-tunnel-live.service").read_text()
    caddy = (deploy / "vps/audio.caddy").read_text()
    assert "handle @acw_ws {" in caddy
    assert "reverse_proxy 127.0.0.1:18795 127.0.0.1:18790 {" in caddy.split("handle @acw_ws {")[1].split("handle ")[0]
    csp = re.search(r'Content-Security-Policy "([^"]*)"', caddy).group(1)
    assert re.search(r"connect-src 'self' https://huggingface.co https://\*\.huggingface.co https://\*\.hf.co;", csp)
