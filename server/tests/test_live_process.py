"""The live process (acw/live_app.py) is the only one that runs live sessions: it serves only the live WebSocket and its
health check, the main app has no WebSocket route, and close_orphans closes the sessions nothing runs any more. The VPS
route, the live tunnel and the install script all point at that process."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocketDisconnect

from acw import auth, db
from acw.stream import routes

from .conftest import ORIGIN


def test_the_live_app_serves_only_the_websocket_and_its_health_check():
    from acw.live_app import create_live_app

    app = create_live_app()
    assert set(app.openapi()["paths"]) == {"/api/health"}  # the WebSocket is not in the schema
    with TestClient(app, base_url=ORIGIN) as c:
        assert c.get("/api/health").json() == {"ok": True, "process": "live"}
        with pytest.raises(WebSocketDenialResponse) as denied:  # the live WebSocket is there: refused for want of a sign-in
            c.websocket_connect("/api/stream/ws", headers={"origin": ORIGIN}).__enter__()
        assert denied.value.status_code == 401


def test_the_main_app_has_no_websocket_route():
    """The live handshake is answered by acw.live_app only: the main app closes the connection before accepting it."""
    from acw.app import create_app

    with TestClient(create_app(), base_url=ORIGIN) as c, pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/api/stream/ws", headers={"origin": ORIGIN}) as ws:
            ws.receive()


def test_close_orphans_closes_an_active_session_nothing_runs():
    uid = auth.get_or_create_user("a@example.com")["id"]
    db.execute("INSERT INTO stream_sessions(id, user_id, denoise_tier, caption_tier, status, started_at)"
               " VALUES ('s-lost', ?, 'light', 'local', 'active', ?)", (uid, db.now() - 600))
    assert routes.close_orphans() == 1
    assert db.one("SELECT status FROM stream_sessions WHERE id = 's-lost'")["status"] == "ended"


def test_the_live_route_tunnel_units_and_install_all_point_at_the_live_process():
    """acw-live listens on 8795 (loopback), its tunnel carries it to the VPS's 18795, the VPS sends the live WebSocket
    there only, and install.sh restarts the live tunnel with the rest. The free tools may fetch models from Hugging Face."""
    deploy = Path(__file__).resolve().parents[2] / "deploy"
    unit = (deploy / "systemd/acw-live.service").read_text()
    assert "acw.live_app:app --host 127.0.0.1 --port 8795" in unit and "0.0.0.0" not in unit
    assert "-R 127.0.0.1:18795:127.0.0.1:8795" in (deploy / "systemd/acw-tunnel-live.service").read_text()
    caddy = (deploy / "vps/audio.caddy").read_text()
    live_route = caddy.split("handle @acw_ws {")[1].split("handle ")[0]
    assert re.findall(r"reverse_proxy (\S+)", live_route) == ["127.0.0.1:18795"]
    csp = re.search(r'Content-Security-Policy "([^"]*)"', caddy).group(1)
    assert re.search(r"connect-src 'self' https://huggingface.co https://\*\.huggingface.co https://\*\.hf.co;", csp)
    units = re.search(r'^UNITS="([^"]*)"', (deploy / "install.sh").read_text(), re.M).group(1)
    assert "acw-tunnel-live.service" in units.split()
