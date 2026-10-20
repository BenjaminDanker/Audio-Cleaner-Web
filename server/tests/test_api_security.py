"""Sign-in, sessions, CSRF and API keys; access between accounts; the money a job holds and returns; input checks."""
from acw import auth, db
from acw.billing import ledger
from acw.security import new_api_key

from .conftest import ORIGIN, login


def make_key(user_id, scopes="stream"):
    key, prefix, h = new_api_key()
    db.execute(
        "INSERT INTO api_keys(id,user_id,name,prefix,key_hash,scopes,created_at) VALUES (?,?,?,?,?,?,?)",
        (db.new_id(), user_id, "t", prefix, h, scopes, db.now()),
    )
    return key


def test_login_token_is_single_use_and_same_origin_only(client):
    tok = auth.create_login_token("x@example.com", "admin_login", "t")
    assert client.post("/api/auth/email/verify", json={"token": tok}, headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/auth/email/verify", json={"token": tok}, headers={"Origin": ORIGIN}).status_code == 200
    assert client.post("/api/auth/email/verify", json={"token": tok}, headers={"Origin": ORIGIN}).status_code == 400


def test_session_cookie_is_httponly_and_only_its_hash_is_stored(client):
    tok = auth.create_login_token("x@example.com", "admin_login", "t")
    r = client.post("/api/auth/email/verify", json={"token": tok}, headers={"Origin": ORIGIN})
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "path=/" in cookie
    raw = r.cookies.get("acw_session")
    assert db.one("SELECT 1 FROM sessions WHERE id_hash = ?", (raw,)) is None


def test_state_changes_need_csrf_and_origin(client):
    _, h = login(client, accept_tos=False)
    body = {"version": auth.tos_version()}
    assert client.post("/api/account/tos", json=body).status_code == 403
    assert client.post("/api/account/tos", json=body, headers={"X-CSRF-Token": "nope"}).status_code == 403
    assert client.post("/api/account/tos", json=body, headers={**h, "Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/account/tos", json=body, headers=h).status_code == 200


def test_api_keys_have_scopes_and_stop_working_when_revoked(client):
    uid, h = login(client)
    key = make_key(uid, "stream,jobs")
    client.cookies.clear()
    kh = {"Authorization": f"Bearer {key}"}
    assert client.get("/api/jobs", headers=kh).status_code == 200
    assert client.get("/api/billing/history", headers=kh).status_code == 403  # browser-only
    assert client.post("/api/keys", json={"name": "x"}, headers=kh).status_code == 403
    assert client.get("/api/keys", headers={"Authorization": "Bearer acw_wrong"}).status_code == 401
    db.execute("UPDATE api_keys SET revoked_at = ? WHERE user_id = ?", (db.now(), uid))
    assert client.get("/api/me", headers=kh).status_code == 401


def test_key_create_shows_key_once_and_stores_hash(client, signed_in):
    _, h = signed_in
    r = client.post("/api/keys", json={"name": "obs", "scopes": ["stream"]}, headers=h)
    assert r.status_code == 200
    key = r.json()["key"]
    assert key.startswith("acw_")
    assert db.one("SELECT COUNT(*) n FROM api_keys WHERE key_hash = ?", (key,))["n"] == 0
    listed = client.get("/api/keys").json()["keys"]
    assert "key" not in listed[0] and listed[0]["prefix"] == key[:12]


def test_jobs_are_private_to_their_owner(client):
    uid_a, h_a = login(client, "a@example.com")
    ledger.credit(uid_a, 1_000_000, "grant", "t", "g")
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav",
                                       "sizeBytes": 100, "durationSeconds": 30}, headers=h_a)
    assert r.status_code == 200
    job_id = r.json()["id"]
    client.cookies.clear()
    _, h_b = login(client, "b@example.com")
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert client.put(f"/api/jobs/{job_id}/input", content=b"x" * 100, headers=h_b).status_code == 404


def test_terms_are_recorded_at_sign_in_and_again_when_they_change(client):
    uid, h = login(client, accept_tos=False)
    assert client.get("/api/me", headers=h).json()["user"]["tosAccepted"]  # signing in means agreeing
    assert db.one("SELECT count(*) AS n FROM audit_log WHERE user_id = ? AND action = 'tos_accepted'", (uid,))["n"] == 1
    body = {"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav", "sizeBytes": 100, "durationSeconds": 30}
    assert client.post("/api/jobs", json=body, headers=h).status_code == 402  # no credit; no Terms step either
    db.execute("UPDATE users SET tos_version = 'old' WHERE id = ?", (uid,))
    client.post("/api/jobs", json=body, headers=h)  # continuing to use the service accepts the new version
    assert db.one("SELECT tos_version FROM users WHERE id = ?", (uid,))["tos_version"] == auth.tos_version()
    assert db.one("SELECT count(*) AS n FROM audit_log WHERE user_id = ? AND action = 'tos_accepted'", (uid,))["n"] == 2


def test_upload_must_match_declared_size_and_cancel_returns_the_money(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 1_000_000, "grant", "t", "g")
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav",
                                       "sizeBytes": 100, "durationSeconds": 600}, headers=h)
    job_id = r.json()["id"]
    assert ledger.balance(uid)["held_micros"] == r.json()["estCostMicros"] > 0
    assert client.put(f"/api/jobs/{job_id}/input", content=b"x" * 99, headers=h).status_code == 411
    assert client.post(f"/api/jobs/{job_id}/cancel", headers=h).status_code == 200
    assert ledger.balance(uid) == {"balance_micros": 1_000_000, "held_micros": 0, "available_micros": 1_000_000}


def test_unknown_job_fields_are_refused_not_ignored(client, signed_in):
    """A typo, or a retired field such as "free": ignoring it could start a paid job the caller didn't mean."""
    uid, h = signed_in
    ledger.credit(uid, 1_000_000, "grant", "t", "g")
    body = {"kind": "transcribe", "transcribeTier": "local", "free": True, "filename": "a.wav", "sizeBytes": 100, "durationSeconds": 60}
    assert client.post("/api/jobs", json=body, headers=h).status_code == 422
    assert db.one("SELECT COUNT(*) n FROM jobs")["n"] == 0


def test_file_names_cannot_escape_and_bad_types_are_refused(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 1_000_000, "grant", "t", "g")
    base = {"kind": "denoise", "denoiseTier": "standard", "sizeBytes": 100, "durationSeconds": 30}
    assert client.post("/api/jobs", json={**base, "filename": "evil.exe"}, headers=h).status_code == 400
    r = client.post("/api/jobs", json={**base, "filename": "../../etc/passwd.wav"}, headers=h)
    assert r.status_code == 200
    assert db.one("SELECT input_name FROM jobs WHERE id = ?", (r.json()["id"],))["input_name"] == "passwd.wav"


def test_security_headers_are_on_every_answer(client):
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert "default-src 'none'" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-store"


def test_suspended_user_is_locked_out(client, signed_in):
    uid, h = signed_in
    db.execute("UPDATE users SET status = 'suspended' WHERE id = ?", (uid,))
    assert client.get("/api/jobs").status_code == 403


def test_export_and_delete_account(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 3_000_000, "grant", "t", "g")
    data = client.get("/api/account/export").json()
    assert data["user"]["email"] == "user@example.com" and data["ledger"]
    assert client.post("/api/account/delete", json={"confirm": "wrong@example.com"}, headers=h).status_code == 400
    assert client.post("/api/account/delete", json={"confirm": "USER@example.com"}, headers=h).status_code == 200
    u = db.one("SELECT email, status, balance_micros FROM users WHERE id = ?", (uid,))
    assert u["status"] == "deleted" and u["balance_micros"] == 0 and u["email"].endswith("@deleted.invalid")
    assert client.get("/api/me").json()["signedIn"] is False
    uid2, _ = login(client, "user@example.com")  # the address can sign up again as a fresh account
    assert uid2 != uid


def test_free_tool_files_need_a_signed_in_browser_and_a_daily_cap(client):
    from acw.security import LIMITS, limiter

    assert client.get("/api/auth/asset-check").status_code == 401
    _uid, h = login(client, "assets@example.com")
    assert client.get("/api/auth/asset-check", headers=h).status_code == 204
    for _ in range(int(LIMITS["asset_fetch"][0])):
        client.get("/api/auth/asset-check", headers=h)
    assert client.get("/api/auth/asset-check", headers=h).status_code == 429  # daily cap per account
    limiter.reset()
