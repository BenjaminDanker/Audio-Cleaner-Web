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


def test_login_token_is_single_use(client):
    tok = auth.create_login_token("x@example.com", "admin_login", "t")
    assert client.post("/api/auth/email/verify", json={"token": tok}, headers={"Origin": ORIGIN}).status_code == 200
    assert client.post("/api/auth/email/verify", json={"token": tok}, headers={"Origin": ORIGIN}).status_code == 400


def test_login_requires_same_origin(client):
    tok = auth.create_login_token("x@example.com", "admin_login", "t")
    r = client.post("/api/auth/email/verify", json={"token": tok}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_state_changes_need_csrf_and_origin(client):
    _, h = login(client, accept_tos=False)
    body = {"version": auth.TOS_VERSION}
    assert client.post("/api/account/tos", json=body).status_code == 403
    assert client.post("/api/account/tos", json=body, headers={"X-CSRF-Token": "nope"}).status_code == 403
    assert client.post("/api/account/tos", json=body, headers={**h, "Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/account/tos", json=body, headers=h).status_code == 200


def test_session_cookie_flags(client):
    tok = auth.create_login_token("x@example.com", "admin_login", "t")
    r = client.post("/api/auth/email/verify", json={"token": tok}, headers={"Origin": ORIGIN})
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "path=/" in cookie
    # the session id is stored hashed
    raw = r.cookies.get("acw_session")
    assert db.one("SELECT 1 FROM sessions WHERE id_hash = ?", (raw,)) is None


def test_api_key_cannot_use_browser_only_endpoints(client):
    uid, _ = login(client)
    key = make_key(uid, "stream,jobs")
    client.cookies.clear()
    h = {"Authorization": f"Bearer {key}"}
    assert client.get("/api/billing/history", headers=h).status_code == 403
    assert client.post("/api/keys", json={"name": "x"}, headers=h).status_code == 403
    assert client.get("/api/jobs", headers=h).status_code == 200


def test_api_key_scopes_and_revocation(client):
    uid, h = login(client)
    key = make_key(uid, "stream")
    client.cookies.clear()
    assert client.get("/api/jobs", headers={"Authorization": f"Bearer {key}"}).status_code == 403
    assert client.get("/api/jobs", headers={"Authorization": "Bearer acw_wrong"}).status_code == 401
    db.execute("UPDATE api_keys SET revoked_at = ? WHERE user_id = ?", (db.now(), uid))
    assert client.get("/api/me", headers={"Authorization": f"Bearer {key}"}).status_code == 401


def test_key_create_needs_credit_shows_key_once_and_stores_hash(client, signed_in):
    uid, h = signed_in
    r = client.post("/api/keys", json={"name": "obs", "scopes": ["stream"]}, headers=h)
    assert r.status_code == 402 and "credit" in r.json()["detail"]
    ledger.credit(uid, 1_000_000, "grant", "t", "g")
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


def test_signing_in_accepts_the_terms_and_records_it(client):
    uid, h = login(client, accept_tos=False)
    me = client.get("/api/me", headers=h).json()
    assert me["user"]["tosAccepted"]  # the sign-in page says signing in means agreeing
    assert db.one("SELECT count(*) AS n FROM audit_log WHERE user_id = ? AND action = 'tos_accepted'", (uid,))["n"] == 1
    body = {"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav", "sizeBytes": 100, "durationSeconds": 30}
    assert client.post("/api/jobs", json=body, headers=h).status_code == 402  # no Terms step; just no credit
    # A Terms update is accepted by continuing to use the service, and recorded again.
    db.execute("UPDATE users SET tos_version = 'old' WHERE id = ?", (uid,))
    client.post("/api/jobs", json=body, headers=h)
    assert db.one("SELECT tos_version FROM users WHERE id = ?", (uid,))["tos_version"] == auth.TOS_VERSION
    assert db.one("SELECT count(*) AS n FROM audit_log WHERE user_id = ? AND action = 'tos_accepted'", (uid,))["n"] == 2


def test_upload_must_match_declared_size_and_cancel_refunds(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 1_000_000, "grant", "t", "g")
    r = client.post("/api/jobs", json={"kind": "denoise", "denoiseTier": "standard", "filename": "a.wav",
                                       "sizeBytes": 100, "durationSeconds": 600}, headers=h)
    job_id = r.json()["id"]
    assert ledger.balance(uid)["held_micros"] == r.json()["estCostMicros"] > 0
    assert client.put(f"/api/jobs/{job_id}/input", content=b"x" * 99, headers=h).status_code == 411
    assert client.post(f"/api/jobs/{job_id}/cancel", headers=h).status_code == 200
    assert ledger.balance(uid) == {"balance_micros": 1_000_000, "held_micros": 0, "available_micros": 1_000_000}


def test_jobs_on_our_side_are_never_free(client, signed_in):
    """Free subtitles and free noise removal run in the browser: a job asking to be free is refused, whatever it is."""
    uid, h = signed_in
    ledger.credit(uid, 1_000_000, "grant", "t", "g")
    body = {"kind": "transcribe", "transcribeTier": "local", "free": True, "filename": "a.wav", "sizeBytes": 100, "durationSeconds": 60}
    for extra in ({}, {"kind": "denoise_transcribe", "denoiseTier": "standard"}, {"kind": "denoise", "denoiseTier": "standard", "transcribeTier": None}):
        r = client.post("/api/jobs", json={**body, **extra}, headers=h)
        assert r.status_code == 400 and "browser" in r.json()["detail"]
    assert db.one("SELECT COUNT(*) n FROM jobs")["n"] == 0
    assert "free" not in client.get("/api/billing/pricing").json()


def test_bad_filenames_and_types_rejected(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 1_000_000, "grant", "t", "g")
    base = {"kind": "denoise", "denoiseTier": "standard", "sizeBytes": 100, "durationSeconds": 30}
    assert client.post("/api/jobs", json={**base, "filename": "evil.exe"}, headers=h).status_code == 400
    r = client.post("/api/jobs", json={**base, "filename": "../../etc/passwd.wav"}, headers=h)
    assert r.status_code == 200
    job = db.one("SELECT input_name FROM jobs WHERE id = ?", (r.json()["id"],))
    assert job["input_name"] == "passwd.wav"


def test_security_headers_present(client):
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
    # the address can sign up again as a fresh account
    uid2, _ = login(client, "user@example.com")
    assert uid2 != uid


def test_free_tool_assets_need_a_signed_in_browser(client):
    from acw.security import limiter

    assert client.get("/api/auth/asset-check").status_code == 401
    _uid, h = login(client, "assets@example.com")
    assert client.get("/api/auth/asset-check", headers=h).status_code == 204
    for _ in range(400):
        r = client.get("/api/auth/asset-check", headers=h)
        if r.status_code == 429:
            break
    assert r.status_code == 429  # daily cap per account
    limiter.reset()
    me = client.get("/api/me", headers=h).json()
    assert "freeTranscribeSecondsLeft" not in me
