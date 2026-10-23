"""Sign in with Google (api/auth_routes.py oauth_start, oauth_callback, _google_identity) with Google's token endpoint stood in for: the
state a start stores and the cookie that goes with it, every way a callback can be refused (the cookie, the state, the provider's error, the
claims of the id token), a suspended account, and the way in."""
import base64
import hashlib
import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi import HTTPException

from acw import db
from acw.api import auth_routes
from acw.security import sha256_hex

from .conftest import ORIGIN

CLIENT_ID = "client-id.apps.googleusercontent.com"
SECRET = "client-secret"
ISSUER = "https://accounts.google.com"


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", SECRET)
    from acw.config import get_settings

    get_settings.cache_clear()


def b64(obj) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")


def id_token(_nonce, /, **over) -> str:
    claims = {"iss": ISSUER, "aud": CLIENT_ID, "exp": db.now() + 3600, "nonce": _nonce, "sub": "sub-1", "email": "Google.User@Example.com",
              "email_verified": True, "name": "Google User", **over}
    claims = {k: v for k, v in claims.items() if v is not ...}  # (... leaves a claim out)
    return f"{b64({'alg': 'RS256'})}.{b64(claims)}.signature"


class Google:
    """Google's token endpoint: what it is asked, and what it answers."""

    def __init__(self, monkeypatch):
        self.posts, self.answer, self.status = [], {}, 200
        google = self

        def handler(request: httpx.Request) -> httpx.Response:
            google.posts.append((str(request.url), parse_qs(request.content.decode())))
            if isinstance(google.answer, Exception):
                raise google.answer
            return httpx.Response(google.status, json=google.answer)

        real = httpx.AsyncClient
        monkeypatch.setattr(auth_routes.httpx, "AsyncClient", lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handler)}))


def started(client):
    """Go through the start: (state, the stored row, the redirect's query)."""
    r = client.get("/api/auth/oauth/google/start", follow_redirects=False)
    assert r.status_code == 303, r.text
    q = parse_qs(urlparse(r.headers["location"]).query)
    state = q["state"][0]
    row = db.one("SELECT * FROM oauth_states WHERE state_hash = ?", (sha256_hex(state),))
    return state, dict(row), q


def callback(client, state, code="the-code", **kw):
    return client.get("/api/auth/oauth/google/callback", params={"code": code, "state": state, **kw}, follow_redirects=False)


def error_of(r):
    assert r.status_code == 303, r.text
    return parse_qs(urlparse(r.headers["location"]).query).get("error", [None])[0]


def users():
    return [dict(r) for r in db.all_rows("SELECT * FROM users")]


# ------------------------------------------------------------------------------------------------------------------------- the start

def test_start_sends_the_browser_to_google_with_a_state_a_nonce_and_a_pkce_challenge_that_are_stored_for_the_callback(client):
    r = client.get("/api/auth/oauth/google/start", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    q = {k: v[0] for k, v in parse_qs(urlparse(r.headers["location"]).query).items()}
    assert q["client_id"] == CLIENT_ID and q["response_type"] == "code" and q["scope"] == "openid email profile" and q["prompt"] == "select_account"
    assert q["redirect_uri"] == f"{ORIGIN}/api/auth/oauth/google/callback" and q["code_challenge_method"] == "S256"
    row = db.one("SELECT * FROM oauth_states WHERE state_hash = ?", (sha256_hex(q["state"]),))
    assert row["provider"] == "google" and row["nonce"] == q["nonce"] and row["expires_at"] - row["created_at"] == auth_routes.OAUTH_STATE_TTL
    challenge = base64.urlsafe_b64encode(hashlib.sha256(row["code_verifier"].encode()).digest()).decode().rstrip("=")
    assert q["code_challenge"] == challenge and q["state"] != row["state_hash"]  # (only the hash of the state is kept)
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{auth_routes.OAUTH_COOKIE}={q['state']}") and "HttpOnly" in cookie and "SameSite=lax" in cookie.replace("Lax", "lax")
    assert "Max-Age=600" in cookie and "Path=/api/auth/oauth/" in cookie
    again = {k: v[0] for k, v in parse_qs(urlparse(client.get("/api/auth/oauth/google/start", follow_redirects=False).headers["location"]).query).items()}
    assert again["state"] != q["state"] and again["nonce"] != q["nonce"] and again["code_challenge"] != q["code_challenge"]


def test_a_provider_that_is_not_there_or_not_configured_is_a_404(client, monkeypatch):
    assert client.get("/api/auth/oauth/github/start", follow_redirects=False).status_code == 404
    assert client.get("/api/auth/oauth/github/callback", params={"code": "c", "state": "s"}, follow_redirects=False).status_code == 404
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET")
    from acw.config import get_settings

    get_settings.cache_clear()
    r = client.get("/api/auth/oauth/google/start", follow_redirects=False)
    assert r.status_code == 404 and db.one("SELECT COUNT(*) AS n FROM oauth_states")["n"] == 0
    assert client.get("/api/auth/providers").json()["google"] is False


# --------------------------------------------------------------------------------------------------------------- refused callbacks

def test_the_provider_saying_no_sends_the_person_back_to_the_sign_in_page_with_nothing_started(client, monkeypatch):
    google = Google(monkeypatch)
    state, _, _ = started(client)
    r = client.get("/api/auth/oauth/google/callback", params={"error": "access_denied", "state": state}, follow_redirects=False)
    assert error_of(r) == "cancelled" and r.headers["location"].startswith(f"{ORIGIN}/login") and users() == [] and google.posts == []
    assert db.one("SELECT COUNT(*) AS n FROM oauth_states")["n"] == 1  # (the state stays until it expires)


@pytest.mark.parametrize("what", ["no cookie", "another cookie", "no state", "no code", "a very long state"])
def test_a_callback_whose_cookie_and_state_do_not_match_is_refused_before_the_state_is_looked_up(client, monkeypatch, what):
    google = Google(monkeypatch)
    state, _, _ = started(client)
    cookie = {"no cookie": None, "another cookie": "x" * 43}.get(what, state)
    client.cookies.clear()
    if cookie is not None:
        client.cookies.set(auth_routes.OAUTH_COOKIE, cookie, path="/api/auth/oauth/")
    asked = {"no state": {"state": ""}, "no code": {"code": ""}, "a very long state": {"state": state + "a" * 200}}.get(what, {})
    r = client.get("/api/auth/oauth/google/callback", params={"code": "c", "state": state, **asked}, follow_redirects=False)
    assert error_of(r) == "state" and google.posts == [] and users() == []
    assert db.one("SELECT COUNT(*) AS n FROM oauth_states")["n"] == 1  # (a stranger's callback does not use up the real one)


def test_a_state_nobody_started_an_expired_one_and_one_used_before_are_refused(client, monkeypatch):
    google = Google(monkeypatch)
    # nobody started it
    client.cookies.set(auth_routes.OAUTH_COOKIE, "made-up-state", path="/api/auth/oauth/")
    assert error_of(callback(client, "made-up-state")) == "expired"
    # started, but too long ago
    state, _, q = started(client)
    db.execute("UPDATE oauth_states SET expires_at = ? WHERE state_hash = ?", (db.now(), sha256_hex(state)))
    assert error_of(callback(client, state)) == "expired" and google.posts == []
    assert db.one("SELECT COUNT(*) AS n FROM oauth_states")["n"] == 0  # (and gone)
    # used once: the second try finds nothing
    state, row, q = started(client)
    google.answer = {"id_token": id_token(row["nonce"])}
    assert callback(client, state).headers["location"].endswith("/app")
    client.cookies.set(auth_routes.OAUTH_COOKIE, state, path="/api/auth/oauth/")
    assert error_of(callback(client, state)) == "expired" and len(google.posts) == 1
    # a state started for another provider
    state, _, _ = started(client)
    db.execute("UPDATE oauth_states SET provider = 'other' WHERE state_hash = ?", (sha256_hex(state),))
    assert error_of(callback(client, state)) == "expired"


def test_the_token_request_carries_the_code_the_pkce_verifier_and_the_clients_secret(client, monkeypatch):
    google = Google(monkeypatch)
    state, row, _ = started(client)
    google.answer = {"id_token": id_token(row["nonce"])}
    assert callback(client, state, code="abc123").status_code == 303
    url, form = google.posts[0]
    assert url == "https://oauth2.googleapis.com/token"
    assert {k: v[0] for k, v in form.items()} == {"client_id": CLIENT_ID, "client_secret": SECRET, "code": "abc123", "grant_type": "authorization_code",
                                                  "redirect_uri": f"{ORIGIN}/api/auth/oauth/google/callback", "code_verifier": row["code_verifier"]}


@pytest.mark.parametrize("answer, status, want", [
    ({"error": "invalid_grant"}, 200, 400),  # (Google refused the code)
    ({"error": "invalid_grant"}, 400, "provider"),  # (an error status)
    ({}, 500, "provider"),
    (httpx.ConnectError("no route"), 200, "provider"),
    ({"access_token": "x"}, 200, "provider"),  # (no id token)
    ({"id_token": "not.a.jwt.at.all"}, 200, "provider"),
    ({"id_token": "a.%%%.c"}, 200, "provider"),
])
def test_a_token_endpoint_that_refuses_fails_or_answers_with_nothing_usable_signs_nobody_in(client, monkeypatch, answer, status, want):
    google = Google(monkeypatch)
    google.answer, google.status = answer, status
    state, _, _ = started(client)
    r = callback(client, state)
    if want == 400:
        assert r.status_code == 400 and "rejected by the provider" in r.json()["detail"]
    else:
        assert error_of(r) == want
    assert users() == [] and "set-cookie" not in r.headers or "acw_session" not in r.headers.get("set-cookie", "")


@pytest.mark.parametrize("over, detail", [
    ({"iss": "https://evil.example"}, "Bad token issuer."),
    ({"iss": ...}, "Bad token issuer."),
    ({"aud": "someone-elses-client"}, "Bad token audience."),
    ({"aud": ...}, "Bad token audience."),
    ({"exp": 1}, "Token expired."),
    ({"exp": ...}, "Token expired."),
    ({"nonce": "not-the-nonce"}, "Bad nonce."),
    ({"nonce": ...}, "Bad nonce."),
    ({"email_verified": False}, "email is not verified"),
    ({"email_verified": "true"}, "email is not verified"),  # (only the real true)
    ({"email_verified": ...}, "email is not verified"),
    ({"email": ""}, "email is not verified"),
    ({"email": ...}, "email is not verified"),
])
def test_an_id_token_whose_claims_are_wrong_is_refused_with_the_reason_and_nobody_is_signed_in(client, monkeypatch, over, detail):
    google = Google(monkeypatch)
    state, row, _ = started(client)
    google.answer = {"id_token": id_token(row["nonce"], **over)}
    r = callback(client, state)
    assert r.status_code == 400 and detail in r.json()["detail"]
    assert users() == [] and db.one("SELECT COUNT(*) AS n FROM identities")["n"] == 0 and db.one("SELECT COUNT(*) AS n FROM sessions")["n"] == 0


def test_google_identity_reads_the_claims_directly(monkeypatch):
    claims = auth_routes._google_identity({"id_token": id_token("n1")}, CLIENT_ID, "n1")
    assert claims == ("sub-1", "Google.User@Example.com", "Google User")
    assert auth_routes._google_identity({"id_token": id_token("n1", iss="accounts.google.com", name=...)}, CLIENT_ID, "n1")[2] is None  # (the other issuer Google uses)
    with pytest.raises(HTTPException, match="Bad nonce"):
        auth_routes._google_identity({"id_token": id_token("n1")}, CLIENT_ID, None)  # (a state stored without a nonce)
    with pytest.raises(ValueError):
        auth_routes._google_identity({}, CLIENT_ID, "n1")
    with pytest.raises(ValueError):
        auth_routes._jwt_payload("only.two")


# ---------------------------------------------------------------------------------------------------------------- the way in

def test_a_good_callback_creates_the_account_links_the_identity_starts_a_session_and_clears_the_cookie(client, monkeypatch):
    google = Google(monkeypatch)
    state, row, _ = started(client)
    google.answer = {"id_token": id_token(row["nonce"])}
    r = callback(client, state)
    assert r.status_code == 303 and r.headers["location"] == f"{ORIGIN}/app"
    (u,) = users()
    assert u["email"] == "google.user@example.com" and u["status"] == "active"
    ident = db.one("SELECT * FROM identities")
    assert (ident["provider"], ident["provider_user_id"], ident["user_id"]) == ("google", "sub-1", u["id"])
    cookies = r.headers.get_list("set-cookie")
    assert any(c.startswith(f"{auth_routes.OAUTH_COOKIE}=") and ("Max-Age=0" in c or "expires" in c.lower()) for c in cookies)  # (cleared)
    me = client.get("/api/me").json()
    assert me["user"]["email"] == "google.user@example.com" and db.one("SELECT COUNT(*) AS n FROM oauth_states")["n"] == 0


def test_signing_in_again_finds_the_same_account_by_the_identity_and_an_existing_account_is_linked_by_its_verified_email(client, monkeypatch):
    from acw import auth

    google = Google(monkeypatch)
    existing = auth.get_or_create_user("google.user@example.com")["id"]  # (made with an email link before)
    for _ in range(2):
        state, row, _ = started(client)
        google.answer = {"id_token": id_token(row["nonce"])}
        assert callback(client, state).headers["location"].endswith("/app")
        client.cookies.clear()
    assert [u["id"] for u in users()] == [existing] and db.one("SELECT COUNT(*) AS n FROM identities")["n"] == 1
    # the same Google account with another address now: still the same person (the identity decides)
    state, row, _ = started(client)
    google.answer = {"id_token": id_token(row["nonce"], email="new.address@example.com")}
    assert callback(client, state).headers["location"].endswith("/app")
    assert [u["id"] for u in users()] == [existing] and client.get("/api/me").json()["user"]["id"] == existing


@pytest.mark.parametrize("status", ["suspended", "deleted"])
def test_an_account_that_is_not_active_is_sent_back_without_a_session(client, monkeypatch, status):
    from acw import auth

    google = Google(monkeypatch)
    uid = auth.get_or_create_user("google.user@example.com")["id"]
    db.execute("UPDATE users SET status = ? WHERE id = ?", (status, uid))
    state, row, _ = started(client)
    google.answer = {"id_token": id_token(row["nonce"])}
    r = callback(client, state)
    assert error_of(r) == "suspended"
    assert db.one("SELECT COUNT(*) AS n FROM sessions")["n"] == 0 and client.get("/api/me").json().get("user") in (None, {})
