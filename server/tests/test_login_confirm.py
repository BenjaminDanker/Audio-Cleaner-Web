"""The sign-in link's confirm step: a link asked for by e-mail remembers the asking browser (HttpOnly cookie acw_login, only its SHA-256 in
login_tokens.browser_hash). In that browser it signs in at once; anywhere else the server answers 409 confirm_sign_in with the address and
leaves the link unused until {token, confirm: true}. Links nobody's browser asked for (the command line) sign in as before."""
import re

import pytest

from acw import auth, db
from acw.security import sha256_hex

from .conftest import ORIGIN

H = {"Origin": ORIGIN}
COOKIE = "acw_login"
CODE = "confirm_sign_in"


@pytest.fixture
def mailbox(monkeypatch):
    """Sign-in mail is on and captured: a list of (to, subject, body)."""
    from acw.api import auth_routes
    from acw.config import get_settings

    monkeypatch.setenv("SMTP_HOST", "smtp.test")
    monkeypatch.setenv("SMTP_FROM", "noreply@test")
    get_settings.cache_clear()
    sent = []
    monkeypatch.setattr(auth_routes, "_send_mail", lambda to, subject, body: sent.append((to, subject, body)))
    return sent


def ask(client, mailbox, email="ann@example.org", headers=None):
    """Ask for a link the way the sign-in page does. Returns (the answer, the link's token)."""
    r = client.post("/api/auth/email", json={"email": email}, headers=headers or H)
    assert r.status_code == 200, r.text
    to, _subject, body = mailbox[-1]
    assert to == email
    token = re.search(r"/login/verify#token=([A-Za-z0-9_-]+)", body).group(1)
    return r, token


def verify(client, token, confirm=None, headers=None):
    body = {"token": token} if confirm is None else {"token": token, "confirm": confirm}
    return client.post("/api/auth/email/verify", json=body, headers=headers or H)


def used(token):
    return db.one("SELECT used_at FROM login_tokens WHERE token_hash = ?", (sha256_hex(token),))["used_at"] is not None


def signed_in_as(client):
    me = client.get("/api/me").json()
    return me["user"]["email"] if me["signedIn"] else None


def another_browser(client, value=None):
    """The same client as a browser that has no acw_login cookie (or a different one) and no session."""
    client.cookies.clear()
    if value is not None:
        client.cookies.set(COOKIE, value, domain="testserver.local", path="/api/auth/email")


def set_cookies(r):
    return r.headers.get_list("set-cookie")


# -------------------------------------------------------------------------------------------------- the asking browser

def test_the_browser_that_asked_for_the_link_is_signed_in_at_once(client, mailbox):
    r, token = ask(client, mailbox)
    cookie = client.cookies.get(COOKIE)
    assert cookie and re.fullmatch(r"[A-Za-z0-9_-]{20,64}", cookie)
    assert db.one("SELECT browser_hash FROM login_tokens WHERE token_hash = ?", (sha256_hex(token),))["browser_hash"] == sha256_hex(cookie)
    r = verify(client, token)
    assert f"{COOKIE}={cookie}" in r.request.headers.get("cookie", "")  # (sent by the jar on its own: the path of the cookie covers the verify call)
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert signed_in_as(client) == "ann@example.org" and used(token)
    assert verify(client, token).status_code == 400  # (single use, even from the same browser)


def test_the_asking_browser_may_also_send_confirm(client, mailbox):
    _, token = ask(client, mailbox)
    assert verify(client, token, True).status_code == 200 and signed_in_as(client) == "ann@example.org"


def test_only_the_hash_of_the_cookie_is_stored(client, mailbox):
    _, token = ask(client, mailbox)
    cookie = client.cookies.get(COOKIE)
    stored = [r["browser_hash"] for r in db.all_rows("SELECT browser_hash FROM login_tokens")]
    assert stored == [sha256_hex(cookie)] and cookie not in stored[0]


# ----------------------------------------------------------------------------------------------------- another browser

@pytest.mark.parametrize("kind", ["no cookie", "wrong cookie"])
def test_another_browser_is_asked_first_and_the_link_stays_good(client, mailbox, kind):
    _, token = ask(client, mailbox)
    another_browser(client, None if kind == "no cookie" else "w" * 24)
    r = verify(client, token)
    sent = r.request.headers.get("cookie", "")  # (what the browser really sent: the test must not pass because a cookie was dropped)
    assert (f"{COOKIE}={'w' * 24}" in sent) == (kind == "wrong cookie") and (COOKIE in sent) == (kind == "wrong cookie")
    assert r.status_code == 409, r.text
    assert r.json()["code"] == CODE and r.json()["email"] == "ann@example.org" and r.headers["X-Error-Code"] == CODE
    assert "detail" in r.json() and not any(c.startswith("acw_session") or "session=" in c for c in set_cookies(r))
    assert signed_in_as(client) is None and not used(token)
    assert db.one("SELECT 1 FROM users WHERE email = ?", ("ann@example.org",)) is None  # (no account is made by asking)
    assert verify(client, token, False).status_code == 409 and not used(token)  # still not used, whatever number of times it is asked
    assert verify(client, token, False, {"Origin": "https://evil.example"}).status_code == 403  # (the same-origin check comes first)
    r = verify(client, token, True)
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert signed_in_as(client) == "ann@example.org" and used(token)
    assert verify(client, token, True).status_code == 400  # a second use: not with confirm, not without
    assert verify(client, token).status_code == 400


def test_the_confirm_of_one_link_does_not_open_another(client, mailbox):
    _, a = ask(client, mailbox, "ann@example.org")
    _, b = ask(client, mailbox, "bob@example.org")
    another_browser(client)
    assert verify(client, a, True).status_code == 200 and signed_in_as(client) == "ann@example.org"
    assert not used(b)  # (a yes is for the one link it names)
    client.cookies.clear()
    r = verify(client, b)
    assert r.status_code == 409 and r.json()["email"] == "bob@example.org"


def test_every_link_a_browser_asked_for_knows_that_browser(client, mailbox):
    _, a = ask(client, mailbox, "ann@example.org")
    _, b = ask(client, mailbox, "ann@example.org")  # (the second ask keeps the cookie: both links know this browser)
    assert verify(client, b).status_code == 200 and signed_in_as(client) == "ann@example.org"
    assert verify(client, a).status_code == 200  # (the older link opens in the same browser as well: no question)


def test_a_browser_that_asked_for_one_link_is_asked_about_a_link_somebody_else_asked_for(client, mailbox):
    """The case the confirm step is for: a link made for somebody else's browser (sent to this person) opened in a browser with an acw_login cookie of its own."""
    _, theirs = ask(client, mailbox, "victim-target@example.org")
    another_browser(client)
    _, mine = ask(client, mailbox, "me@example.org")  # (this browser has a cookie now, a different one)
    assert verify(client, theirs).status_code == 409 and signed_in_as(client) is None
    assert verify(client, mine).status_code == 200 and signed_in_as(client) == "me@example.org"


# ---------------------------------------------------------------------------------------- links nobody's browser asked for

@pytest.mark.parametrize("cookie", [None, "z" * 30])
def test_a_command_line_link_signs_in_with_no_cookie_and_no_confirm(client, cookie):
    token = auth.create_login_token("cli@example.org", "admin_login", "cli")
    assert db.one("SELECT browser_hash FROM login_tokens WHERE token_hash = ?", (sha256_hex(token),))["browser_hash"] is None
    another_browser(client, cookie)  # (a cookie it does not know changes nothing)
    r = verify(client, token)
    assert (COOKIE in r.request.headers.get("cookie", "")) == (cookie is not None)
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert signed_in_as(client) == "cli@example.org"
    assert verify(client, token).status_code == 400


def test_the_consume_function_itself(client):
    cli = auth.create_login_token("cli@example.org", "admin_login", "cli")
    assert auth.consume_login_token(cli) == "cli@example.org"
    assert auth.consume_login_token(cli) is None
    bound = auth.create_login_token("Bound@Example.org", "email_login", "1.2.3.4", "b" * 30)
    assert auth.consume_login_token(bound, "b" * 30) == "bound@example.org"
    other = auth.create_login_token("bound@example.org", "email_login", "1.2.3.4", "b" * 30)
    for wrong in (None, "", "c" * 30, "b" * 29):
        with pytest.raises(auth.ConfirmSignIn) as e:
            auth.consume_login_token(other, wrong)
        assert e.value.email == "bound@example.org" and not used(other)
        with pytest.raises(auth.ConfirmSignIn):
            auth.consume_login_token(other, wrong, False)
    assert auth.consume_login_token(other, "c" * 30, True) == "bound@example.org"
    assert auth.consume_login_token(other, "c" * 30, True) is None


# ------------------------------------------------------------------------------------------- invalid and expired links

@pytest.mark.parametrize("confirm", [None, False, True])
def test_an_invalid_link_is_400_with_and_without_confirm(client, confirm):
    for cookie in (None, "q" * 30):
        another_browser(client, cookie)
        r = verify(client, "not-a-token", confirm)
        assert r.status_code == 400 and "invalid or expired" in r.json()["detail"] and "code" not in r.json() and CODE not in r.headers.get("X-Error-Code", "")


@pytest.mark.parametrize("confirm", [None, True])
def test_an_expired_link_is_400_not_a_question_and_never_names_its_address(client, mailbox, confirm):
    _, token = ask(client, mailbox)
    db.execute("UPDATE login_tokens SET expires_at = ?", (db.now() - 1,))
    for same_browser in (True, False):
        if not same_browser:
            another_browser(client)
        r = verify(client, token, confirm)
        assert r.status_code == 400 and "ann@example.org" not in r.text and "code" not in r.json()
    assert signed_in_as(client) is None


def test_a_used_link_opened_in_another_browser_is_400_not_a_question(client, mailbox):
    _, token = ask(client, mailbox)
    assert verify(client, token).status_code == 200
    another_browser(client)
    r = verify(client, token)
    assert r.status_code == 400 and "ann@example.org" not in r.text


# ------------------------------------------------------------------------------------------- asking: the cookie it sets

def test_asking_for_a_link_sets_the_httponly_cookie_on_the_sign_in_path(client, mailbox):
    r, token = ask(client, mailbox)
    (c,) = [c for c in set_cookies(r) if c.startswith(COOKIE + "=")]
    low = c.lower()
    assert "httponly" in low and "path=/api/auth/email" in low and "samesite=lax" in low and f"max-age={auth.LOGIN_LINK_TTL}" in low
    assert "secure" not in low.replace("samesite", "")  # (tests run as development: no Secure over plain http)
    value = re.match(rf"{COOKIE}=([^;]+)", c).group(1)
    assert re.fullmatch(r"[A-Za-z0-9_-]{20,64}", value)
    assert value not in mailbox[-1][2]  # (the mail never holds the cookie value)


def test_the_cookie_is_secure_in_production_and_its_path_follows_the_base_path(client, mailbox, monkeypatch):
    from acw.config import get_settings

    monkeypatch.setenv("ACW_ENV", "production")
    monkeypatch.setenv("ACW_SESSION_SECRET", "x" * 40)
    monkeypatch.setenv("ACW_BASE_PATH", "/audio")
    get_settings.cache_clear()
    r, _ = ask(client, mailbox)
    (c,) = [c for c in set_cookies(r) if c.startswith(COOKIE + "=")]
    low = c.lower()
    assert "secure" in low.replace("samesite", "") and "httponly" in low and "path=/audio/api/auth/email" in low


def test_asking_again_keeps_the_valid_cookie_the_browser_sent(client, mailbox):
    r1, t1 = ask(client, mailbox)
    first = client.cookies.get(COOKIE)
    r2, t2 = ask(client, mailbox)
    (c2,) = [c for c in set_cookies(r2) if c.startswith(COOKIE + "=")]
    assert re.match(rf"{COOKIE}=([^;]+)", c2).group(1) == first  # (the same value, sent again with a fresh lifetime)
    assert client.cookies.get(COOKIE) == first and t1 != t2
    hashes = {r["browser_hash"] for r in db.all_rows("SELECT browser_hash FROM login_tokens")}
    assert hashes == {sha256_hex(first)}
    # a browser with no cookie gets a new one, and each is its own
    another_browser(client)
    ask(client, mailbox, "bob@example.org")
    assert client.cookies.get(COOKIE) and client.cookies.get(COOKIE) != first


@pytest.mark.parametrize("bad", ["short", "x" * 65, "dots.are.not.allowed." + "c" * 30, "bang!" + "c" * 30])
def test_a_cookie_of_the_wrong_shape_is_replaced_not_stored(client, mailbox, bad):
    r = client.post("/api/auth/email", json={"email": "ann@example.org"}, headers={**H, "Cookie": f"{COOKIE}={bad}"})
    assert r.status_code == 200
    (c,) = [c for c in set_cookies(r) if c.startswith(COOKIE + "=")]
    value = re.match(rf"{COOKIE}=([^;]+)", c).group(1)
    assert value != bad and re.fullmatch(r"[A-Za-z0-9_-]{20,64}", value)
    (row,) = db.all_rows("SELECT browser_hash FROM login_tokens")
    assert row["browser_hash"] == sha256_hex(value)


def test_a_refused_ask_sets_no_cookie_and_makes_no_link(client, mailbox):
    r = client.post("/api/auth/email", json={"email": "not an address"}, headers=H)
    assert r.status_code == 400 and not any(c.startswith(COOKIE + "=") for c in set_cookies(r))
    r = client.post("/api/auth/email", json={"email": "ann@example.org"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403 and not any(c.startswith(COOKIE + "=") for c in set_cookies(r))
    assert db.one("SELECT COUNT(*) AS n FROM login_tokens")["n"] == 0 and mailbox == []


# ----------------------------------------------------------------------------------------------------------- migration

def test_init_db_adds_browser_hash_to_a_table_that_has_none_and_old_links_still_work():
    cols = lambda: {r["name"] for r in db.all_rows("PRAGMA table_info(login_tokens)")}  # noqa: E731
    assert "browser_hash" in cols()
    db.execute("ALTER TABLE login_tokens DROP COLUMN browser_hash")  # (a database made before the confirm step)
    assert "browser_hash" not in cols()
    now = db.now()
    db.execute("INSERT INTO login_tokens(token_hash, email, purpose, created_at, expires_at, ip) VALUES (?,?,?,?,?,?)",
               (sha256_hex("old-link"), "old@example.org", "email_login", now, now + 600, "1.2.3.4"))
    db.init_db()
    assert "browser_hash" in cols()
    assert db.one("SELECT browser_hash FROM login_tokens WHERE token_hash = ?", (sha256_hex("old-link"),))["browser_hash"] is None
    assert auth.consume_login_token("old-link") == "old@example.org"  # (a link asked for before the change signs in as it did: no browser to compare)
    db.init_db()  # (and a second run changes nothing)
    assert "browser_hash" in cols()
