"""The contact form: POST /api/contact mails the owner; the owner's address never reaches a browser; limits; no message
text is stored."""
import pytest

from acw import alerts, db, security
from acw.api import contact_routes
from acw.config import get_settings

from .conftest import ORIGIN, login

OWNER = "owner-secret@example.org"
H = {"Origin": ORIGIN}
GOOD = {"name": "Sam", "email": "sam@example.com", "topic": "question", "message": "How long are results kept for?"}


@pytest.fixture
def mail(monkeypatch):
    """Mail configured (alert address only); captures what would be sent."""
    sent = []
    monkeypatch.setenv("ACW_ALERT_EMAIL", OWNER)
    monkeypatch.delenv("ACW_CONTACT_EMAIL", raising=False)
    monkeypatch.setenv("SMTP_HOST", "smtp.invalid")
    monkeypatch.setenv("SMTP_FROM", "acw@example.com")
    get_settings.cache_clear()
    monkeypatch.setattr(alerts, "_deliver_message", sent.append)
    return sent


def send(client, headers=H, **over):
    return client.post("/api/contact", json={**GOOD, **over}, headers=headers)


def test_message_is_mailed_with_reply_to_and_no_address_or_text_in_the_log(client, mail):
    r = send(client)
    assert r.status_code == 200 and r.json() == {"ok": True}
    (msg,) = mail
    assert msg["To"] == OWNER and msg["Reply-To"] == "sam@example.com" and msg["From"] == "acw@example.com"
    assert msg["Subject"] == "[Audio Cleaner] Question: How long are results kept for?"
    body = msg.get_content()
    assert "Sam <sam@example.com>" in body and "Signed in: no" in body and "How long are results kept for?" in body
    row = db.one("SELECT * FROM audit_log WHERE action = 'contact_sent'")
    assert row["detail"] == "question" and "results" not in str(dict(row)) and "sam@example.com" not in str(dict(row))


def test_signed_in_post_needs_the_csrf_token_and_a_same_origin_request(client, mail):
    _, h = login(client, "acct@example.com")
    assert send(client, headers=H).status_code == 403  # cookie but no token
    assert send(client, headers={**h, "Origin": "https://evil.example"}).status_code == 403
    assert not mail
    assert send(client, headers={"Origin": "https://evil.example"}).status_code == 403
    assert not mail


def test_job_id_is_looked_up_and_must_be_a_real_job(client, mail):
    uid, h = login(client, "j@example.com")
    jid = db.new_id()
    db.execute(
        "INSERT INTO jobs(id,user_id,kind,tier,status,params_json,input_name,input_bytes,claimed_seconds,created_at,error)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (jid, uid, "denoise", "standard", "failed", "{}", "x.wav", 1, 1.0, db.now(), "Could not read the file"),
    )
    assert send(client, headers=h, topic="job", job=jid).status_code == 200
    body = mail[0].get_content()
    assert f"Job: {jid} (denoise, failed, this account’s" in body and "Could not read the file" in body
    assert send(client, headers=h, topic="job", job="not a job id").status_code == 400
    assert len(mail) == 1


def test_validation_refuses_bad_addresses_and_lengths_and_accepts_the_longest_message(client, mail):
    assert send(client, email="not-an-email").status_code == 400
    assert send(client, email="a@b.com, c@d.com").status_code == 400
    assert send(client, email="a@b.com\nBcc: x@y.com").status_code == 400
    assert send(client, message="short").status_code == 400
    assert send(client, message="          x          ").status_code == 400
    assert send(client, message="x" * 5001).status_code == 422
    assert send(client, topic="nonsense").status_code == 422
    assert send(client, message="x" * 5000).status_code == 200
    assert len(mail) == 1


def test_header_injection_and_control_characters_are_stripped(client, mail):
    evil = "Sam\r\nBcc: attacker@example.net"
    r = send(client, name=evil, message="Hello there\r\nBcc: attacker@example.net\x00\x07‮ is this safe?")
    assert r.status_code == 200
    (msg,) = mail
    assert msg["Bcc"] is None and msg["Cc"] is None
    assert "\n" not in msg["Subject"] and "\r" not in msg["Subject"] and "‮" not in msg["Subject"]
    assert msg["To"] == OWNER and msg["Reply-To"] == "sam@example.com"
    body = msg.get_content()
    assert "\x00" not in body and "\x07" not in body and "‮" not in body and "\r" not in body
    assert "Sam Bcc: attacker@example.net <sam@example.com>" in body  # one line: it is text, not a header


def test_a_honeypot_field_looks_like_success_but_sends_nothing(client, mail):
    r = send(client, website="http://spam.example")
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert not mail
    assert db.one("SELECT COUNT(*) n FROM audit_log WHERE action = 'contact_sent'")["n"] == 0


def test_limits_per_network_and_per_mailbox(client, mail):
    # three an hour per network: the fourth is refused with a wait
    for i in range(3):
        assert send(client, email=f"p{i}@example.com").status_code == 200
    r = send(client, email="p9@example.com")
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0
    assert len(mail) == 3
    # plus tags and Gmail dots are the same mailbox: one message per hour for it, whichever network
    for i, addr in enumerate(["a.b@gmail.com", "ab+one@gmail.com", "a.b+two@gmail.com"]):
        h = {**H, "X-Forwarded-For": f"203.0.113.{i + 1}"}
        assert send(client, headers=h, email=addr).status_code == 200
    h = {**H, "X-Forwarded-For": "203.0.113.50"}
    assert send(client, headers=h, email="ab@gmail.com").status_code == 429


def test_a_network_gets_ten_messages_a_day(client, mail):
    h = {**H, "X-Forwarded-For": "203.0.113.7"}
    for i in range(10):
        # refill the hourly bucket between messages so only the daily limit can stop us
        security.limiter._tables.get("contact_ip", {}).clear()
        assert send(client, headers=h, email=f"d{i}@example.com").status_code == 200
    security.limiter._tables.get("contact_ip", {}).clear()
    assert send(client, headers=h, email="d99@example.com").status_code == 429
    assert len(mail) == 10


def test_the_whole_site_has_a_daily_cap_that_says_try_tomorrow(client, mail):
    now = db.now()
    for i in range(contact_routes.CONTACT_DAILY_CAP):
        db.execute("INSERT INTO audit_log(user_id, actor, action, detail, ip, created_at) VALUES (NULL,'visitor','contact_sent','question',NULL,?)",
                   (now - 3600 - i,))
    r = send(client, email="cap@example.com")
    assert r.status_code == 429 and "tomorrow" in r.json()["detail"]
    assert 60 <= int(r.headers["Retry-After"]) <= 86400
    assert not mail
    # the oldest of them ages out of the rolling day: sending works again
    db.execute("UPDATE audit_log SET created_at = created_at - 90000 WHERE id = (SELECT MIN(id) FROM audit_log WHERE action = 'contact_sent')")
    assert send(client, email="cap2@example.com").status_code == 200


def test_me_and_the_public_pages_say_the_form_exists_without_the_address(client, mail):
    me = client.get("/api/me")
    assert me.json()["contactForm"] is True
    assert OWNER not in me.text and "example.org" not in me.text
    for path in ("/api/auth/providers", "/api/health", "/api/billing/pricing"):
        assert OWNER not in client.get(path).text


def test_the_address_never_appears_in_errors_and_a_failed_send_is_not_counted(client, mail, monkeypatch):
    monkeypatch.setattr(alerts, "_deliver_message", lambda m: (_ for _ in ()).throw(RuntimeError(f"550 {OWNER} rejected")))
    r = send(client)
    assert r.status_code == 503 and OWNER not in r.text and "rejected" not in r.text
    assert db.one("SELECT COUNT(*) n FROM audit_log WHERE action = 'contact_sent'")["n"] == 0


def test_message_text_is_not_stored_anywhere(client, mail):
    send(client, message="a very private sentence about my account")
    for table in ("audit_log", "jobs", "users", "alert_state"):
        for row in db.all_rows(f"SELECT * FROM {table}"):
            assert "private sentence" not in str(dict(row))


def test_the_real_send_path_uses_tls_login_and_the_reply_to(client, monkeypatch):
    import ssl

    seen = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            seen.update(host=host, port=port, timeout=timeout)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, context=None):
            seen["context"] = context

        def login(self, user, password):
            seen["login"] = (user, password)

        def send_message(self, msg):
            seen["msg"] = msg

    monkeypatch.setenv("ACW_CONTACT_EMAIL", "help@example.org")
    monkeypatch.setenv("SMTP_HOST", "smtp.invalid")
    monkeypatch.setenv("SMTP_USER", "acw@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    monkeypatch.setenv("SMTP_FROM", "acw@example.com")
    get_settings.cache_clear()
    monkeypatch.setattr(alerts.smtplib, "SMTP", FakeSMTP)
    assert send(client).status_code == 200
    assert seen["context"].verify_mode == ssl.CERT_REQUIRED and seen["context"].check_hostname
    assert seen["login"] == ("acw@example.com", "pw") and seen["timeout"] <= 30
    assert seen["msg"]["To"] == "help@example.org" and seen["msg"]["Reply-To"] == "sam@example.com"
