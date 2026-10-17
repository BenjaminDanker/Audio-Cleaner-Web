"""Job callbacks (callbackUrl): address checks, SSRF refusal, signing, delivery and retries. Nothing here touches the
network: DNS and the connection are replaced (`webhooks._getaddrinfo`, `webhooks._open`)."""
import asyncio
import hashlib
import hmac
import json
import re
import ssl
import time

import pytest

from acw import db, jobs, webhooks
from acw.billing import ledger

URL = "https://hooks.example.com/acw?token=secret-token"
PUBLIC = "93.184.216.34"
REAL_OWN_ADDRESSES = webhooks.own_addresses  # the fixture below replaces it for every test


class FakeWriter:
    def __init__(self):
        self.sent = b""
        self.closed = False

    def write(self, data):
        self.sent += data

    async def drain(self):
        pass

    def close(self):
        self.closed = True


class Net:
    """The replaced network: what DNS answers, what the receiver answers, and what was asked of both."""

    def __init__(self):
        self.dns = {}  # host -> list of addresses
        self.default_dns = [PUBLIC]
        self.dns_calls = []
        self.connects = []  # (ip, host)
        self.writers = []
        self.answer = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
        self.error = None
        self.delay = 0.0

    async def getaddrinfo(self, host):
        self.dns_calls.append(host)
        return self.dns.get(host, self.default_dns)

    async def open(self, ip, host):
        self.connects.append((ip, host))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        reader = asyncio.StreamReader()
        reader.feed_data(self.answer)
        reader.feed_eof()
        writer = FakeWriter()
        self.writers.append(writer)
        return reader, writer


@pytest.fixture(autouse=True)
def net(monkeypatch):
    n = Net()
    monkeypatch.setattr(webhooks, "_getaddrinfo", n.getaddrinfo)
    monkeypatch.setattr(webhooks, "_open", n.open)
    monkeypatch.setattr(webhooks, "own_addresses", lambda: frozenset({"198.51.100.77"}))
    monkeypatch.setattr(webhooks, "POLL_SECONDS", 3600)  # the background loop of the app stays out of the way
    return n


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- the address

def test_good_addresses_are_accepted_and_normalised():
    for url, expected in [
        ("https://hooks.example.com/acw", "https://hooks.example.com/acw"),
        ("https://hooks.example.com:443/acw?a=b&c=d", "https://hooks.example.com/acw?a=b&c=d"),
        ("HTTPS://Hooks.Example.COM/a%20b", "https://hooks.example.com/a%20b"),
        ("  https://hooks.example.com./x  ", "https://hooks.example.com/x"),
    ]:
        assert webhooks.check_url(url) == expected, url


def test_bad_addresses_are_refused():
    cases = [
        ("", "empty"),
        ("http://hooks.example.com/x", "https"),
        ("//hooks.example.com/x", "https"),
        ("https://hooks.example.com:8443/x", "443"),
        ("https://user:pw@hooks.example.com/x", "user name"),
        ("https://hooks.example.com/x#frag", "fragment"),
        ("https://127.0.0.1/x", "IP address"),
        ("https://[2606:4700::1111]/x", "IP address"),
        ("https://2130706433/x", "public host name"),  # decimal 127.0.0.1: not a name with a dot
        ("https://localhost/x", "public host name"),
        ("https://db.internal/x", "public host name"),
        ("https://hooks.exämple.com/x", "ASCII"),
        ("https://hooks.example.com/" + "a" * 600, "at most"),
        ("https://testserver/x", "public host name"),
    ]
    for url, why in cases:
        with pytest.raises(ValueError) as e:
            webhooks.check_url(url)
        assert why in str(e.value), url


def test_the_service_itself_is_not_a_target(monkeypatch):
    from acw.config import get_settings

    monkeypatch.setenv("ACW_PUBLIC_ORIGIN", "https://audio.example.net")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="back at this service"):
        webhooks.check_url("https://audio.example.net/api/jobs")


def test_only_the_public_internet_is_a_target():
    cases = [
        ("93.184.216.34", True), ("2606:4700:4700::1111", True),
        ("0.0.0.0", False), ("10.0.0.1", False), ("100.64.0.1", False),  # CGNAT
        ("127.0.0.1", False), ("169.254.169.254", False), ("172.16.0.1", False), ("192.168.1.1", False),
        ("198.51.100.1", False), ("224.0.0.1", False), ("240.0.0.1", False),
        ("::1", False), ("fe80::1", False), ("fc00::1", False), ("::ffff:127.0.0.1", False),  # IPv4 inside IPv6
        ("::ffff:169.254.169.254", False), ("64:ff9b::7f00:1", False), ("2002:7f00:1::1", False),
        ("198.51.100.77", False),  # this machine's own address (the fixture's)
        ("not an address", False), ("", False),
    ]
    for ip, public in cases:
        assert webhooks.is_public_ip(ip) is public, ip


def test_the_machines_own_addresses_are_found_and_refused(monkeypatch):
    monkeypatch.setattr(webhooks, "own_addresses", REAL_OWN_ADDRESSES)
    monkeypatch.setattr(webhooks.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.99", 0)), (10, 1, 6, "", ("::ffff:93.184.216.98", 0, 0, 0))])
    webhooks._own.update(at=-1e9)
    found = webhooks.own_addresses()
    assert isinstance(found, frozenset) and {"93.184.216.99", "93.184.216.98"} <= found
    assert not webhooks.is_public_ip("93.184.216.99") and not webhooks.is_public_ip("::ffff:93.184.216.98")
    assert webhooks.is_public_ip("93.184.216.100")
    webhooks._own.update(at=-1e9, addrs=frozenset())  # do not leave the fake list for other tests


# ---------------------------------------------------------------- resolving and one attempt

def test_every_resolved_address_must_be_public_and_a_failing_lookup_is_a_refusal(net, monkeypatch):
    net.dns["mixed.example.com"] = [PUBLIC, "10.0.0.8"]  # a rebinding answer: one good, one private
    with pytest.raises(webhooks.Refused, match="can't be called"):
        run(webhooks.resolve_public("mixed.example.com"))
    net.dns["mapped.example.com"] = ["::ffff:127.0.0.1"]
    with pytest.raises(webhooks.Refused):
        run(webhooks.resolve_public("mapped.example.com"))
    net.dns["empty.example.com"] = []
    with pytest.raises(webhooks.Refused, match="DNS"):
        run(webhooks.resolve_public("empty.example.com"))
    net.dns["ok.example.com"] = [PUBLIC, "2606:4700:4700::1111"]
    assert run(webhooks.resolve_public("ok.example.com")) == PUBLIC

    async def boom(host):
        raise OSError("no such host")

    monkeypatch.setattr(webhooks, "_getaddrinfo", boom)
    with pytest.raises(webhooks.Refused, match="DNS"):
        run(webhooks.resolve_public("nowhere.example.com"))


def test_a_delivery_goes_to_the_checked_address_with_a_valid_signature(net):
    body = b'{"jobId":"x"}'
    status, error = run(webhooks.attempt(URL, body, "whsec_test"))
    assert (status, error) == (200, None)
    assert net.connects == [(PUBLIC, "hooks.example.com")]  # connected to the address that was checked, TLS name from the URL
    assert net.dns_calls == ["hooks.example.com"]  # looked up once: nothing resolves again between the check and the connect
    head, _, got_body = net.writers[0].sent.partition(b"\r\n\r\n")
    assert got_body == body and net.writers[0].closed
    lines = head.decode().split("\r\n")
    assert lines[0] == "POST /acw?token=secret-token HTTP/1.1"
    headers = dict(l.split(": ", 1) for l in lines[1:])
    assert headers["Host"] == "hooks.example.com" and headers["Content-Type"] == "application/json"
    assert headers["Content-Length"] == str(len(body)) and headers["Connection"] == "close"
    t, v1 = re.fullmatch(r"t=(\d+),v1=([0-9a-f]{64})", headers["X-Audio-Cleaner-Signature"]).groups()
    assert abs(int(t) - time.time()) < 5
    assert hmac.compare_digest(v1, hmac.new(b"whsec_test", t.encode() + b"." + body, hashlib.sha256).hexdigest())


@pytest.mark.parametrize("answer,expected_status,words", [
    (b"HTTP/1.1 301 Moved\r\nLocation: http://169.254.169.254/\r\n\r\n", 301, "redirects are not followed"),
    (b"HTTP/1.1 500 Oops\r\n\r\n", 500, "HTTP 500"),
    (b"HTTP/1.1 204 No Content\r\n\r\n", 204, None),
    (b"", None, "could not connect"),
])
def test_only_a_2xx_answer_counts_and_redirects_are_never_followed(net, answer, expected_status, words):
    net.answer = answer
    status, error = run(webhooks.attempt(URL, b"{}", "whsec_x"))
    assert status == expected_status
    assert (error is None) if words is None else (words in error)
    assert len(net.connects) == 1  # no second request to wherever the redirect pointed


def test_an_endless_answer_is_cut_off_and_a_slow_receiver_times_out(net, monkeypatch):
    net.answer = b"HTTP/1.1 200 " + b"x" * 100_000 + b"\r\n"
    status, error = run(webhooks.attempt(URL, b"{}", "whsec_x"))
    assert status is None and error
    net.answer = b"HTTP/1.1 200 OK\r\n\r\n"
    monkeypatch.setattr(webhooks, "ATTEMPT_SECONDS", 0.1)
    net.delay = 2
    started = time.monotonic()
    assert run(webhooks.attempt(URL, b"{}", "s")) == (None, "timed out")
    assert time.monotonic() - started < 1


def test_a_private_address_is_refused_before_any_connection(net):
    for target in ("10.1.2.3", "127.0.0.1", "169.254.169.254", "::1", "198.51.100.77"):
        net.dns["evil.example.com"] = [target]
        status, error = run(webhooks.attempt("https://evil.example.com/hook", b"{}", "s"))
        assert status is None and error.startswith("refused:"), target
    assert net.connects == []  # not even a connection attempt


def test_connection_trouble_and_bad_certificates_are_reported_in_plain_words(net):
    net.error = ConnectionRefusedError()
    assert run(webhooks.attempt(URL, b"{}", "s")) == (None, "could not connect or read the answer")
    net.error = ssl.SSLCertVerificationError("bad cert")
    assert run(webhooks.attempt(URL, b"{}", "s"))[1].startswith("TLS error")


# ---------------------------------------------------------------- jobs with a callback

def credit(uid):
    ledger.credit(uid, 5_000_000, "grant", "t", "g")


def job_body(**over):
    return {"kind": "denoise", "denoiseTier": "standard", "filename": "talk.wav", "sizeBytes": 1000, "durationSeconds": 600, "options": {}, **over}


def create(client, h, url=URL, **over):
    r = client.post("/api/jobs", json=job_body(callbackUrl=url, **over), headers=h)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def finish(job_id, status="done"):
    job = dict(db.one("SELECT * FROM jobs WHERE id = ?", (job_id,)))
    if status == "done":
        cost = ledger.capture_hold(job["hold_id"], 90_000)
        db.execute(
            "UPDATE jobs SET status = 'done', cost_micros = ?, finished_at = ?, expires_at = ?, outputs_json = ? WHERE id = ?",
            (cost, db.now(), db.now() + 86400,
             json.dumps([{"file": "talk-cleaned.wav", "kind": "audio", "bytes": 4321}, {"file": "talk.es.srt", "kind": "captions", "bytes": 80, "lang": "es"}]), job_id))
    elif status == "failed":
        assert jobs.finish_failed(job, "Processing failed; you were not charged.")
    else:
        assert jobs.finish_failed(job, "Canceled.", status="canceled")


def callback_of(client, job_id):
    return client.get(f"/api/jobs/{job_id}").json().get("callback")


def sent_json(net, n=-1):
    return json.loads(net.writers[n].sent.partition(b"\r\n\r\n")[2])


def test_a_bad_callback_address_stops_the_job_from_being_made(client, signed_in):
    uid, h = signed_in
    credit(uid)
    for bad in ("http://hooks.example.com/x", "https://127.0.0.1/x", "https://hooks.example.com:8443/x", "https://localhost/x", "x"):
        r = client.post("/api/jobs", json=job_body(callbackUrl=bad), headers=h)
        assert r.status_code == 400, bad
    assert db.one("SELECT COUNT(*) n FROM jobs")["n"] == 0 and ledger.balance(uid)["held_micros"] == 0


def test_a_finished_job_is_delivered_once_signed_with_the_account_secret(client, signed_in, net):
    uid, h = signed_in
    credit(uid)
    job_id = create(client, h)
    assert callback_of(client, job_id) == {
        "url": "https://hooks.example.com/acw", "state": "pending", "attempts": 0, "lastStatus": None,  # the token after ? is not shown back
        "lastError": None, "deliveredAt": None, "nextAttemptAt": None}
    assert run(webhooks.deliver_due()) == 0  # the job is still running: nothing to say yet
    finish(job_id)
    assert run(webhooks.deliver_due()) == 1
    cb = callback_of(client, job_id)
    assert cb["state"] == "delivered" and cb["attempts"] == 1 and cb["lastStatus"] == 200 and cb["lastError"] is None
    assert abs(cb["deliveredAt"] - time.time()) < 5
    row = db.one("SELECT finished_at, expires_at FROM jobs WHERE id = ?", (job_id,))
    assert sent_json(net) == {
        "type": "job.done", "jobId": job_id, "status": "done", "error": None, "costMicros": 90_000,
        "outputs": [{"name": "talk-cleaned.wav", "kind": "audio", "bytes": 4321}, {"name": "talk.es.srt", "kind": "captions", "bytes": 80, "lang": "es"}],
        "finishedAt": row["finished_at"], "expiresAt": row["expires_at"],
    }
    secret = db.one("SELECT webhook_secret s FROM users WHERE id = ?", (uid,))["s"]
    assert secret.startswith("whsec_")
    sig = re.search(rb"X-Audio-Cleaner-Signature: t=(\d+),v1=([0-9a-f]+)", net.writers[0].sent)
    body = net.writers[0].sent.partition(b"\r\n\r\n")[2]
    assert sig.group(2).decode() == hmac.new(secret.encode(), sig.group(1) + b"." + body, hashlib.sha256).hexdigest()
    assert run(webhooks.deliver_due()) == 0  # once only
    assert len(net.connects) == 1
    assert client.get("/api/jobs").json()["jobs"][0]["callback"]["state"] == "delivered"


def test_failed_and_canceled_jobs_are_delivered_too(client, signed_in, net):
    uid, h = signed_in
    credit(uid)
    a, b = create(client, h), create(client, h)
    finish(a, "failed")
    assert client.post(f"/api/jobs/{b}/cancel", headers=h).status_code == 200
    assert run(webhooks.deliver_due()) == 2
    got = {sent_json(net, i)["jobId"]: sent_json(net, i) for i in range(2)}
    assert got[a]["type"] == "job.failed" and got[a]["error"] == "Processing failed; you were not charged." and got[a]["costMicros"] == 0
    assert got[b]["type"] == "job.canceled" and got[b]["status"] == "canceled" and got[b]["outputs"] == []


def test_failed_deliveries_are_retried_three_times_with_backoff_then_given_up(client, signed_in, net):
    uid, h = signed_in
    credit(uid)
    job_id = create(client, h)
    finish(job_id)
    net.answer = b"HTTP/1.1 503 Busy\r\n\r\n"
    assert run(webhooks.deliver_due()) == 1
    cb = callback_of(client, job_id)
    assert cb["state"] == "pending" and cb["attempts"] == 1 and cb["lastStatus"] == 503 and cb["lastError"] == "HTTP 503"
    assert 25 <= cb["nextAttemptAt"] - time.time() <= 35  # about 30 s later
    assert run(webhooks.deliver_due()) == 0  # not before then
    db.execute("UPDATE job_callbacks SET next_attempt_at = 0")
    net.error = ConnectionResetError()
    assert run(webhooks.deliver_due()) == 1
    cb = callback_of(client, job_id)
    assert cb["state"] == "pending" and cb["attempts"] == 2 and 290 <= cb["nextAttemptAt"] - time.time() <= 310  # about 5 min
    db.execute("UPDATE job_callbacks SET next_attempt_at = 0")
    assert run(webhooks.deliver_due()) == 1
    cb = callback_of(client, job_id)
    assert cb["state"] == "failed" and cb["attempts"] == 3 and cb["nextAttemptAt"] is None and cb["lastError"]
    db.execute("UPDATE job_callbacks SET next_attempt_at = 0")
    assert run(webhooks.deliver_due()) == 0  # never a fourth
    assert len(net.connects) == 3


def test_an_address_that_resolves_to_a_private_place_is_given_up_at_once(client, signed_in, net):
    uid, h = signed_in
    credit(uid)
    job_id = create(client, h)
    finish(job_id)
    net.dns["hooks.example.com"] = ["192.168.1.10"]
    run(webhooks.deliver_due())
    cb = callback_of(client, job_id)
    assert cb["state"] == "failed" and cb["attempts"] == 1 and cb["lastError"].startswith("refused:") and "public internet" in cb["lastError"]
    assert net.connects == []


def test_many_deliveries_run_side_by_side_at_most_four_at_once_without_blocking_the_loop(client, signed_in, net, monkeypatch):
    assert webhooks.CONCURRENCY == 4
    uid, h = signed_in
    credit(uid)
    seen = {"now": 0, "max": 0}

    async def slow_open(ip, host):
        seen["now"] += 1
        seen["max"] = max(seen["max"], seen["now"])
        await asyncio.sleep(0.05)
        seen["now"] -= 1
        return await net.open(ip, host)

    monkeypatch.setattr(webhooks, "_open", slow_open)
    for _ in range(7):
        job_id = db.new_id()
        db.execute("INSERT INTO jobs(id,user_id,kind,tier,status,params_json,input_name,input_bytes,claimed_seconds,created_at,finished_at)"
                   " VALUES (?,?,'denoise','standard','failed','{}','a',1,1,?,?)", (job_id, uid, db.now(), db.now()))
        db.execute("INSERT INTO job_callbacks(job_id,user_id,url,created_at) VALUES (?,?,?,?)", (job_id, uid, URL, db.now()))

    async def main():
        ticks = 0

        async def beat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(beat())
        tried = await webhooks.deliver_due(limit=20)
        t.cancel()
        return tried, ticks

    tried, ticks = run(main())
    assert tried == 7 and seen["max"] == 4
    assert ticks > 3  # the loop kept running while they waited (timing-based: kept loose)


def test_a_callback_is_claimed_by_one_process_and_a_dead_claim_is_retried(client, signed_in, net):
    uid, h = signed_in
    credit(uid)
    job_id = create(client, h)
    finish(job_id)
    first = webhooks._claim_due(10)
    assert len(first) == 1 and webhooks._claim_due(10) == []  # the other caller finds nothing until the claim runs out
    db.execute("UPDATE job_callbacks SET next_attempt_at = ?", (db.now() - 1,))  # then the process dies without reporting back
    assert run(webhooks.deliver_due()) == 1
    assert callback_of(client, job_id)["state"] == "delivered"


def test_a_suspended_account_gets_no_callbacks(client, signed_in, net):
    uid, h = signed_in
    credit(uid)
    finish(create(client, h))
    db.execute("UPDATE users SET status = 'suspended' WHERE id = ?", (uid,))
    assert run(webhooks.deliver_due()) == 0 and net.connects == []


def test_old_results_are_pruned_and_endless_waits_given_up():
    uid = "u"
    now = db.now()
    for jid, state, age, attempted in (("old-ok", "delivered", 8 * 86400, 8 * 86400), ("new-ok", "delivered", 3600, 3600),
                                       ("stuck", "pending", 3 * 86400, None), ("fresh", "pending", 60, None)):
        db.execute("INSERT INTO job_callbacks(job_id,user_id,url,state,created_at,last_attempt_at) VALUES (?,?,?,?,?,?)",
                   (jid, uid, URL, state, now - age, None if attempted is None else now - attempted))
    webhooks.prune(now)
    left = {r["job_id"]: r["state"] for r in db.all_rows("SELECT job_id, state FROM job_callbacks")}
    assert left == {"new-ok": "delivered", "stuck": "failed", "fresh": "pending"}


# ---------------------------------------------------------------- the secret

def test_the_secret_is_made_shown_and_rotated_and_a_rotation_signs_the_waiting_callbacks(client, signed_in, net):
    uid, h = signed_in
    assert client.get("/api/keys/webhook-secret").json() == {"secret": None}
    first = client.post("/api/keys/webhook-secret", headers=h).json()["secret"]
    assert first.startswith("whsec_") and len(first) > 30
    assert client.get("/api/keys/webhook-secret").json() == {"secret": first}
    credit(uid)
    job_id = create(client, h)
    finish(job_id)
    second = client.post("/api/keys/webhook-secret", headers=h).json()["secret"]
    assert second != first and client.get("/api/keys/webhook-secret").json() == {"secret": second}
    assert "webhook_secret_rotated" in [r["action"] for r in db.all_rows("SELECT action FROM audit_log")]
    run(webhooks.deliver_due())  # the callback waiting since before the rotation is signed with the new secret
    sig = re.search(rb"v1=([0-9a-f]+)", net.writers[0].sent).group(1).decode()
    stamp = re.search(rb"t=(\d+)", net.writers[0].sent).group(1)
    body = net.writers[0].sent.partition(b"\r\n\r\n")[2]
    assert sig == hmac.new(second.encode(), stamp + b"." + body, hashlib.sha256).hexdigest()


def test_an_api_key_can_neither_read_nor_rotate_the_secret(client, signed_in):
    from .test_api_security import make_key

    uid, h = signed_in
    client.post("/api/keys/webhook-secret", headers=h)
    key = make_key(uid, "jobs,stream")
    client.cookies.clear()
    kh = {"Authorization": f"Bearer {key}"}
    assert client.get("/api/keys/webhook-secret", headers=kh).status_code == 403
    assert client.post("/api/keys/webhook-secret", headers=kh).status_code == 403


def test_the_secret_is_not_in_the_account_export_and_goes_with_the_account(client, signed_in):
    uid, h = signed_in
    credit(uid)
    job_id = create(client, h)
    client.post(f"/api/jobs/{job_id}/cancel", headers=h)
    secret = client.get("/api/keys/webhook-secret").json()["secret"]
    export = client.get("/api/account/export")
    assert export.status_code == 200 and secret not in export.text and export.json()["user"]["webhook_secret"] is True
    assert client.post("/api/account/delete", json={"confirm": "user@example.com"}, headers=h).status_code == 200
    assert db.one("SELECT webhook_secret s FROM users WHERE id = ?", (uid,))["s"] is None
    assert db.one("SELECT COUNT(*) n FROM job_callbacks")["n"] == 0
    assert db.one("SELECT COUNT(*) n FROM idempotency_keys")["n"] == 0
