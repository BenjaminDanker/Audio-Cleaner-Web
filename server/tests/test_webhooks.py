"""Job callbacks (callbackUrl): address checks, SSRF refusal, signing, delivery and retries. Nothing here touches the
network: DNS and the connection are replaced (`webhooks._getaddrinfo`, `webhooks._open`)."""
import asyncio
import hashlib
import hmac
import json
import re
import ssl
import time
from pathlib import Path

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

@pytest.mark.parametrize("url,expected", [
    ("https://hooks.example.com/acw", "https://hooks.example.com/acw"),
    ("https://hooks.example.com:443/acw?a=b&c=d", "https://hooks.example.com/acw?a=b&c=d"),
    ("HTTPS://Hooks.Example.COM/a%20b", "https://hooks.example.com/a%20b"),
    ("https://hooks.example.com", "https://hooks.example.com/"),
    ("  https://hooks.example.com./x  ", "https://hooks.example.com/x"),
    ("https://a.b.c.example.org/x/y/z", "https://a.b.c.example.org/x/y/z"),
])
def test_good_addresses_are_accepted_and_normalised(url, expected):
    assert webhooks.check_url(url) == expected


@pytest.mark.parametrize("url,why", [
    ("", "empty"),
    ("http://hooks.example.com/x", "https"),
    ("ftp://hooks.example.com/x", "https"),
    ("//hooks.example.com/x", "https"),
    ("hooks.example.com/x", "https"),
    ("https://hooks.example.com:8443/x", "443"),
    ("https://hooks.example.com:80/x", "443"),
    ("https://hooks.example.com:abc/x", "valid"),
    ("https://user:pw@hooks.example.com/x", "user name"),
    ("https://user@hooks.example.com/x", "user name"),
    ("https://hooks.example.com/x#frag", "fragment"),
    ("https://127.0.0.1/x", "IP address"),
    ("https://10.0.0.5/x", "IP address"),
    ("https://[::1]/x", "IP address"),
    ("https://[2606:4700::1111]/x", "IP address"),
    ("https://2130706433/x", "public host name"),   # decimal 127.0.0.1: not a name with a dot
    ("https://localhost/x", "public host name"),
    ("https://printer.local/x", "public host name"),
    ("https://db.internal/x", "public host name"),
    ("https://intranet/x", "public host name"),
    ("https://exa mple.com/x", "ASCII"),
    ("https://hooks.exämple.com/x", "ASCII"),
    ("https://-bad-.example.com/x", "public host name"),
    ("https://hooks.example.com/" + "a" * 600, "at most"),
    ("https://testserver/x", "public host name"),
])
def test_bad_addresses_are_refused(url, why):
    with pytest.raises(ValueError) as e:
        webhooks.check_url(url)
    assert why in str(e.value)


def test_the_service_itself_is_not_a_target(monkeypatch):
    from acw.config import get_settings

    monkeypatch.setenv("ACW_PUBLIC_ORIGIN", "https://audio.example.net")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="back at this service"):
        webhooks.check_url("https://audio.example.net/api/jobs")
    assert webhooks.check_url("https://other.example.net/x")


@pytest.mark.parametrize("ip", [
    "93.184.216.34", "8.8.8.8", "1.1.1.1", "2606:4700:4700::1111", "2a00:1450:4009:81f::200e",
])
def test_public_addresses_pass(ip):
    assert webhooks.is_public_ip(ip)


@pytest.mark.parametrize("ip", [
    "0.0.0.0", "0.1.2.3", "10.0.0.1", "10.255.255.255", "100.64.0.1", "100.127.255.254",  # CGNAT
    "127.0.0.1", "127.1.2.3", "169.254.169.254", "172.16.0.1", "172.31.255.255", "192.0.0.1", "192.0.2.1",
    "192.168.1.1", "198.18.0.1", "198.51.100.1", "203.0.113.1", "224.0.0.1", "239.255.255.250", "240.0.0.1",
    "255.255.255.255",
    "::", "::1", "::2", "fe80::1", "fe80::1%eth0", "fc00::1", "fd12:3456::1", "ff02::1", "2001:db8::1",
    "::ffff:127.0.0.1", "::ffff:10.0.0.1", "::ffff:169.254.169.254", "::ffff:192.168.0.1",  # IPv4 inside IPv6
    "64:ff9b::7f00:1", "2002:7f00:1::1", "2001::1", "100::1", "fec0::1",
    "198.51.100.77",  # this machine's own address (the fixture's)
    "not an address", "",
])
def test_everything_that_is_not_the_public_internet_is_refused(ip):
    assert not webhooks.is_public_ip(ip)


def test_extra_networks_can_be_denied_by_the_owner(monkeypatch):
    monkeypatch.setenv("ACW_WEBHOOK_DENY", "93.184.216.0/24, garbage, 2606:4700::/32")
    assert not webhooks.is_public_ip("93.184.216.34")
    assert not webhooks.is_public_ip("2606:4700:4700::1111")
    assert webhooks.is_public_ip("8.8.8.8")


def test_the_machines_own_addresses_are_found_and_refused(monkeypatch):
    monkeypatch.setattr(webhooks, "own_addresses", REAL_OWN_ADDRESSES)
    monkeypatch.setattr(webhooks.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.99", 0)), (10, 1, 6, "", ("::ffff:93.184.216.98", 0, 0, 0))])
    webhooks._own.update(at=-1e9)
    found = webhooks.own_addresses()
    assert isinstance(found, frozenset) and {"93.184.216.99", "93.184.216.98"} <= found
    assert not webhooks.is_public_ip("93.184.216.99") and not webhooks.is_public_ip("::ffff:93.184.216.98")
    assert webhooks.is_public_ip("93.184.216.100")
    webhooks._own.update(at=-1e9, addrs=frozenset())  # do not leave the fake list for other tests


# ---------------------------------------------------------------- resolving

def test_every_resolved_address_must_be_public(net):
    net.dns["mixed.example.com"] = [PUBLIC, "10.0.0.8"]  # a rebinding answer: one good, one private
    with pytest.raises(webhooks.Refused, match="can't be called"):
        run(webhooks.resolve_public("mixed.example.com"))
    net.dns["v6.example.com"] = ["fd00::1"]
    with pytest.raises(webhooks.Refused):
        run(webhooks.resolve_public("v6.example.com"))
    net.dns["mapped.example.com"] = ["::ffff:127.0.0.1"]
    with pytest.raises(webhooks.Refused):
        run(webhooks.resolve_public("mapped.example.com"))
    net.dns["empty.example.com"] = []
    with pytest.raises(webhooks.Refused, match="DNS"):
        run(webhooks.resolve_public("empty.example.com"))
    net.dns["ok.example.com"] = [PUBLIC, "2606:4700:4700::1111"]
    assert run(webhooks.resolve_public("ok.example.com")) == PUBLIC


def test_a_failing_lookup_is_a_refusal_not_a_crash(monkeypatch):
    async def boom(host):
        raise OSError("no such host")

    monkeypatch.setattr(webhooks, "_getaddrinfo", boom)
    with pytest.raises(webhooks.Refused, match="DNS"):
        run(webhooks.resolve_public("nowhere.example.com"))


# ---------------------------------------------------------------- one attempt

def test_signature_is_hmac_sha256_over_timestamp_dot_body():
    sig = webhooks.sign("whsec_abc", 1700000000, b'{"a":1}')
    expect = hmac.new(b"whsec_abc", b'1700000000.{"a":1}', hashlib.sha256).hexdigest()
    assert sig == f"t=1700000000,v1={expect}"


def test_a_delivery_goes_to_the_resolved_address_with_a_valid_signature(net):
    body = b'{"jobId":"x"}'
    status, error = run(webhooks.attempt(URL, body, "whsec_test"))
    assert (status, error) == (200, None)
    assert net.connects == [(PUBLIC, "hooks.example.com")]  # connected to the address that was checked, TLS name from the URL
    assert net.dns_calls == ["hooks.example.com"]  # looked up once: nothing resolves again between the check and the connect
    sent = net.writers[0].sent
    head, _, got_body = sent.partition(b"\r\n\r\n")
    assert got_body == body and net.writers[0].closed
    lines = head.decode().split("\r\n")
    assert lines[0] == "POST /acw?token=secret-token HTTP/1.1"
    headers = dict(l.split(": ", 1) for l in lines[1:])
    assert headers["Host"] == "hooks.example.com" and headers["Content-Type"] == "application/json"
    assert headers["Content-Length"] == str(len(body)) and headers["Connection"] == "close"
    assert headers["User-Agent"].startswith("AudioCleaner-Webhook")
    t, v1 = re.fullmatch(r"t=(\d+),v1=([0-9a-f]{64})", headers["X-Audio-Cleaner-Signature"]).groups()
    assert abs(int(t) - time.time()) < 5
    assert hmac.compare_digest(v1, hmac.new(b"whsec_test", t.encode() + b"." + body, hashlib.sha256).hexdigest())


@pytest.mark.parametrize("answer,expected_status,words", [
    (b"HTTP/1.1 301 Moved\r\nLocation: http://169.254.169.254/\r\n\r\n", 301, "redirects are not followed"),
    (b"HTTP/1.1 302 Found\r\nLocation: https://elsewhere.example.com/\r\n\r\n", 302, "redirects"),
    (b"HTTP/1.1 500 Oops\r\n\r\n", 500, "HTTP 500"),
    (b"HTTP/1.0 404 Nope\r\n\r\n", 404, "HTTP 404"),
    (b"HTTP/1.1 204 No Content\r\n\r\n", 204, None),
    (b"garbage\r\n", None, "could not connect"),
    (b"", None, "could not connect"),
])
def test_only_a_2xx_answer_counts_and_redirects_are_never_followed(net, answer, expected_status, words):
    net.answer = answer
    status, error = run(webhooks.attempt(URL, b"{}", "whsec_x"))
    assert status == expected_status
    assert (error is None) if words is None else (words in error)
    assert len(net.connects) == 1  # no second request to wherever the redirect pointed


def test_an_endless_answer_is_cut_off(net):
    net.answer = b"HTTP/1.1 200 " + b"x" * 100_000 + b"\r\n"
    status, error = run(webhooks.attempt(URL, b"{}", "whsec_x"))
    assert status is None and error


def test_connection_trouble_is_reported_in_plain_words(net):
    net.error = ConnectionRefusedError()
    assert run(webhooks.attempt(URL, b"{}", "s")) == (None, "could not connect or read the answer")
    net.error = ssl.SSLCertVerificationError("bad cert")
    assert run(webhooks.attempt(URL, b"{}", "s"))[1].startswith("TLS error")


def test_a_slow_receiver_times_out_after_five_seconds(net, monkeypatch):
    assert webhooks.ATTEMPT_SECONDS == 5.0
    monkeypatch.setattr(webhooks, "ATTEMPT_SECONDS", 0.1)
    net.delay = 2
    started = time.monotonic()
    assert run(webhooks.attempt(URL, b"{}", "s")) == (None, "timed out")
    assert time.monotonic() - started < 1


def test_a_private_address_is_refused_before_any_connection(net):
    for target in ("10.1.2.3", "127.0.0.1", "169.254.169.254", "100.64.9.9", "::1", "fe80::5", "198.51.100.77", "224.0.0.1"):
        net.dns["evil.example.com"] = [target]
        status, error = run(webhooks.attempt("https://evil.example.com/hook", b"{}", "s"))
        assert status is None and error.startswith("refused:"), target
    assert net.connects == []  # not even a connection attempt


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


def test_a_finished_job_is_delivered_and_the_result_shows_with_the_job(client, signed_in, net):
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
    payload = sent_json(net)
    row = db.one("SELECT finished_at, expires_at FROM jobs WHERE id = ?", (job_id,))
    assert payload == {
        "type": "job.done", "jobId": job_id, "status": "done", "error": None, "costMicros": 90_000,
        "outputs": [{"name": "talk-cleaned.wav", "kind": "audio", "bytes": 4321}, {"name": "talk.es.srt", "kind": "captions", "bytes": 80, "lang": "es"}],
        "finishedAt": row["finished_at"], "expiresAt": row["expires_at"],
    }
    assert len(net.writers[0].sent) < 2000  # small
    # the signature uses the account's secret, the one the card shows
    secret = db.one("SELECT webhook_secret s FROM users WHERE id = ?", (uid,))["s"]
    assert secret.startswith("whsec_")
    sig = re.search(rb"X-Audio-Cleaner-Signature: t=(\d+),v1=([0-9a-f]+)", net.writers[0].sent)
    body = net.writers[0].sent.partition(b"\r\n\r\n")[2]
    assert sig.group(2).decode() == hmac.new(secret.encode(), sig.group(1) + b"." + body, hashlib.sha256).hexdigest()
    assert run(webhooks.deliver_due()) == 0  # once only
    assert len(net.connects) == 1
    listed = client.get("/api/jobs").json()["jobs"]
    assert listed[0]["callback"]["state"] == "delivered"


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


def test_a_later_success_after_a_failure_is_recorded(client, signed_in, net):
    uid, h = signed_in
    credit(uid)
    job_id = create(client, h)
    finish(job_id)
    net.answer = b"HTTP/1.1 500 x\r\n\r\n"
    run(webhooks.deliver_due())
    net.answer = b"HTTP/1.1 200 OK\r\n\r\n"
    db.execute("UPDATE job_callbacks SET next_attempt_at = 0")
    run(webhooks.deliver_due())
    cb = callback_of(client, job_id)
    assert cb["state"] == "delivered" and cb["attempts"] == 2 and cb["lastError"] is None and cb["lastStatus"] == 200


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


def test_deliveries_run_side_by_side_and_never_block_the_event_loop(client, signed_in, net):
    uid, h = signed_in
    credit(uid)
    ids = [create(client, h) for _ in range(3)]
    for i in ids:
        finish(i)
    net.delay = 0.3

    async def main():
        ticks = 0

        async def beat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(beat())
        started = time.monotonic()
        tried = await webhooks.deliver_due()
        took = time.monotonic() - started
        t.cancel()
        return tried, took, ticks

    tried, took, ticks = run(main())
    assert tried == 3 and took < 0.8  # three at once, not one after the other
    assert ticks > 15  # the loop kept running while they waited


def test_at_most_four_deliveries_at_once(client, signed_in, net, monkeypatch):
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
    for i in range(7):
        job_id = db.new_id()
        db.execute("INSERT INTO jobs(id,user_id,kind,tier,status,params_json,input_name,input_bytes,claimed_seconds,created_at,finished_at)"
                   " VALUES (?,?,'denoise','standard','failed','{}','a',1,1,?,?)", (job_id, uid, db.now(), db.now()))
        db.execute("INSERT INTO job_callbacks(job_id,user_id,url,created_at) VALUES (?,?,?,?)", (job_id, uid, URL, db.now()))
    assert run(webhooks.deliver_due(limit=20)) == 7
    assert seen["max"] == 4


def test_two_processes_cannot_deliver_the_same_callback(client, signed_in, net):
    uid, h = signed_in
    credit(uid)
    finish(create(client, h))
    first = webhooks._claim_due(10)
    second = webhooks._claim_due(10)
    assert len(first) == 1 and second == []  # claimed: the other caller finds nothing until the claim runs out


def test_a_claim_that_never_reported_back_is_tried_again(client, signed_in, net):
    uid, h = signed_in
    credit(uid)
    job_id = create(client, h)
    finish(job_id)
    assert len(webhooks._claim_due(10)) == 1  # then the process dies
    db.execute("UPDATE job_callbacks SET next_attempt_at = ?", (db.now() - 1,))
    assert run(webhooks.deliver_due()) == 1
    assert callback_of(client, job_id)["state"] == "delivered"


def test_a_suspended_account_gets_no_callbacks(client, signed_in, net):
    uid, h = signed_in
    credit(uid)
    finish(create(client, h))
    db.execute("UPDATE users SET status = 'suspended' WHERE id = ?", (uid,))
    assert run(webhooks.deliver_due()) == 0 and net.connects == []


def test_the_background_loop_delivers(client, signed_in, net, monkeypatch):
    uid, h = signed_in
    credit(uid)
    job_id = create(client, h)
    finish(job_id)
    monkeypatch.setattr(webhooks, "POLL_SECONDS", 0.05)
    # the app's loop sleeps with the old value for its first round; start a fresh one on its own event loop
    import threading

    stop = threading.Event()

    def loop_thread():
        async def main():
            t = asyncio.create_task(webhooks.run_loop())
            while not stop.is_set():
                await asyncio.sleep(0.05)
            t.cancel()

        asyncio.run(main())

    th = threading.Thread(target=loop_thread, daemon=True)
    th.start()
    try:
        for _ in range(60):
            if callback_of(client, job_id)["state"] == "delivered":
                break
            time.sleep(0.05)
    finally:
        stop.set()
        th.join(5)
    assert callback_of(client, job_id)["state"] == "delivered"


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


def test_only_the_web_process_delivers():
    root = Path(__file__).resolve().parents[1] / "acw"
    assert "webhooks.run_loop" in (root / "app.py").read_text()
    for other in ("bulk_app.py", "worker/main.py", "worker/runner.py"):
        assert "webhooks" not in (root / other).read_text(), other


# ---------------------------------------------------------------- the secret

def test_the_secret_is_made_shown_and_rotated_in_the_browser(client, signed_in):
    uid, h = signed_in
    assert client.get("/api/keys/webhook-secret").json() == {"secret": None}
    first = client.post("/api/keys/webhook-secret", headers=h).json()["secret"]
    assert first.startswith("whsec_") and len(first) > 30
    assert client.get("/api/keys/webhook-secret").json() == {"secret": first}
    second = client.post("/api/keys/webhook-secret", headers=h).json()["secret"]
    assert second != first and client.get("/api/keys/webhook-secret").json() == {"secret": second}
    assert "webhook_secret_rotated" in [r["action"] for r in db.all_rows("SELECT action FROM audit_log")]


def test_an_api_key_can_neither_read_nor_rotate_the_secret(client, signed_in):
    from .test_api_security import make_key

    uid, h = signed_in
    client.post("/api/keys/webhook-secret", headers=h)
    key = make_key(uid, "jobs,stream")
    client.cookies.clear()
    kh = {"Authorization": f"Bearer {key}"}
    assert client.get("/api/keys/webhook-secret", headers=kh).status_code == 403
    assert client.post("/api/keys/webhook-secret", headers=kh).status_code == 403


def test_using_a_callback_makes_the_secret_so_it_can_be_seen_and_checked(client, signed_in):
    uid, h = signed_in
    credit(uid)
    assert client.get("/api/keys/webhook-secret").json()["secret"] is None
    create(client, h)
    assert client.get("/api/keys/webhook-secret").json()["secret"].startswith("whsec_")


def test_a_rotated_secret_signs_the_waiting_callbacks(client, signed_in, net):
    uid, h = signed_in
    credit(uid)
    job_id = create(client, h)
    finish(job_id)
    new = client.post("/api/keys/webhook-secret", headers=h).json()["secret"]
    run(webhooks.deliver_due())
    sig = re.search(rb"v1=([0-9a-f]+)", net.writers[0].sent).group(1).decode()
    stamp = re.search(rb"t=(\d+)", net.writers[0].sent).group(1)
    body = net.writers[0].sent.partition(b"\r\n\r\n")[2]
    assert sig == hmac.new(new.encode(), stamp + b"." + body, hashlib.sha256).hexdigest()


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
