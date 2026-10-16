"""The public file API: Idempotency-Key, price estimates, and monthly spending caps per API key."""
import json
import time
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocketDisconnect

from acw import db, jobs, keycaps
from acw.billing import ledger
from acw.security import sha256_hex

from .conftest import login
from .test_api_security import make_key
from .test_live_trial import fake_worker  # noqa: F401 - a fixture: a fake live worker behind the real endpoint


def body(**over):
    return {"kind": "denoise", "denoiseTier": "standard", "filename": "talk.wav", "sizeBytes": 1000, "durationSeconds": 600,
            "options": {}, **over}


def key_headers(user_id, scopes="jobs", **cols):
    key = make_key(user_id, scopes)
    for col, value in cols.items():
        db.execute(f"UPDATE api_keys SET {col} = ? WHERE key_hash = ?", (value, sha256_hex(key)))
    return {"Authorization": f"Bearer {key}"}


def count(table, where="1=1", args=()):
    return db.one(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", args)["n"]


# ---------------------------------------------------------------- Idempotency-Key

def test_the_same_key_gives_the_same_job_and_holds_money_once(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    first = client.post("/api/jobs", json=body(), headers={**h, "Idempotency-Key": "order-1"})
    again = client.post("/api/jobs", json=body(), headers={**h, "Idempotency-Key": "order-1"})
    assert first.status_code == again.status_code == 200, (first.text, again.text)
    assert again.json()["id"] == first.json()["id"]
    assert again.json()["estCostMicros"] == first.json()["estCostMicros"] > 0
    assert again.json()["uploadUrl"] == first.json()["uploadUrl"]
    assert again.json()["idempotentReplay"] is True and again.headers["Idempotent-Replayed"] == "true"
    assert "idempotentReplay" not in first.json() and "Idempotent-Replayed" not in first.headers
    assert count("jobs") == 1 and count("holds") == 1
    assert ledger.balance(uid)["held_micros"] == first.json()["estCostMicros"]


def test_without_a_key_every_request_is_a_new_job(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    a = client.post("/api/jobs", json=body(), headers=h)
    b = client.post("/api/jobs", json=body(), headers=h)
    assert a.json()["id"] != b.json()["id"] and count("jobs") == 2


def test_the_same_key_with_other_details_is_a_conflict(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    hk = {**h, "Idempotency-Key": "k"}
    assert client.post("/api/jobs", json=body(), headers=hk).status_code == 200
    r = client.post("/api/jobs", json=body(durationSeconds=601), headers=hk)
    assert r.status_code == 409 and "Idempotency-Key" in r.json()["detail"]
    assert client.post("/api/jobs", json=body(options={"noise": "heavy"}), headers=hk).status_code == 409
    assert count("jobs") == 1


def test_a_failed_attempt_does_not_use_up_the_key(client, signed_in):
    uid, h = signed_in
    hk = {**h, "Idempotency-Key": "retry-me"}
    assert client.post("/api/jobs", json=body(), headers=hk).status_code == 402  # no credit yet
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    ok = client.post("/api/jobs", json=body(), headers=hk)
    assert ok.status_code == 200 and "idempotentReplay" not in ok.json()


def test_keys_are_per_account(client):
    a, ha = login(client, "a@example.com")
    ledger.credit(a, 5_000_000, "grant", "t", "g")
    first = client.post("/api/jobs", json=body(), headers={**ha, "Idempotency-Key": "same"}).json()["id"]
    client.cookies.clear()
    b, hb = login(client, "b@example.com")
    ledger.credit(b, 5_000_000, "grant", "t", "g-b")
    second = client.post("/api/jobs", json=body(), headers={**hb, "Idempotency-Key": "same"}).json()["id"]
    assert first != second


def test_a_key_is_forgotten_after_a_day(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    hk = {**h, "Idempotency-Key": "old"}
    first = client.post("/api/jobs", json=body(), headers=hk).json()["id"]
    db.execute("UPDATE idempotency_keys SET created_at = ?", (db.now() - 86400 - 5,))
    assert client.post("/api/jobs", json=body(), headers=hk).json()["id"] != first
    db.execute("UPDATE idempotency_keys SET created_at = ?", (db.now() - 86400 - 5,))
    jobs.sweep()
    assert count("idempotency_keys") == 0


@pytest.mark.parametrize("bad", ["", " ", "a b", "x" * 129, "é"])
def test_a_malformed_key_is_refused(client, signed_in, bad):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    try:
        r = client.post("/api/jobs", json=body(), headers={**h, "Idempotency-Key": bad})
    except UnicodeEncodeError:  # the client library can't even send it
        return
    assert r.status_code == 400 and count("jobs") == 0


def test_a_repeat_costs_no_job_creation_allowance(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    hk = {**h, "Idempotency-Key": "cheap"}
    assert client.post("/api/jobs", json=body(), headers=hk).status_code == 200
    for _ in range(40):  # more than the hourly allowance of 20
        assert client.post("/api/jobs", json=body(), headers=hk).json()["idempotentReplay"] is True


def test_a_parallel_request_with_the_same_key_is_caught_inside_the_transaction(client, signed_in):
    """The early lookup missed (both requests saw no row); the second one then meets the first one's row in the transaction."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    first = jobs.create_job(uid, kind="denoise", denoise_tier="standard", transcribe_tier=None, filename="a.wav", size_bytes=1000,
                            claimed_seconds=60.0, options={}, free=False, idempotency=("race", "h"))
    with pytest.raises(jobs.Replay) as r:
        jobs.create_job(uid, kind="denoise", denoise_tier="standard", transcribe_tier=None, filename="a.wav", size_bytes=1000,
                        claimed_seconds=60.0, options={}, free=False, idempotency=("race", "h"))
    assert r.value.job_id == first["id"] and count("jobs") == 1 and count("holds") == 1


# ---------------------------------------------------------------- estimate

def est(client, h, **over):
    return client.post("/api/jobs/estimate", json={"kind": "denoise", "denoiseTier": "standard", "durationSeconds": 600, "options": {}, **over},
                       headers=h)


def test_estimate_is_the_price_creating_the_job_would_hold(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    cases = [
        dict(),
        dict(denoiseTier="rebuild", durationSeconds=1234.5),
        dict(kind="transcribe", denoiseTier=None, transcribeTier="premium", durationSeconds=95.2),
        dict(kind="denoise_transcribe", transcribeTier="local", options={"translate": ["es", "de"]}),
        dict(kind="transcribe", denoiseTier=None, transcribeTier="local", options={"translate": ["fr"], "language": "en"}),
    ]
    for case in cases:
        e = est(client, h, **case)
        assert e.status_code == 200, e.text
        created = client.post("/api/jobs", json=body(filename="a.wav", **case), headers=h)
        assert created.status_code == 200, created.text
        assert e.json()["estCostMicros"] == created.json()["estCostMicros"]
        assert abs(sum(line["micros"] for line in e.json()["lines"]) - e.json()["estCostMicros"]) == 0
        client.post(f"/api/jobs/{created.json()['id']}/cancel", headers=h)  # keeps the active-jobs limit free


def test_estimate_lists_the_parts_and_creates_nothing(client, signed_in):
    uid, h = signed_in
    e = est(client, h, kind="denoise_transcribe", transcribeTier="local", options={"translate": ["es", "de"]}).json()
    assert [x["item"] for x in e["lines"]] == ["denoise", "transcribe", "translate"]
    assert e["lines"][2]["quantity"] == 2 and e["estCostUsd"] == e["estCostMicros"] / 1e6
    assert e["maxSeconds"] == 4 * 3600 and e["options"]["translate"] == ["es", "de"]
    assert e["availableMicros"] == 0 and e["affordable"] is False
    assert count("jobs") == 0 and count("holds") == 0 and ledger.balance(uid)["held_micros"] == 0


def test_estimate_refuses_what_creating_would_refuse(client, signed_in):
    _, h = signed_in
    assert est(client, h, durationSeconds=0.1).status_code == 400
    assert est(client, h, durationSeconds=5 * 3600).status_code == 400
    r = est(client, h, denoiseTier="rebuild", durationSeconds=100 * 60)
    assert r.status_code == 400 and "90 minutes" in r.json()["detail"]
    assert est(client, h, options={"loudness_lufs": -5}).status_code == 400
    assert est(client, h, denoiseTier="gold").status_code == 400
    assert est(client, h, kind="mystery").status_code == 400
    assert est(client, h, options={"translate": ["es"]}).status_code == 400  # translations need subtitles
    assert est(client, h, durationSeconds=-3).status_code == 422
    assert est(client, h, options={"x": "y" * 3000}).status_code == 400


def test_estimate_needs_the_jobs_scope_and_shows_the_keys_limit(client, signed_in):
    uid, _ = signed_in
    client.cookies.clear()
    stream_only = key_headers(uid, "stream")
    assert est(client, stream_only).status_code == 403
    capped = key_headers(uid, "jobs", monthly_cap_micros=100_000)
    r = est(client, capped).json()
    assert r["keyLimit"]["monthlyCapMicros"] == 100_000 and r["keyLimit"]["remainingMicros"] == 100_000
    assert "keyLimit" not in est(client, key_headers(uid, "jobs")).json()


# ---------------------------------------------------------------- spending caps

def test_a_capped_key_cannot_start_a_job_over_its_cap(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    k = client.post("/api/keys", json={"name": "ci", "scopes": ["jobs"], "monthlyCapCents": 20}, headers=h)  # $0.20
    assert k.status_code == 200 and k.json()["monthlyCapCents"] == 20
    kh = {"Authorization": f"Bearer {k.json()['key']}"}
    client.cookies.clear()
    # 10 minutes of standard cleanup is $0.10
    a = client.post("/api/jobs", json=body(), headers=kh)
    b = client.post("/api/jobs", json=body(), headers=kh)
    assert a.status_code == b.status_code == 200
    c = client.post("/api/jobs", json=body(durationSeconds=60), headers=kh)
    assert c.status_code == 403 and c.headers["X-Error-Code"] == "key_spending_cap"
    assert "monthly spending limit" in c.json()["detail"] and "$0.20" in c.json()["detail"]
    assert count("jobs") == 2 and ledger.balance(uid)["held_micros"] == 200_000  # nothing held for the refused one
    # the account itself (a browser session) is not capped
    login_uid, h2 = login(client)
    assert client.post("/api/jobs", json=body(), headers=h2).status_code in (200, 429)


def test_the_cap_counts_real_cost_of_finished_jobs_and_ignores_failed_and_last_months(client, signed_in):
    uid, _ = signed_in
    kh = key_headers(uid, "jobs", monthly_cap_micros=300_000)
    key_id = db.one("SELECT id FROM api_keys")["id"]
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    start, end = keycaps.month_bounds()

    def add(status, cost, est_cost, finished):
        db.execute(
            "INSERT INTO jobs(id,user_id,kind,tier,free,status,params_json,input_name,input_bytes,claimed_seconds,est_cost_micros,"
            "cost_micros,created_at,finished_at,api_key_id) VALUES (?,?,'denoise','standard',0,?,'{}','a',1,1,?,?,?,?,?)",
            (db.new_id(), uid, status, est_cost, cost, finished or db.now(), finished, key_id))

    add("done", 90_000, 100_000, db.now())             # counts at its real cost
    add("expired", 60_000, 60_000, db.now())           # still counts: it only expired
    add("done", 500_000, 500_000, start - 10)          # last month
    add("failed", None, 400_000, db.now())             # failed: free
    add("canceled", None, 400_000, db.now())
    add("queued", None, 40_000, None)                  # running: its reservation counts
    assert keycaps.usage_for([key_id]) == {key_id: 190_000}
    assert client.post("/api/jobs", json=body(durationSeconds=660), headers=kh).status_code == 200  # $0.11: 190k + 110k = 300k
    r = client.post("/api/jobs", json=body(durationSeconds=6), headers=kh)  # even $0.001 is over now
    assert r.status_code == 403
    # the next month is a fresh allowance
    assert keycaps.month_bounds(end)[0] == end and keycaps.month_bounds(start - 1)[1] == start


def test_months_roll_over_at_the_end_of_the_year():
    from datetime import datetime, timezone

    start, end = keycaps.month_bounds(int(datetime(2026, 12, 15, 12, tzinfo=timezone.utc).timestamp()))
    assert datetime.fromtimestamp(start, timezone.utc) == datetime(2026, 12, 1, tzinfo=timezone.utc)
    assert datetime.fromtimestamp(end, timezone.utc) == datetime(2027, 1, 1, tzinfo=timezone.utc)


def test_the_card_shows_usage_and_the_cap_can_be_changed_or_removed(client, signed_in):
    uid, h = signed_in
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    k = client.post("/api/keys", json={"name": "ci", "scopes": ["jobs"], "monthlyCapCents": 5}, headers=h).json()
    kh = {"Authorization": f"Bearer {k['key']}"}
    listed = client.get("/api/keys").json()["keys"][0]
    assert listed["monthlyCapCents"] == 5 and listed["usedThisMonthMicros"] == 0 and listed["capResetsAt"] > time.time()
    cookies = dict(client.cookies)
    client.cookies.clear()
    assert client.post("/api/jobs", json=body(durationSeconds=300), headers=kh).status_code == 200  # $0.05: exactly the cap
    assert client.post("/api/jobs", json=body(durationSeconds=60), headers=kh).status_code == 403
    client.cookies.update(cookies)
    assert client.get("/api/keys").json()["keys"][0]["usedThisMonthMicros"] == 50_000
    r = client.patch(f"/api/keys/{k['id']}", json={"monthlyCapCents": 100}, headers=h)
    assert r.status_code == 200 and r.json()["monthlyCapCents"] == 100 and r.json()["usedThisMonthMicros"] == 50_000
    client.cookies.clear()
    assert client.post("/api/jobs", json=body(durationSeconds=60), headers=kh).status_code == 200
    client.cookies.update(cookies)
    assert client.patch(f"/api/keys/{k['id']}", json={"monthlyCapCents": None}, headers=h).json()["monthlyCapCents"] is None
    assert client.patch(f"/api/keys/{k['id']}", json={}, headers=h).status_code == 400
    assert client.patch(f"/api/keys/{k['id']}", json={"monthlyCapCents": 0}, headers=h).status_code == 422
    assert client.patch("/api/keys/nope", json={"monthlyCapCents": 5}, headers=h).status_code == 404


def test_a_key_cannot_change_caps_and_other_accounts_keys_are_out_of_reach(client, signed_in):
    uid, h = signed_in
    k = client.post("/api/keys", json={"name": "ci", "scopes": ["jobs"], "monthlyCapCents": 5}, headers=h).json()
    client.cookies.clear()
    kh = {"Authorization": f"Bearer {k['key']}"}
    assert client.patch(f"/api/keys/{k['id']}", json={"monthlyCapCents": None}, headers=kh).status_code == 403
    _, h2 = login(client, "other@example.com")
    assert client.patch(f"/api/keys/{k['id']}", json={"monthlyCapCents": None}, headers=h2).status_code == 404
    assert db.one("SELECT monthly_cap_micros m FROM api_keys")["m"] == 50_000


def test_me_with_a_key_shows_its_limit(client, signed_in):
    uid, _ = signed_in
    kh = key_headers(uid, "jobs", monthly_cap_micros=250_000)
    client.cookies.clear()
    me = client.get("/api/me", headers=kh).json()
    assert me["keyLimit"]["monthlyCapMicros"] == 250_000 and me["keyLimit"]["usedMicros"] == 0
    assert "keyLimit" not in client.get("/api/me", headers=key_headers(uid, "jobs")).json()


# ---- live

def live_session(uid, key_id, rate=4000):
    from acw.stream import routes

    p = SimpleNamespace(user_id=uid, api_key_id=key_id)
    return routes.Session(SimpleNamespace(), p, {"denoise": "light", "captions": None, "echo_reference": False}, rate)


def test_a_live_session_cannot_reserve_past_the_cap():
    from acw import auth

    uid = auth.get_or_create_user("a@example.com")["id"]
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    db.execute("INSERT INTO api_keys(id,user_id,name,prefix,key_hash,scopes,created_at,monthly_cap_micros) VALUES ('k1',?,'k','p','h','stream',?,?)",
               (uid, db.now(), 28_000))  # five minutes (20,000) and two more minutes (8,000)
    sess = live_session(uid, "k1")
    sess.id = "s1"
    sess.open_hold()
    db.execute("INSERT INTO stream_sessions(id,user_id,api_key_id,denoise_tier,status,hold_id,started_at) VALUES ('s1',?,'k1','light','active',?,?)",
               (uid, sess.hold_id, db.now()))
    assert keycaps.usage_for(["k1"]) == {"k1": 20_000}  # the open session counts by what is held for it
    assert sess.extend_hold() and sess.extend_hold()
    assert not sess.capped
    assert not sess.extend_hold() and sess.capped
    assert db.one("SELECT amount_micros a FROM holds WHERE id = ?", (sess.hold_id,))["a"] == 28_000
    # a second session of the same key can't even start
    other = live_session(uid, "k1")
    with pytest.raises(keycaps.KeyCapReached):
        other.open_hold()
    assert ledger.balance(uid)["held_micros"] == 28_000
    # a session of a browser (no key) is never capped
    free = live_session(uid, None)
    free.open_hold()


def test_the_billing_tick_warns_then_stops_a_capped_session():
    import asyncio

    from acw import auth
    from acw.stream import routes

    uid = auth.get_or_create_user("a@example.com")["id"]
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    db.execute("INSERT INTO api_keys(id,user_id,name,prefix,key_hash,scopes,created_at,monthly_cap_micros) VALUES ('k1',?,'k','p','h','stream',?,?)",
               (uid, db.now(), 20_000))
    sent = []

    async def send_text(text):
        sent.append(json.loads(text))

    sess = live_session(uid, "k1")
    sess.ws = SimpleNamespace(send_text=send_text)
    sess.id = "s1"
    sess.open_hold()
    db.execute("INSERT INTO stream_sessions(id,user_id,api_key_id,denoise_tier,status,hold_id,started_at) VALUES ('s1',?,'k1','light','active',?,?)",
               (uid, sess.hold_id, db.now()))
    sess.samples = 48_000 * 60 * 4 + 48_000 * 30  # 4.5 minutes of the 5 held: the next minute is due
    assert asyncio.run(routes._billing_tick(sess)) is True
    assert sent[0]["type"] == "warning" and sent[0]["code"] == "key_spending_cap"
    sess.samples = 48_000 * 60 * 5
    assert asyncio.run(routes._billing_tick(sess)) is False
    assert sent[-1] == {"type": "stopped", "reason": "key_spending_cap"} and sess.end_reason == "key_spending_cap"


def test_starting_live_with_a_capped_key_says_why(client, signed_in, fake_worker):  # noqa: F811
    uid, _ = signed_in
    ledger.credit(uid, 50_000_000, "grant", "t", "g")
    kh = key_headers(uid, "stream", monthly_cap_micros=10_000)  # less than the five minutes live reserves
    client.cookies.clear()
    with client.websocket_connect("/api/stream/ws", headers=kh) as ws:
        ws.send_text(json.dumps({"type": "start", "denoise": "light", "sampleRate": 48000}))
        err = ws.receive_json()
        assert err["type"] == "error" and err["code"] == "key_spending_cap" and "monthly spending limit" in err["message"]
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()
    assert count("stream_sessions") == 0 and ledger.balance(uid)["held_micros"] == 0
    # a roomier cap starts normally
    db.execute("UPDATE api_keys SET monthly_cap_micros = 1000000")
    with client.websocket_connect("/api/stream/ws", headers=kh) as ws:
        ws.send_text(json.dumps({"type": "start", "denoise": "light", "sampleRate": 48000}))
        assert ws.receive_json()["type"] == "ready"
        ws.send_text(json.dumps({"type": "stop"}))
    row = db.one("SELECT api_key_id FROM stream_sessions")
    assert row["api_key_id"] == db.one("SELECT id FROM api_keys")["id"]
