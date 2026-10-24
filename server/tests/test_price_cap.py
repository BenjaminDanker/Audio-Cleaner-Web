"""A job never costs more than 10 % above the price its caller was shown (`maxCostMicros`, jobs.PRICE_TOLERANCE): creating one that would is
refused (409 price_changed, nothing held), one whose real length makes it dearer is stopped when its file is read (failed, hold released, nothing
charged), within the 10 % it goes on and is charged what it really costs, and without `maxCostMicros` nothing changes. Edit runs are not part of it
(they are exact: tests/test_editor_api.py). The runner's stages are replaced (tests/test_automatic_settings.py Stages), so no models are needed."""
import math

import pytest

from acw import jobs
from acw.billing import ledger
from acw.worker import main as worker

from .job_support import params_of, roomy_disk, row  # noqa: F401 - fixture
from .test_automatic_settings import Stages
from .test_timeline_jobs import funded, held, standard, take

pytestmark = pytest.mark.usefixtures("roomy_disk")


def create(client, h, seconds=60, max_cost=None, **extra):
    body = {"kind": "denoise", "filename": "talk.wav", "sizeBytes": 3000, "durationSeconds": seconds, "denoiseTier": "standard", "options": {}, **extra}
    if max_cost is not None:
        body["maxCostMicros"] = max_cost
    return client.post("/api/jobs", json=body, headers=h)


def upload(client, h, jid):
    assert client.put(f"/api/jobs/{jid}/input", content=bytes(3000), headers=h).status_code == 200


def test_over_quote_is_more_than_ten_percent_above_the_price_shown_and_never_when_none_was_given():
    assert jobs.PRICE_TOLERANCE == 0.10 and jobs.PRICE_CHANGED == "price_changed"
    assert jobs.over_quote(100, None) is False
    assert jobs.over_quote(100, 100) is False
    assert jobs.over_quote(110, 100) is False  # (exactly 10 % above is allowed)
    assert jobs.over_quote(111, 100) is True
    assert jobs.over_quote(0, 0) is False and jobs.over_quote(1, 0) is True
    assert jobs.over_quote(50, 100) is False  # (cheaper is always fine)


def test_creating_a_job_more_than_ten_percent_above_the_price_shown_is_refused_and_nothing_is_held(client, signed_in):
    uid, h = funded(signed_in)
    est = standard(60)
    ok = math.ceil(est / 1.1)  # (the smallest price shown that allows it)
    assert not jobs.over_quote(est, ok) and jobs.over_quote(est, ok - 1)
    r = create(client, h, max_cost=ok - 1)
    assert r.status_code == 409 and r.json().get("code") == "price_changed", r.text
    assert f"${est / 1e6:.2f}" in r.json()["detail"] and "Nothing was charged" in r.json()["detail"]
    assert held(uid) == 0 and ledger.balance(uid)["balance_micros"] == 5_000_000
    from acw import db
    assert db.one("SELECT COUNT(*) AS n FROM jobs WHERE user_id = ?", (uid,))["n"] == 0
    # at the limit (and above the price): accepted, held at the real estimate, the price shown is kept for the worker
    r = create(client, h, max_cost=ok)
    assert r.status_code == 200, r.text
    assert held(uid) == est and params_of(r.json()["id"])["max_cost_micros"] == ok
    r2 = create(client, h, max_cost=est * 3)
    assert r2.status_code == 200 and params_of(r2.json()["id"])["max_cost_micros"] == est * 3


def test_a_price_of_exactly_ten_percent_more_is_accepted_and_one_micro_less_is_not(client, signed_in):
    uid, h = funded(signed_in)
    seconds = next(s for s in range(60, 400) if standard(s) % 11 == 0 and not jobs.over_quote(standard(s), standard(s) * 10 // 11))  # (a length whose price is exactly 110 % of a whole number of micros)
    est = standard(seconds)
    quoted = est * 10 // 11
    assert quoted * 11 == est * 10 and quoted * 1.1 == pytest.approx(est)
    assert create(client, h, seconds=seconds, max_cost=quoted).status_code == 200
    assert create(client, h, seconds=seconds, max_cost=quoted - 1).status_code == 409


def test_without_a_price_shown_nothing_is_checked_or_stored(client, signed_in, monkeypatch):
    uid, h = funded(signed_in)
    Stages(monkeypatch, None, duration=600.0)  # (a file ten times longer than it was said to be)
    jid = create(client, h, seconds=60).json()["id"]
    upload(client, h, jid)
    assert "max_cost_micros" not in params_of(jid)
    take()
    done = client.get(f"/api/jobs/{jid}").json()
    assert done["status"] == "done" and done["costMicros"] == standard(600) and held(uid) == 0


def test_a_file_whose_real_length_makes_it_more_than_ten_percent_dearer_fails_when_it_is_read_and_is_not_charged(client, signed_in, monkeypatch):
    uid, h = funded(signed_in)
    Stages(monkeypatch, None, duration=90.0)
    shown = standard(60)
    jid = create(client, h, seconds=60, max_cost=shown).json()["id"]
    upload(client, h, jid)
    assert held(uid) == shown
    take()
    j = client.get(f"/api/jobs/{jid}").json()
    assert j["status"] == "failed" and not j.get("costMicros")
    assert "more than 10% above" in j["error"] and "weren't charged" in j["error"] and "1.5 min long, not the 1.0 min" in j["error"], j["error"]
    assert held(uid) == 0 and ledger.balance(uid)["balance_micros"] == 5_000_000


def test_a_file_a_little_longer_than_said_goes_on_within_ten_percent_and_is_charged_what_it_really_costs(client, signed_in, monkeypatch):
    uid, h = funded(signed_in)
    real = 63.0  # 5 % longer
    assert not jobs.over_quote(standard(real), standard(60))
    Stages(monkeypatch, None, duration=real)
    jid = create(client, h, seconds=60, max_cost=standard(60)).json()["id"]
    upload(client, h, jid)
    take()
    j = client.get(f"/api/jobs/{jid}").json()
    assert j["status"] == "done" and j["costMicros"] == standard(real) > standard(60) and held(uid) == 0
    assert ledger.balance(uid)["balance_micros"] == 5_000_000 - standard(real)


def test_a_shorter_file_than_said_costs_less_and_never_trips_the_cap(client, signed_in, monkeypatch):
    uid, h = funded(signed_in)
    Stages(monkeypatch, None, duration=30.0)
    jid = create(client, h, seconds=60, max_cost=standard(60)).json()["id"]
    upload(client, h, jid)
    take()
    j = client.get(f"/api/jobs/{jid}").json()
    assert j["status"] == "done" and j["costMicros"] == standard(30) and held(uid) == 0


def test_the_cap_applies_to_a_job_made_with_a_timeline_too_whose_parts_do_not_grow_with_the_file(client, signed_in, monkeypatch):
    from acw import editing

    assert editing.PRICE_CHANGED == jobs.PRICE_CHANGED  # (the edit runs' refusal is the same code; their price is exact, not capped by a tolerance)
    uid, h = funded(signed_in)
    Stages(monkeypatch, None, duration=120.0)
    tl = {"version": 1, "seconds": 60, "layers": {"noise": [{"start": 0, "end": 60, "strength_db": 24, "remover": "light"}]}}
    est = client.post("/api/jobs/estimate", json={"kind": "denoise", "durationSeconds": 60, "options": {}, "timeline": tl}, headers=h).json()["estCostMicros"]
    body = {"kind": "denoise", "filename": "talk.wav", "sizeBytes": 3000, "durationSeconds": 60, "options": {}, "timeline": tl}
    assert client.post("/api/jobs", json={**body, "maxCostMicros": est - 1_000}, headers=h).status_code == 409
    jid = client.post("/api/jobs", json={**body, "maxCostMicros": est}, headers=h).json()["id"]
    upload(client, h, jid)
    take()
    j = client.get(f"/api/jobs/{jid}").json()
    # (the parts a person chose stay what they were, whatever the file's real length: the price did not change, so the cap is not tripped)
    assert j["status"] == "done" and j["costMicros"] == est and held(uid) == 0
