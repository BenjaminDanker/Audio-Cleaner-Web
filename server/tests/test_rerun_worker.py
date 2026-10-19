"""Re-runs through the real worker and the sandboxed runner ("Change and save again" end to end).

Needs the native pieces (~/audio-cleaner/bin, models) and the speech sample the other pipeline tests use; skipped
otherwise. The sample is cut to eight seconds. These tests are slow (a model loads in every runner process): run them
detached, see the notes in the task that asked for them."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from acw import db, jobs
from acw.billing import ledger, pricing
from acw_common import kept as kept_copies
from acw_common import pipeline
from acw_common.ffmpeg import FFmpeg

from .rerun_support import SR, count, files_of, folder, params_of, row, same_file
from .test_job_pipeline import HOME, SAMPLE, _run_worker_once

NATIVE = pytest.mark.skipif(
    not (Path(SAMPLE).exists() and (HOME / "bin/libdf.so").exists() and (HOME / "bin/ffmpeg").exists()),
    reason="native binaries/models or sample audio not available",
)
SECONDS = 8
MODELS = HOME / "models"
HAVE_ASR = all((MODELS / m).is_file() for m in ("parakeet-v2/encoder-model.onnx", "whisper/base/model.bin"))


@pytest.fixture(scope="module")
def speech(tmp_path_factory):
    """The first eight seconds of the speech sample, as a WAV file."""
    out = tmp_path_factory.mktemp("rerun-sample") / "talk.wav"
    subprocess.run([str(HOME / "bin/ffmpeg"), "-v", "error", "-i", SAMPLE, "-t", str(SECONDS), "-y", str(out)], check=True)
    return out


def ff() -> FFmpeg:
    return FFmpeg(HOME / "bin")


def start_job(client, h, data, kind="denoise", transcribe_tier=None, tier="standard", **options) -> str:
    """Create a job and send its upload; returns its id (the job is queued)."""
    r = client.post("/api/jobs", json={
        "kind": kind, "denoiseTier": tier, "transcribeTier": transcribe_tier, "filename": "talk.wav", "sizeBytes": len(data),
        "durationSeconds": SECONDS,
        "options": {"output_video": False, "noise": "light", "audio_format": "flac", "loudness_lufs": -16.0, **options},
    }, headers=h)
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert client.put(f"/api/jobs/{jid}/input", content=data, headers=h).status_code == 200
    return jid


def run_rerun(client, h, source, expect_free=None, **body) -> str:
    """Ask for a re-run, run the worker once, return the new job's id."""
    est = client.post(f"/api/jobs/{source}/rerun/estimate", json=body, headers=h)
    assert est.status_code == 200, est.text
    made = client.post(f"/api/jobs/{source}/rerun", json={**body, "maxCostMicros": est.json()["estCostMicros"]}, headers=h)
    assert made.status_code == 200, made.text
    assert made.json()["estCostMicros"] == est.json()["estCostMicros"]
    if expect_free is not None:
        assert made.json()["free"] is expect_free
    new_id = _run_worker_once("done")
    assert new_id == made.json()["id"]
    return new_id


def decode(path: Path, channels=1):
    """The samples of an audio file, float32 (samples, channels), 48 kHz."""
    f = ff()
    return f.decode(path, f.probe(path), SR, channels, SECONDS + 5)


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    n = min(len(reference), len(test))
    err = test[:n].astype(np.float64) - reference[:n].astype(np.float64)
    return float(10 * np.log10(np.sum(reference[:n].astype(np.float64) ** 2) / max(float(np.sum(err**2)), 1e-30)))


def usage_lines(uid):
    return db.all_rows("SELECT * FROM ledger WHERE user_id = ? AND kind = 'usage' ORDER BY id", (uid,))


@NATIVE
def test_change_and_save_again_end_to_end(client, signed_in, speech, tmp_path):
    """A Denoising job, then changes through the worker: file type (free), strength (free, a true blend of the kept copies),
    Leveling (charged once), a re-run of a re-run, and finally everything of the chain is deleted together."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    data = speech.read_bytes()

    # (1) the first job keeps its upload, both kept copies and its result, for two hours
    a = start_job(client, h, data)
    assert _run_worker_once("done") == a
    first = row(a)
    assert files_of(a) == {"input.wav", "kept-noise.flac", "kept-input.flac", "talk-cleaned.flac"}
    assert first["expires_at"] == first["finished_at"] + 7200
    shown = client.get(f"/api/jobs/{a}").json()
    assert shown["redo"]["available"] and shown["redo"]["noise"] and shown["expiresAt"] == first["expires_at"]
    preview = shown["redo"]["preview"]
    seconds = first["duration_seconds"]
    assert preview["channels"] == [0] and preview["plan"] == [0] and abs(preview["samples"] - seconds * SR) < SR // 10
    assert preview["bytes"] == {k: (folder(a) / f"kept-{k}.flac").stat().st_size for k in ("noise", "input")}
    paid_first = usage_lines(uid)
    assert len(paid_first) == 1 and paid_first[0]["description"].startswith("Denoising + Leveling: talk.wav")
    price = pricing.job_cost_micros("denoise", "standard", None, seconds, {"denoise": 9_000, "leveling": 1_000}, 0, True)
    assert -paid_first[0]["amount_micros"] == price == first["cost_micros"] > 0

    # the two kept copies are what the page blends: the recording as the remover heard it, and the noise removal
    f = ff()
    src = folder(a) / "input.wav"
    x = pipeline.decode_for_clean(f, src, f.probe(src), SECONDS * 1.02 + 2)
    entry_in, entry_noise = params_of(a)["kept"]["input"], params_of(a)["kept"]["noise"]
    heard = kept_copies.load(f, entry_in, folder(a), np.zeros(x.shape, np.float32), tmp_path / "s1.f32")
    assert np.max(np.abs(heard - x)) < 1e-6
    full = kept_copies.load(f, entry_noise, folder(a), np.zeros(x.shape, np.float32), tmp_path / "s2.f32")
    assert snr_db(x, full) < 30  # the noise removal really changed the sound

    # (2) another file type only: free, nothing but a new encoding of the result that is there
    r1 = run_rerun(client, h, a, expect_free=True, options={"audio_format": "wav"})
    assert row(r1)["cost_micros"] == 0 and row(r1)["hold_id"] is None and len(usage_lines(uid)) == 1
    assert params_of(r1)["rerun"]["clean"] == "reuse" and "talk-cleaned.wav" in files_of(r1)
    assert np.max(np.abs(decode(folder(r1) / "talk-cleaned.wav") - decode(folder(a) / "talk-cleaned.flac"))) < 1e-6  # lossless

    # (3) strength only, without Leveling: free, and exactly the blend of the kept copies
    r2 = run_rerun(client, h, r1, expect_free=True, options={"atten_lim_db": 24, "leveling": False})
    done2 = row(r2)
    assert done2["status"] == "done" and done2["cost_micros"] == 0 and done2["hold_id"] is None
    assert params_of(r2)["rerun"]["clean"] == "denoised" and params_of(r2)["rerun"]["paid"] == []
    assert len(usage_lines(uid)) == 1  # still only the first job's line
    # no noise model ran: it kept the copy of the first job (a new copy would have been written as a new file)
    for name in ("kept-noise.flac", "kept-input.flac"):
        assert same_file(folder(a) / name, folder(r2) / name), name
    mix = 10 ** (-24 / 20)
    expected = (1 - mix) * full + mix * x
    got = decode(folder(r2) / "talk-cleaned.wav")
    assert abs(len(got) - len(expected)) <= 1
    assert snr_db(expected, got) > 70
    assert np.max(np.abs(got[:len(expected)] - expected[:len(got)])) < 1e-3
    # and it is a different sound from the full-strength result (the blend took effect)
    assert snr_db(full, got) < 40

    # (4) Leveling on again: charged, once, for the Leveling step only
    before_lines = len(usage_lines(uid))
    r3 = run_rerun(client, h, r2, expect_free=False, options={"leveling": True})
    done3 = row(r3)
    lines = usage_lines(uid)
    assert len(lines) == before_lines + 1
    assert lines[-1]["description"].startswith("Changed and saved again (Leveling): talk.wav")
    assert -lines[-1]["amount_micros"] == pricing.rerun_cost_micros({"leveling": 1_000}, done3["duration_seconds"]) == done3["cost_micros"] > 0
    assert params_of(r3)["rerun"]["paid"] == ["leveling"] and params_of(r3)["rates"] == {"leveling": 1_000}
    leveled = decode(folder(r3) / "talk-cleaned.wav")
    assert np.max(np.abs(leveled[:len(got)] - got[:len(leveled)])) > 1e-3  # Leveling changed the sound
    assert ledger.balance(uid)["held_micros"] == 0 and ledger.invariant_problems() == {"balance_mismatch": 0, "held_mismatch": 0}

    # the chain: one root, every re-run ends with the first job and shows what it was made from
    chain = [a, r1, r2, r3]
    assert [row(j)["root_job_id"] for j in chain] == [None, a, a, a]
    assert [row(j)["source_job_id"] for j in chain] == [None, a, r1, r2]
    assert {row(j)["expires_at"] for j in chain} == {first["expires_at"]}
    assert client.get(f"/api/jobs/{r3}").json()["sourceJobId"] == r2

    # (5) when the two hours are over, everything of the chain goes together and a new change is refused
    db.execute("UPDATE jobs SET expires_at = ? WHERE status = 'done'", (db.now() - 1,))
    jobs.sweep()
    for j in chain:
        assert row(j)["status"] == "expired" and not folder(j).exists(), j
    spent = ledger.balance(uid)
    r = client.post(f"/api/jobs/{r3}/rerun", json={"options": {"atten_lim_db": 12}}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "job_files_gone" and r.headers["X-Error-Code"] == "job_files_gone"
    assert client.get(f"/api/jobs/{a}/kept/noise").status_code == 404
    assert client.get(f"/api/jobs/{a}/outputs/talk-cleaned.flac", headers=h).status_code == 404
    assert ledger.balance(uid) == spent and count("jobs", "status = 'done'") == 0


@NATIVE
def test_a_strength_change_never_calls_the_noise_model_in_the_runner(client, signed_in, speech, monkeypatch):
    """The runner's "denoised" mode, run in this process with the noise model and the noise choice made to fail: the re-run
    still finishes from the kept copies."""
    from acw.worker import main as worker
    from acw.worker import runner

    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    a = start_job(client, h, speech.read_bytes(), leveling=False)
    assert _run_worker_once("done") == a
    made = client.post(f"/api/jobs/{a}/rerun", json={"options": {"atten_lim_db": 30}}, headers=h)
    assert made.status_code == 200 and made.json()["free"] is True, made.text

    def boom(*args, **kwargs):
        raise AssertionError("the noise model must not run for a change of strength")

    monkeypatch.setattr(pipeline, "clean_audio", boom)
    monkeypatch.setattr(pipeline, "choose_noise_profile", boom)
    job = worker.claim("test")
    assert job is not None and job["id"] == made.json()["id"]
    d = worker._write_spec(job, {**worker._settings_extra(), "threads": 2}, job["duration_seconds"])
    spec = json.loads((d / "spec.json").read_text())
    assert spec["rerun"]["clean"] == "denoised" and spec["rerun"]["transcribe"] is False and set(spec["rerun"]["kept"]) == {"noise", "input"}
    res = runner.stage_main(d, spec)
    assert res["kept"] == {} and res["noise"] == "light"  # nothing new was kept; the remover named is the kept copy's
    worker._complete(job, d, res)
    assert row(job["id"])["status"] == "done"
    assert (folder(job["id"]) / "talk-cleaned.flac").stat().st_size > 1000
    assert same_file(folder(a) / "kept-noise.flac", folder(job["id"]) / "kept-noise.flac")
    assert client.get(f"/api/jobs/{job['id']}").json()["redo"]["preview"]["samples"] == params_of(a)["kept"]["noise"]["samples"]


@NATIVE
def test_a_blend_of_a_longer_strength_matches_the_noise_removal_the_job_did(client, signed_in, speech, tmp_path):
    """Strength 100 dB (no limit) from the kept noise removal is that noise removal itself, without Leveling."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    a = start_job(client, h, speech.read_bytes(), leveling=False, atten_lim_db=12, audio_format="wav")
    assert _run_worker_once("done") == a
    limited = decode(folder(a) / "talk-cleaned.wav")
    b = run_rerun(client, h, a, expect_free=True, options={"atten_lim_db": 100, "audio_format": "wav"})
    full_strength = decode(folder(b) / "talk-cleaned.wav")
    f = ff()
    src = folder(a) / "input.wav"
    x = pipeline.decode_for_clean(f, src, f.probe(src), SECONDS * 1.02 + 2)
    kept = kept_copies.load(f, params_of(a)["kept"]["noise"], folder(a), np.zeros(x.shape, np.float32), tmp_path / "k.f32")
    assert snr_db(kept, full_strength) > 70  # the full-strength result is the kept noise removal as it is
    mix = 10 ** (-12 / 20)
    assert snr_db((1 - mix) * kept + mix * x, limited) > 70  # and the first job's own 12 dB limit was the same blend
    c = run_rerun(client, h, b, expect_free=True, options={"atten_lim_db": 12})
    assert snr_db(limited, decode(folder(c) / "talk-cleaned.wav")) > 60  # going back gives the first job's sound again (here 24-bit)


@NATIVE
@pytest.mark.skipif(not HAVE_ASR, reason="transcription models not installed")
def test_subtitle_files_again_are_free_and_a_new_language_transcribes_again_and_is_charged(client, signed_in, speech):
    """From a job with subtitles: other caption files only (free, the transcript is reused as it is); another language
    runs the transcription again (charged for it only, the cleaned sound is reused)."""
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    a = start_job(client, h, speech.read_bytes(), kind="denoise_transcribe", transcribe_tier="local", language="en",
                  caption_formats=["srt"])
    assert _run_worker_once("done") == a
    assert files_of(a) >= {"segments.json", "talk.srt", "kept-noise.flac", "talk-cleaned.flac"} and "talk.vtt" not in files_of(a)
    assert client.get(f"/api/jobs/{a}").json()["redo"]["transcript"] is True
    lines = len(usage_lines(uid))

    b = run_rerun(client, h, a, expect_free=True, options={"caption_formats": ["srt", "vtt", "txt"]})
    assert {"talk.srt", "talk.vtt", "talk-transcript.txt"} <= set(files_of(b)) and row(b)["cost_micros"] == 0
    assert same_file(folder(a) / "segments.json", folder(b) / "segments.json")  # the transcript was not made again
    assert params_of(b)["rerun"]["transcribe"] is False and params_of(b)["rerun"]["clean"] == "reuse"
    assert len(usage_lines(uid)) == lines
    assert (folder(b) / "talk.srt").read_text() == (folder(a) / "talk.srt").read_text()

    if (MODELS / "parakeet-v3/encoder-model.onnx").is_file():
        c = run_rerun(client, h, b, expect_free=False, options={"language": "de"})
        assert params_of(c)["rerun"]["transcribe"] is True and params_of(c)["rerun"]["paid"] == ["transcribe"]
        assert not same_file(folder(b) / "segments.json", folder(c) / "segments.json")  # a new transcript
        assert same_file(folder(a) / "kept-noise.flac", folder(c) / "kept-noise.flac")  # the cleaning was not done again
        line = usage_lines(uid)[-1]
        assert line["description"].startswith("Changed and saved again (Subtitles & transcript): talk.wav")
        assert -line["amount_micros"] == pricing.rerun_cost_micros({"transcribe": 4_000}, row(c)["duration_seconds"]) > 0
