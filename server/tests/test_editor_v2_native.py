"""Timeline version 2 end to end through the real worker and the sandboxed runner: a part the browser cleaned, subtitles per part (Standard
with the local recogniser, Accurate and the translations with the web process's work faked, the browser's own cues), the edit runs that
follow, and the outline of a layer made by the confined helper. Needs the native pieces like test_editor_native.py; skipped otherwise."""
import asyncio
import json
import subprocess

import numpy as np
import pytest

from acw import app as web, db, editing, stretch, translate
from acw.asr import openai_asr
from acw.billing import ledger
from acw_common import blocks
from acw_common.captions import Segment, segments_to_json

from .rerun_support import SR, clock, files_of, folder, params_of, roomy_disk, row, set_env  # noqa: F401
from .test_editor_native import (HOME, NATIVE, SECONDS, decode, ff, level, noise, speech, usage_lines)  # noqa: F401
from .test_editor_v2 import browser, part_cost, put_cues, put_noise, sub, v2
from .test_job_pipeline import _run_worker_once

MODELS = HOME / "models"
needs_asr = pytest.mark.skipif(not all((MODELS / m).is_file() for m in ("parakeet-v2/encoder-model.onnx", "whisper/base/model.bin")),
                               reason="Parakeet or Whisper base not installed")


def cleaned_flac(speech_path, tmp_path, a: float, b: float, gain=0.5, channels=1) -> bytes:
    """What the browser would send for [a, b): the recording's samples there (made quieter, so it can be told from the recording), FLAC 48 kHz."""
    out = tmp_path / f"cleaned-{a}-{b}.flac"
    subprocess.run([str(HOME / "bin/ffmpeg"), "-v", "error", "-i", str(speech_path), "-ss", str(a), "-t", str(b - a), "-af", f"volume={gain}", "-ar", "48000",
                    "-ac", str(channels), "-y", str(out)], check=True)
    return out.read_bytes()


def create(client, h, data, timeline, kind="denoise_transcribe", **options):
    r = client.post("/api/jobs", json={"kind": kind, "filename": "talk.wav", "sizeBytes": len(data), "durationSeconds": SECONDS, "timeline": timeline,
                                      "options": {"output_video": False, "audio_format": "flac", "caption_formats": ["srt", "txt"], **options}}, headers=h)
    assert r.status_code == 200, r.text
    return r.json()["id"], r.json().get("needs", [])


def run_the_web_process(jid):
    """The cloud work of a job in `cloud_asr`: the web process's loop for it (the transcription is one phase, the translations the next),
    after which the job is back in the queue for the worker's last stage."""
    phases = []
    while row(jid)["status"] == "cloud_asr":
        phases.append(editing.subtitle_cloud_phase(jid, params_of(jid)))
        asyncio.run(web._run_subtitle_parts(row(jid), params_of(jid)))
        assert len(phases) <= 2
    return phases


@pytest.fixture
def fake_cloud(monkeypatch):
    """Accurate transcription and translations without the internet: one segment per speech chunk, and `[<language>] <text>` translations."""
    async def run_for_job(job_dir, language, billed_seconds=None):
        index = json.loads((job_dir / "asr_chunks" / "index.json").read_text())
        segs = [Segment(c["offset"], c["offset"] + c["seconds"], f"accurate chunk {i}") for i, c in enumerate(index["chunks"])]
        (job_dir / "segments.json").write_text(json.dumps({"language": language, "engine": "premium", "segments": segments_to_json(segs)}))

    async def translate_job(job_dir, targets):
        data = json.loads((job_dir / "segments.json").read_text())
        for lang in targets:
            (job_dir / f"segments.{lang}.json").write_text(json.dumps({"language": lang, "engine": "translation", "segments": [
                {**s, "text": f"[{lang}] {s['text']}", "words": []} for s in data["segments"]]}))
        return list(targets)

    monkeypatch.setattr(openai_asr, "run_for_job", run_for_job)
    monkeypatch.setattr(translate, "translate_job", translate_job)


@NATIVE
@needs_asr
def test_a_first_run_with_a_browser_cleaned_part_and_subtitles_per_part(client, signed_in, speech, roomy_disk, tmp_path, fake_cloud):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    data = speech.read_bytes()
    tl = v2(SECONDS, noise=[browser(2, 5)], leveling=[], subtitles=[sub(0, 5.5, "standard", "en", ["es"]), sub(6, 8, "free", "en")])
    jid, needs = create(client, h, data, tl)
    assert needs == [{"kind": "noise", "start": 2.0, "end": 5.0}, {"kind": "cues", "start": 6.0, "end": 8.0}]
    hold = row(jid)["est_cost_micros"]
    assert hold == part_cost("transcribe_standard", 6) + part_cost("translate", 6)
    assert client.put(f"/api/jobs/{jid}/input", content=data, headers=h).status_code == 200
    assert row(jid)["status"] == "uploading"
    cleaned = cleaned_flac(speech, tmp_path, 2, 5)
    assert put_noise(client, h, jid, 2, 5, cleaned).status_code == 200
    assert row(jid)["status"] == "uploading"
    assert put_cues(client, h, jid, 6, 8, [{"start": 6.2, "end": 7.9, "text": "A line from the browser"}, {"start": 5, "end": 5.5, "text": "outside"}][:1]).status_code == 200
    assert row(jid)["status"] == "queued"
    # the worker: the sound and the Standard part's text; the Spanish waits for the web process
    _run_worker_once("cloud_asr")
    seg = json.loads(next((folder(jid) / "subs").iterdir()).joinpath("segments.json").read_text())
    assert seg["language"] == "en" and "held" in " ".join(s["text"] for s in seg["segments"]).lower()
    assert all(s["end"] <= 5.5 + 1e-6 and s["start"] >= 0 for s in seg["segments"])  # cut to the part
    assert run_the_web_process(jid) == ["translate"]
    assert row(jid)["status"] == "queued" and row(jid)["next_stage"] == "finalize"
    _run_worker_once("done")
    done = row(jid)
    assert done["cost_micros"] == hold == part_cost("transcribe_standard", 6) + part_cost("translate", 6)
    names = {o["name"] for o in client.get(f"/api/jobs/{jid}").json()["outputs"]}
    assert names == {"talk-cleaned.flac", "talk.srt", "talk-transcript.txt", "talk.es.srt", "talk.es-transcript.txt"}
    srt = client.get(f"/api/jobs/{jid}/outputs/talk.srt").text
    assert "A line from the browser" in srt and "-->" in srt and "outside" not in srt
    assert "[es] " in client.get(f"/api/jobs/{jid}/outputs/talk.es-transcript.txt").text
    # the sound: the recording outside the part the browser cleaned, the browser's audio inside it
    got = decode(folder(jid) / "talk-cleaned.flac")
    x = ff().decode(folder(jid) / "input.wav", ff().probe(folder(jid) / "input.wav"), SR, 1, SECONDS + 5)
    assert np.max(np.abs(got[: int(1.8 * SR), 0] - x[: int(1.8 * SR), 0])) < 1e-4
    assert np.max(np.abs(got[int(5.2 * SR): len(x), 0] - x[int(5.2 * SR): len(x), 0])) < 1e-4
    ref = decode(speech)[int(2.2 * SR): int(4.8 * SR), 0] * 0.5
    assert np.max(np.abs(got[int(2.2 * SR): int(4.8 * SR), 0] - ref)) < 2e-4
    # nothing was made by a noise remover (no blocks), the files that matter are kept, the chunks are gone
    p = params_of(jid)
    assert p["edit"]["rendered"]["layers"]["noise"] == [{"start": 2.0, "end": 5.0, "browser": True}] and "subtitles" not in p["edit"]["rendered"]["layers"]
    assert {"browser", "subs", "input.wav"} <= files_of(jid) and not list((folder(jid) / "subs").glob("*/asr_chunks"))
    assert [-r["amount_micros"] for r in usage_lines(uid)] == [hold]
    shown = client.get(f"/api/jobs/{jid}/timeline").json()
    assert shown["timeline"]["version"] == 2 and shown["needs"] == [] and shown["cached"]["subtitles"] == [[0.0, 5.5], [6.0, 8.0]]
    # the result files hold this timeline
    q = client.post(f"/api/jobs/{jid}/quote", json={}, headers=h).json()
    assert q["unchanged"] is True and q["estCostMicros"] == 0
    # the outline of the recording, made by the confined helper
    src = client.get(f"/api/jobs/{jid}/peaks/source")
    assert src.status_code == 200, src.text
    peaks = src.json()["peaks"]
    assert len(peaks) == -(-len(x) // 12_000) and max(peaks) == pytest.approx(float(np.max(np.abs(x))), abs=2e-3) and src.json()["rate"] == 4
    assert client.get(f"/api/jobs/{jid}/peaks/noise").status_code == 404  # no remover ran: nothing of that layer is cached


@NATIVE
@needs_asr
def test_an_edit_run_makes_only_the_subtitle_parts_that_changed(client, signed_in, speech, roomy_disk, tmp_path, fake_cloud):
    uid, h = signed_in
    ledger.credit(uid, 5_000_000, "grant", "t", "g")
    data = speech.read_bytes()
    first = v2(SECONDS, noise=[{"start": 2, "end": 5, "strength_db": 100, "remover": "light"}], subtitles=[sub(0, 5.5, "standard", "en")])
    jid, _ = create(client, h, data, first)
    assert client.put(f"/api/jobs/{jid}/input", content=data, headers=h).status_code == 200
    _run_worker_once("done")
    spent = row(jid)["cost_micros"]
    assert spent == part_cost("transcribe_standard", 6) + editing.step_micros(editing.step_rates()["light"], 3, 0.27 / 0.116)
    audio_before = (folder(jid) / "talk-cleaned.flac").read_bytes()
    # an Accurate part is added: only it is made and paid for; the sound is not rendered again and the first part's text is kept
    second = v2(SECONDS, noise=first["layers"]["noise"], subtitles=[sub(0, 5.5, "standard", "en"), sub(5.5, 8, "accurate", "en", ["fr"])])
    q = client.post(f"/api/jobs/{jid}/quote", json={"timeline": second}, headers=h).json()
    want = part_cost("transcribe_accurate", 3) + part_cost("translate", 3)
    assert q["estCostMicros"] == want and q["steps"] == {"subtitles": {"parts": 1, "seconds": 6}} and q["unchanged"] is False
    r = client.post(f"/api/jobs/{jid}/process", json={"timeline": second, "maxCostMicros": want}, headers=h)
    assert r.status_code == 200, r.text
    run = params_of(jid)["run"]
    assert run["audio"] is False
    _run_worker_once("cloud_asr")
    assert run_the_web_process(jid) == ["asr", "translate"]
    _run_worker_once("done")
    assert row(jid)["cost_micros"] == spent + want
    assert (folder(jid) / "talk-cleaned.flac").read_bytes() == audio_before
    names = {o["name"] for o in client.get(f"/api/jobs/{jid}").json()["outputs"]}
    assert names == {"talk-cleaned.flac", "talk.srt", "talk-transcript.txt", "talk.fr.srt", "talk.fr-transcript.txt"}
    srt = client.get(f"/api/jobs/{jid}/outputs/talk.srt").text
    assert "held" in srt.lower() and "accurate chunk" in srt
    assert "[fr] accurate chunk" in client.get(f"/api/jobs/{jid}/outputs/talk.fr.srt").text and "held" not in client.get(f"/api/jobs/{jid}/outputs/talk.fr.srt").text.lower()
    assert [-r["amount_micros"] for r in usage_lines(uid)][-1] == want
    assert client.post(f"/api/jobs/{jid}/quote", json={"timeline": second}, headers=h).json()["unchanged"] is True
    # a French part moved to German: the text is kept, only the German translation is made (paid for one part's language)
    third = v2(SECONDS, noise=first["layers"]["noise"], subtitles=[sub(0, 5.5, "standard", "en"), sub(5.5, 8, "accurate", "en", ["fr", "de"])])
    q = client.post(f"/api/jobs/{jid}/quote", json={"timeline": third}, headers=h).json()
    assert q["estCostMicros"] == part_cost("translate", 3) and q["lines"] == [l for l in q["lines"] if l["item"] == "translate"]
    # taking the subtitles away: nothing to pay, the caption files go
    none = {"version": 1, "seconds": SECONDS, "layers": {"noise": first["layers"]["noise"]}}
    q = client.post(f"/api/jobs/{jid}/quote", json={"timeline": none}, headers=h).json()
    assert q["estCostMicros"] == 0 and q["unchanged"] is False and q["needsProcess"] is False
    assert client.post(f"/api/jobs/{jid}/export", json={"timeline": none}, headers=h).status_code == 200
    _run_worker_once("done")
    assert {o["name"] for o in client.get(f"/api/jobs/{jid}").json()["outputs"]} == {"talk-cleaned.flac"}
    assert not any(n.startswith("talk.") and n.endswith(".srt") for n in files_of(jid))
