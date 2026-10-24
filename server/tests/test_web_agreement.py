"""Things the website and the server each know and must agree on: the subtitle languages (the site's list, the shared timeline's list, the
translator's), and the rules for making a video's picture again on our side (which file types take a picture as it is, and how much work
making it again is: what the website quotes before the upload, and what the server charges). The website's side is read from its source
text, or computed by node (web/tests/convert-grid.mjs, the site's own code bundled; needs web/node_modules, so it is skipped without them)."""
import itertools
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from acw import translate
from acw_common import ffmpeg, timeline

WEB = Path(__file__).resolve().parents[2] / "web"


def js_languages() -> list[str]:
    """The codes of web/src/jobs.js LANGUAGES (its first entry, '', is "detect automatically")."""
    text = (WEB / "src/jobs.js").read_text()
    body = re.search(r"export const LANGUAGES = \[(.*?)\n\];", text, re.S).group(1)
    return re.findall(r"\['([a-z]*)',\s*'[^']*'\]", body)


def js_iso639_2() -> list[str]:
    text = (WEB / "src/media/convert.js").read_text()
    body = re.search(r"export const ISO639_2 = \{(.*?)\};", text, re.S).group(1)
    return re.findall(r"\b([a-z]{2}): '[a-z]{3}'", body)


def test_the_language_lists_of_the_site_the_shared_timeline_and_the_translator_are_the_same():
    site = js_languages()
    assert site[0] == "" and len(site) == len(set(site)) > 20
    assert site[1:] == list(timeline.LANGUAGES)  # (same codes, same order)
    assert set(translate.LANGUAGE_NAMES) == set(timeline.LANGUAGES)
    assert list(translate.LANGUAGE_NAMES) == list(timeline.LANGUAGES)


def test_every_language_of_the_site_has_a_three_letter_code_for_the_subtitle_tracks_it_writes():
    assert sorted(js_iso639_2()) == sorted(timeline.LANGUAGES)


# ------------------------------------------------------------------------------------------------------------- the picture of a video

needs_node = pytest.mark.skipif(shutil.which("node") is None or not (WEB / "node_modules/esbuild").is_dir(),
                                reason="needs node and the website's packages (npm ci in web/)")

CODECS = ["h264", "hevc", "vp8", "vp9", "av1", "mpeg4", "prores", "mjpeg", "mpeg2video", "dnxhd", "flv1", "wmv3", "xyz", None]
SOURCES = ["mp4", "m4v", "mov", "mkv", "webm", "avi", "wmv", "flv"]
FORMATS = ["mp4", "mov", "mkv", "webm"]  # (the file types the website offers)
SHAPES = [(1920, 1080), (1280, 720), (3840, 2160), (2560, 1440), (640, 360), (854, 480), (1080, 1920), (720, 1280), (1440, 1080), (3000, 2000),
          (1024, 768), (1000, 600), (2048, 858), (7680, 4320)]
FPS = [None, 0, 12, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 120, 240, 241]
SIZES = [None, 2160, 1440, 1080, 720, 480]


def ask_node(cases: dict) -> dict:
    run = subprocess.run(["node", str(WEB / "tests/convert-grid.mjs")], input=json.dumps(cases), capture_output=True, text=True, timeout=120, cwd=WEB)
    assert run.returncode == 0, run.stderr[-800:]
    return json.loads(run.stdout)


@needs_node
def test_the_website_and_the_server_work_out_the_conversion_factor_the_same(tmp_path):
    factors = [{"w": w, "h": h, "fps": fps, "out": out} for (w, h), fps in itertools.product(SHAPES, FPS)
               for out in (None, (1920, 1080), (1280, 720), (3840, 2160), (640, 360))]
    got = ask_node({"factors": factors, "works": []})["factors"]
    mine = [ffmpeg.convert_factor(c["w"], c["h"], c["fps"], tuple(c["out"]) if c["out"] else None) for c in factors]
    wrong = [(c, a, b) for c, a, b in zip(factors, got, mine) if a != b]
    assert not wrong, wrong[:5]


@needs_node
def test_the_website_and_the_server_agree_on_which_file_types_take_a_picture_as_it_is_and_on_the_work_of_making_it_again():
    works = [{"w": w, "h": h, "fps": fps, "codec": codec, "source": source, "format": fmt, "size": size}
             for (w, h), fps, codec, source, fmt, size in itertools.product([(1920, 1080), (3840, 2160), (1080, 1920), (1440, 1080)], [None, 30, 60],
                                                                          CODECS, SOURCES, FORMATS, [None, 720])]
    got = ask_node({"factors": [], "works": works})["works"]
    wrong = []
    for c, site in zip(works, got):
        ext, src = f".{c['format']}", f".{c['source']}"
        copies = ffmpeg.picture_copies(c["codec"], src, ext)
        if c["size"] is None:
            # the same size: copied, or made again, or (the codec is not known to the browser) not known yet
            if site["encode"] != (None if copies is None else not copies):
                wrong.append((c, site, copies))
            continue
        # a new size: made again unless it already is that size; the work is the factor of the result
        info = ffmpeg.ProbeResult("x", 10.0, 0, 2, "aac", True, c["codec"], src, video_width=c["w"], video_height=c["h"], video_fps=c["fps"] or 0.0)
        encode, factor = ffmpeg.picture_work(info, ext, ffmpeg.VideoOut(short_side=c["size"]))
        if c["codec"] is None:  # (the server knows the codec once it has read the file; the website only has its guess)
            encode = site["encode"]
        if site["encode"] != encode or (encode and site["factor"] != factor):
            wrong.append((c, site, (encode, factor)))
    assert not wrong, f"{len(wrong)} of {len(works)}: {wrong[:5]}"


# ------------------------------------------------------------------------------------------------------------------ the price of a job

def _off_gaps(T: float, parts: list[dict], off: dict) -> list[dict]:
    """The sections of one track covering 0..T: `parts` (sorted, not overlapping) with `off` sections in the gaps, as the editor holds them."""
    out, at = [], 0.0
    for p in sorted(parts, key=lambda p: p["s"]):
        if p["s"] > at:
            out.append({"s": at, "e": p["s"], "hard": False, "auto": False, **off})
        out.append({"hard": False, "auto": False, **p})
        at = p["e"]
    if at < T:
        out.append({"s": at, "e": T, "hard": False, "auto": False, **off})
    return out


def _tracks(T, voice=(), echo=(), level=(), subs=()):
    return {"voice": _off_gaps(T, list(voice), {"mode": 0, "str": 2, "kind": "paid", "rem": 0, "db": None}),
            "echo": _off_gaps(T, list(echo), {"on": 0}),
            "level": _off_gaps(T, list(level), {"lvl": 0, "lufs": None}),
            "subs": _off_gaps(T, list(subs), {"q": 0, "lang": "auto", "tr": []})}


def noise(s, e, rem=0, kind="paid", st=2):
    return {"s": s, "e": e, "mode": 1, "str": st, "kind": kind, "rem": rem, "db": None}


def rebuild(s, e):
    return {"s": s, "e": e, "mode": 2, "str": 2, "kind": "paid", "rem": 0, "db": None}


def echo(s, e):
    return {"s": s, "e": e, "on": 1}


def lvl(s, e, lvl=2, lufs=None):
    return {"s": s, "e": e, "lvl": lvl, "lufs": lufs}


def sub(s, e, q=2, lang="auto", tr=()):
    return {"s": s, "e": e, "q": q, "lang": lang, "tr": list(tr)}


def price_cases() -> dict[str, tuple[float, dict]]:
    """Track sets as the editor holds them, each with its recording's length."""
    c: dict[str, tuple[float, dict]] = {}
    c["whole file: light noise removal + Leveling"] = (600.0, _tracks(600.0, [noise(0, 600.0, 1)], level=[lvl(0, 600.0)]))
    c["whole file, the editor's usual start (Auto noise removal, Leveling)"] = (3600.0, _tracks(3600.0, [noise(0, 3600.0)], level=[lvl(0, 3600.0)]))
    c["several parts of one step round up once over their union"] = (100.0, _tracks(100.0, [noise(0, 10.2, 1), noise(20, 30.7, 1), noise(40, 50.1, 1)]))
    c["heavy noise removal"] = (900.0, _tracks(900.0, [noise(0, 450.0, 2), noise(600, 900.0, 2)]))
    c["light and heavy parts, each step rounded on its own"] = (300.0, _tracks(300.0, [noise(0, 100.3, 1), noise(100.3, 200.6, 2), noise(220, 300.0, 1)]))
    c["auto, light and heavy side by side (auto is held as light)"] = (400.0, _tracks(400.0, [noise(0, 100.5, 0), noise(100.5, 200.25, 1), noise(250, 330.5, 2)]))
    c["Voice rebuild parts"] = (1200.0, _tracks(1200.0, [rebuild(10, 70.4), rebuild(300, 420.2)]))
    c["Voice rebuild with noise removal around it"] = (500.0, _tracks(500.0, [noise(0, 100, 1), rebuild(100, 250.5), noise(250.5, 500, 2)]))
    c["echo removal on parts"] = (800.0, _tracks(800.0, echo=[echo(5.5, 95.1), echo(400, 700.7)]))
    c["echo removal on the whole file"] = (2400.0, _tracks(2400.0, echo=[echo(0, 2400.0)]))
    c["Leveling on parts (Podcast, YouTube, Keep loudness, a custom loudness)"] = (
        500.0, _tracks(500.0, level=[lvl(0, 100.2, 1), lvl(100.2, 200.4, 2), lvl(200.4, 300.1, 4), lvl(300.1, 400.9, 9, -18)]))
    c["free (browser) noise parts cost nothing, paid ones beside them do"] = (
        600.0, _tracks(600.0, [noise(0, 200.5, 1, "free", 3), noise(200.5, 400.25, 1), noise(400.25, 600.0, 2, "free", 1)]))
    c["only free noise parts"] = (300.0, _tracks(300.0, [noise(0, 300.0, 0, "free", 2)]))
    c["Standard subtitles on the whole file"] = (1800.0, _tracks(1800.0, subs=[sub(0, 1800.0, 2, "en")]))
    c["Accurate subtitles with two translations"] = (1000.0, _tracks(1000.0, subs=[sub(0, 1000.0, 3, "en", ["es", "fr"])]))
    c["free subtitles"] = (700.0, _tracks(700.0, subs=[sub(0, 350.5, 1, "en"), sub(350.5, 700.0, 1, "auto")]))
    c["subtitle parts: Standard, Accurate and free, translations (the spoken language never counts)"] = (
        900.0, _tracks(900.0, subs=[sub(0, 100.4, 2, "en", ["en", "de"]), sub(100.4, 300.9, 3, "auto", ["es"]), sub(300.9, 500.2, 1, "fr"),
                                    sub(500.2, 900.0, 3, "de", ["en", "es", "fr"])]))
    c["subtitle parts under a second cost a whole second"] = (60.0, _tracks(60.0, [noise(10, 10.2, 1)], subs=[sub(5, 5.3, 2), sub(20, 20.01, 3, "en", ["es"])]))
    c["fractional part ends"] = (
        120.0, _tracks(120.0, [noise(0.1, 33.333, 1), noise(40.4, 66.666, 2)], [echo(1.234, 9.876)], [lvl(2.5, 99.99)], [sub(0.7, 59.999, 2, "en", ["es"])]))
    c["a recording of 58:19 with everything on"] = (
        3499.0, _tracks(3499.0, [noise(0, 1200, 1), rebuild(1200, 1500.5), noise(1500.5, 3499.0, 2)], [echo(0, 3499.0)], [lvl(0, 3499.0, 3)],
                        [sub(0, 3499.0, 3, "en", ["es", "fr", "de"])]))
    c["a recording of 58:19.4 (not a whole second), parts ending at its end"] = (
        3499.4, _tracks(3499.4, [noise(0, 3499.4, 1)], echo=[echo(100, 3499.4)], level=[lvl(0, 3499.4)], subs=[sub(0, 3499.4, 2)]))
    c["a part of a whole number of seconds is not rounded up"] = (120.0, _tracks(120.0, [noise(10, 40, 1)], level=[lvl(0, 120.0)]))
    c["everything at once on a two hour recording"] = (
        7200.0, _tracks(7200.0, [noise(0, 1000.7, 1), noise(1000.7, 2000, 2, "free", 2), rebuild(2100, 2600.5), noise(3000, 7200.0, 0)],
                        [echo(0, 500.5), echo(4000, 4100.25)], [lvl(0, 7200.0, 1)],
                        [sub(0, 3600.5, 2, "en", ["es"]), sub(3600.5, 7200.0, 3, "en", ["es", "de"])]))
    return c


def _run_prices(cases: dict, converts: list[dict] | None = None) -> dict:
    from acw.api import billing_routes

    run = subprocess.run(["node", str(WEB / "tests/price-grid.mjs")], capture_output=True, text=True, timeout=120, cwd=WEB,
                         input=json.dumps({"pricing": billing_routes.get_pricing(), "converts": converts or [],
                                           "cases": [{"T": T, "tracks": tracks} for T, tracks in cases.values()]}))
    assert run.returncode == 0, run.stderr[-800:]
    return json.loads(run.stdout)


def _server_price(T: float, timeline: dict) -> dict:
    from acw import jobs

    layers = timeline["layers"]
    kind = "transcribe" if list(layers) == ["subtitles"] else "denoise_transcribe" if layers.get("subtitles") else "denoise"
    return jobs.estimate_timeline(kind, None, None, T, {}, timeline)


def _line(x: dict) -> tuple:
    return (x["item"], x["seconds"], round(x["fixedSeconds"], 3), x["micros"])


@needs_node
def test_the_prices_the_website_works_out_for_tracks_are_the_servers_to_the_micro_line_by_line_and_in_total(monkeypatch):
    from acw import broker

    monkeypatch.setattr(broker, "rebuild_available", lambda live=False: True)
    cases = price_cases()
    assert len(cases) >= 20
    got = _run_prices(cases)["cases"]
    wrong = []
    for (name, (T, _)), site in zip(cases.items(), got):
        server = _server_price(T, site["timeline"])
        theirs = [_line(x) for x in server["lines"]]
        mine = [_line(x) for x in site["price"]["lines"] if not x.get("free")]
        # (compared as sets: the website's list puts Noise removal first, the server's its own order; every line's number must match)
        if sorted(theirs) != sorted(mine) or site["price"]["micros"] != server["estCostMicros"]:
            wrong.append((name, {"site": mine, "site_total": site["price"]["micros"]}, {"server": theirs, "server_total": server["estCostMicros"]}))
    assert not wrong, f"{len(wrong)} of {len(cases)}:\n" + "\n".join(map(str, wrong))


@needs_node
def test_a_part_end_in_a_fraction_of_a_millisecond_is_priced_the_same_by_the_website_and_the_server(monkeypatch):
    from acw import broker

    monkeypatch.setattr(broker, "rebuild_available", lambda live=False: True)
    cases = {
        "sub-millisecond ends": (200.0, _tracks(200.0, [noise(0, 10.0004, 1), noise(50.0006, 60.0009, 2)], level=[lvl(0.0004, 30.0001)])),
        "the last end a hair before the recording's end": (600.0, _tracks(600.0, [noise(0, 599.9997, 1)], level=[lvl(0, 599.9996)], subs=[sub(0, 599.9997, 2, "en")])),
        "the last end a hair before the end of a fractional recording": (3499.4, _tracks(3499.4, [noise(100, 3499.3997, 2)], echo=[echo(0.0002, 3499.3996)])),
        "starts a hair after a whole second": (120.0, _tracks(120.0, [noise(10.0004, 40.0, 1), noise(60.0006, 90.0004, 2)], subs=[sub(10.0003, 40.0004, 3, "en", ["es"])])),
    }
    for (name, (T, _)), site in zip(cases.items(), _run_prices(cases)["cases"]):
        server = _server_price(T, site["timeline"])
        assert sorted(_line(x) for x in site["price"]["lines"]) == sorted(_line(x) for x in server["lines"]), name
        assert site["price"]["micros"] == server["estCostMicros"], (name, site["price"]["micros"], server["estCostMicros"])


@needs_node
def test_every_free_part_the_website_lists_costs_nothing_and_the_server_lists_no_line_for_it(monkeypatch):
    from acw import broker

    monkeypatch.setattr(broker, "rebuild_available", lambda live=False: True)
    cases = {k: v for k, v in price_cases().items() if "free" in k}
    got = _run_prices(cases)["cases"]
    assert got
    for (name, (T, _)), site in zip(cases.items(), got):
        free = [x for x in site["price"]["lines"] if x.get("free")]
        assert free and all(x["micros"] == 0 and x["fixedSeconds"] == 0 for x in free), name
        assert not {x["item"] for x in _server_price(T, site["timeline"])["lines"]} & {"denoise_free", "transcribe_free"}, name


@needs_node
def test_the_conversion_price_of_the_website_is_the_servers_for_every_factor_and_length():
    from acw import job_convert
    from acw.billing import pricing

    factors = [1, 1.0, 1.25, 1.5, 1.76, 2, 2.01, 2.3333, 3.14159, 4, 4.5, 8, 16.01]
    lengths = [0, 0.2, 1, 59.4, 60, 61, 599.9, 600, 3499, 3499.4, 7200, 14400]
    converts = [{"factor": f, "seconds": s} for f in factors for s in lengths]
    got = _run_prices({}, converts)["converts"]
    rate = pricing.rate("convert", "standard").micros_per_minute
    mine = [job_convert.convert_cost({"convert_rate": rate, "picture_encode": True, "convert_factor": c["factor"]}, c["seconds"]) for c in converts]
    wrong = [(c, a, b) for c, a, b in zip(converts, got, mine) if a != b]
    assert not wrong, f"{len(wrong)} of {len(converts)}: {wrong[:8]}"
