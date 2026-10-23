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
