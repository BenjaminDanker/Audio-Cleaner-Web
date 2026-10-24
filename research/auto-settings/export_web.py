"""Write the automatic settings model for both products from out/reco-model.json (train.py), in one go:

    web/src/reco-model-data.js          read by the website (web/src/reco-model.js)
    common/acw_common/reco_model.json   read by the desktop app (common/acw_common/reco_model.py)

Both hold the same trees: the heavy-remover head is left out (the products keep that as a rule), so is the restore-highs head
(the products no longer have that step: a recording that is missing its highs gets the voice rebuilt; train.py still trains it),
and the numbers are rounded to 6 significant digits. The format is version 3 (2 had the restore head).

usage: export_web.py [<out/reco-model.json> [<web/src/reco-model-data.js> [<common/acw_common/reco_model.json>]]]
Without arguments: ~/audio-cleaner/research/auto-settings/out/reco-model.json into this repository's two files. With --check (first
argument) nothing is written: the exit status says whether both files already hold exactly what would be written."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FORMAT_VERSION = 3  # reco_model.VERSION and MODEL.version in web/src/reco-model.js
JS_HEADER = ("// Generated: the trees of the automatic settings model (research/auto-settings, train.py -> out/reco-model.json), with the\n"
             "// heavy-remover and restore heads left out (the first stays a rule, the second is no longer used) and the numbers rounded to 6\n"
             "// significant digits. Read by src/reco-model.js.\n"
             "// To refresh it: research/auto-settings/export_web.py. Do not edit by hand.\n")


def r(x):
    if isinstance(x, float):
        return float(f"{x:.6g}")
    if isinstance(x, list):
        return [r(v) for v in x]
    if isinstance(x, dict):
        return {k: r(v) for k, v in x.items()}
    return x


def render(src: Path) -> tuple[str, str]:
    """(the website's file, the desktop app's file) for the model in `src`."""
    m = json.load(open(src))
    for k in ("heavy", "restore", "rebuild_uncertain", "restore_uncertain"):
        m.pop(k, None)
    m["version"] = FORMAT_VERSION
    payload = json.dumps(r(m), separators=(",", ":"))
    return JS_HEADER + "export default " + payload + ";\n", payload + "\n"


def main(argv: list[str]) -> int:
    check = bool(argv) and argv[0] == "--check"
    if check:
        argv = argv[1:]
    src = Path(argv[0]) if len(argv) > 0 else Path.home() / "audio-cleaner" / "research" / "auto-settings" / "out" / "reco-model.json"
    web = Path(argv[1]) if len(argv) > 1 else REPO / "web" / "src" / "reco-model-data.js"
    desktop = Path(argv[2]) if len(argv) > 2 else REPO / "common" / "acw_common" / "reco_model.json"
    js, py = render(src)
    if check:
        same = web.is_file() and desktop.is_file() and web.read_text() == js and desktop.read_text() == py
        print("up to date" if same else "out of date")
        return 0 if same else 1
    web.write_text(js)
    desktop.write_text(py)
    print(f"wrote {web} and {desktop}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
