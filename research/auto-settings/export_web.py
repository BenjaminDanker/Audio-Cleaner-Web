"""Write web/src/reco-model-data.js from out/reco-model.json (train.py): the heavy-remover head is left out (the product keeps
that as a rule) and the numbers are rounded to 6 significant digits.
usage: export_web.py <out/reco-model.json> <web/src/reco-model-data.js>"""
import json
import sys

src, dst = sys.argv[1:3]
m = json.load(open(src))
for k in ("heavy", "rebuild_uncertain", "restore_uncertain"):
    m.pop(k, None)


def r(x):
    if isinstance(x, float):
        return float(f"{x:.6g}")
    if isinstance(x, list):
        return [r(v) for v in x]
    if isinstance(x, dict):
        return {k: r(v) for k, v in x.items()}
    return x


with open(dst, "w") as f:
    f.write("// Generated: the trees of the automatic settings model (research/auto-settings, train.py -> out/reco-model.json), with the\n"
            "// heavy-remover head left out (that stays a rule) and the numbers rounded to 6 significant digits. Read by src/reco-model.js.\n"
            "// To refresh it: research/auto-settings/export_web.py. Do not edit by hand.\n"
            "export default " + json.dumps(r(m), separators=(",", ":")) + ";\n")
