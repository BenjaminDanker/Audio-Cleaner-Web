"""Converts a split's labels to the current scheme (labelscheme.py) from the PESQ scores already stored in them: the old
label_noise (0 = off/clean, 1..4 = gentle..max) becomes label_strength (0..3), and label_heavy is added from the measured
noise figure. usage: relabel.py <split>   (keeps the previous file as labels_v1.jsonl)"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labelscheme import heavy_label
B = os.path.expanduser("~/audio-cleaner/research/auto-settings")
S = sys.argv[1]
d = f"{B}/work/{S}/feat"
stats = {}
for l in open(f"{d}/stats.jsonl"):
    if l.strip():
        s = json.loads(l); stats.setdefault(s["id"], s)
old = [json.loads(l) for l in open(f"{d}/labels.jsonl") if l.strip()]
if not os.path.exists(f"{d}/labels_v1.jsonl"):
    os.rename(f"{d}/labels.jsonl", f"{d}/labels_v1.jsonl")
mismatch = 0
with open(f"{d}/labels.jsonl", "w") as f:
    for r in old:
        r2 = {k: v for k, v in r.items() if k != "label_noise"}
        r2["label_strength"] = max(int(r["label_noise"]) - 1, 0)
        r2["label_heavy"] = heavy_label(stats[r["id"]]["noiseDb"])
        r2["noise_db"] = stats[r["id"]]["noiseDb"]
        f.write(json.dumps(r2) + "\n")
print(S, len(old), "relabelled; heavy=1 in", sum(json.loads(l)["label_heavy"] == 1 for l in open(f"{d}/labels.jsonl")))
