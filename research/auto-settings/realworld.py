"""Real recordings (Common Voice 17.0 English test, CC0): DNSMOS OVRL of the input and of the noise remover's output at each
limit, scored for the NOISE decision only (strength: gentle/balanced/strong/max; the voice rebuild is not run here, so a clip
that the rules would rebuild is still scored on its noise removal). usage: realworld.py <stats.jsonl> <rules.jsonl> <new.jsonl> <rawdir>"""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
sys.path.insert(0, "/home/silver/audio-cleaner/research/auto-settings/src")
from dnsmos import dnsmos16
B = "/home/silver/audio-cleaner/research/auto-settings"
stats_p, rules_p, new_p, raw_dir = sys.argv[1:5]
ATTEN = [12, 24, 40, 100]
ATT_OF = {"gentle": 12, "balanced": 24, "strong": 40, "max": 100}


def first(path):
    out = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l); out.setdefault(r["id"], r)
    return out


rules = {k: v["rules"] for k, v in first(rules_p).items()}
new = first(new_p)
rows = []
for cid, st in first(stats_p).items():
    x, _ = sf.read(f"{B}/clips/cv/{cid}.wav", dtype="float32")
    x16 = resample_poly(x, 1, 3)
    n = len(x)
    raw = np.fromfile(f"{raw_dir}/{cid}.f32", dtype=np.float32)
    s_in = dnsmos16(x16)
    s_out = {a: dnsmos16(resample_poly(raw[k * n:(k + 1) * n], 1, 3)) for k, a in enumerate(ATTEN)}
    r = rules[cid]
    cur_att = ATT_OF[r["strength"]] if r else None
    d = new[cid]
    new_att = ATT_OF[d["strength"]]
    rows.append(dict(id=cid, input=s_in["OVRL"], cur_att=cur_att, new_att=new_att,
                     outs={str(a): s_out[a]["OVRL"] for a in ATTEN}, in_sig=s_in["SIG"], in_bak=s_in["BAK"],
                     sig={str(a): s_out[a]["SIG"] for a in ATTEN}, bak={str(a): s_out[a]["BAK"] for a in ATTEN}))
json.dump(rows, open(f"{B}/out/realworld_rows.json", "w"))


def score(choice_att, row):
    return row["input"] if choice_att is None else row["outs"][str(choice_att)]


res = {"n": len(rows), "input_OVRL_mean": round(float(np.mean([r["input"] for r in rows])), 3)}
methods = {
    "current_rules": lambda r: r["cur_att"],
    "new_model": lambda r: r["new_att"],
    "no_noise_removal (input)": lambda r: None,
    "always_gentle_12dB": lambda r: 12,
    "always_balanced_24dB": lambda r: 24,
    "always_strong_40dB": lambda r: 40,
    "always_max_100dB": lambda r: 100,
}
for name, f in methods.items():
    deltas = np.array([score(f(r), r) - r["input"] for r in rows])
    res[name] = dict(mean_delta_OVRL=round(float(deltas.mean()), 4), improved_by_0_02=round(float(np.mean(deltas > 0.02)), 4),
                     worse_by_0_05=round(float(np.mean(deltas < -0.05)), 4))
    counts = {}
    for r in rows:
        k = str(f(r)); counts[k] = counts.get(k, 0) + 1
    res[name]["choices"] = counts
diff = [r for r in rows if r["cur_att"] != r["new_att"]]
res["disagree_n"] = len(diff)
for name in ("current_rules", "new_model"):
    f = methods[name]
    dd = np.array([score(f(r), r) - r["input"] for r in diff]) if diff else np.array([0.0])
    res[f"disagree_{name}_mean_delta"] = round(float(dd.mean()), 4)
    res[f"disagree_{name}_improved_by_0_02"] = round(float(np.mean(dd > 0.02)), 4)
json.dump(res, open(f"{B}/out/realworld_report.json", "w"), indent=1)
print(json.dumps(res, indent=1))
