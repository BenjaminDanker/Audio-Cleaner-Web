"""Labels per clip: noise strength by outcome (PESQ-WB of the site's own noise remover at four limits against the clean
reference, see labelscheme.py), heavy remover by the server's rule on the measured noise figure, rebuild and restore-highs by
construction (gen.py ground truth). usage: label.py <split> <stats.jsonl> <rawdir> <out.jsonl>
A clip whose raw output is missing is skipped and logged (its id goes to <out>.skipped.txt). Duplicate stats rows (same id) are
read once."""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from pesq import pesq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labelscheme import ATTEN, strength_label, heavy_label

ROOT = os.environ.get("AC_ROOT", os.path.expanduser("~/audio-cleaner/research/auto-settings"))
split, stats_path, raw_dir, out_path = sys.argv[1:5]
clip_dir = os.path.join(ROOT, "clips", split)


def score(ref, deg):
    try:
        return float(pesq(16000, ref, deg, "wb"))
    except Exception:
        return float("nan")


rows, skipped, seen = [], [], set()
for line in open(stats_path):
    if not line.strip():
        continue
    st = json.loads(line)
    cid = st["id"]
    if cid in seen:
        continue
    seen.add(cid)
    raw_path = os.path.join(raw_dir, cid + ".f32")
    if not os.path.exists(raw_path):
        print("SKIP no raw output:", cid, flush=True)
        skipped.append(cid)
        continue
    meta = json.load(open(os.path.join(clip_dir, cid + ".json")))
    ref = np.load(os.path.join(clip_dir, cid + ".ref16.npy")).astype(np.float32) / 32767
    noisy, _ = sf.read(os.path.join(clip_dir, cid + ".wav"), dtype="float32")
    noisy16 = resample_poly(noisy, 1, 3)[: len(ref)]
    raw = np.fromfile(raw_path, dtype=np.float32)
    n = len(raw) // len(ATTEN)
    deg = {}
    for k, a in enumerate(ATTEN):
        y = raw[k * n:(k + 1) * n]
        deg[a] = resample_poly(y, 1, 3)[: len(ref)]
    scores = {a: score(ref, deg[a]) for a in ATTEN}
    os.remove(raw_path)  # the 4 outputs are big: only the scores are kept
    s_in = score(ref, noisy16)
    rows.append(dict(id=cid, label_strength=strength_label(meta["snr_db"] is None, s_in, scores),
                     label_heavy=heavy_label(st.get("noiseDb")), noise_db=st.get("noiseDb"),
                     pesq_in=s_in, pesq_12=scores[12], pesq_24=scores[24], pesq_40=scores[40], pesq_100=scores[100],
                     label_rebuild=int(meta["gt_rebuild"]), label_restore=int(meta["gt_restore_highs"]),
                     label_dull=int(meta["gt_dull"]), snr=meta["snr_db"], family=meta["noise_family"], reverb=meta["reverb"],
                     rt60=meta["rt60_target"], codec=meta["codec"], codec_value=meta["codec_value"], cut_hz=meta["cut_hz"],
                     dull_db=meta["dull_db"], clip_frac_set=meta["clip_frac_set"], speaker=meta["speaker"]))
    if len(rows) % 100 == 0:
        print("labelled", len(rows), flush=True)
with open(out_path, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with open(out_path + ".skipped.txt", "w") as f:
    f.write("\n".join(skipped) + ("\n" if skipped else ""))
print(split, len(rows), "labelled", len(skipped), "skipped (no raw output)", flush=True)
