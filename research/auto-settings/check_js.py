"""Agreement of the JS evaluator (out/reco-model.js via src/decide.mjs) with the Python model on the test split, and a PESQ
regret that counts the input as an option (the product's gentle output is judged against max(best output, input))."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load, cur_strength, cur_heavy, cur_rebuild, cur_restore
B = os.path.expanduser("~/audio-cleaner-train"); OUT = f"{B}/out"
te = load("test")
ids = json.load(open(f"{OUT}/test_ids.json"))["ids"]
assert ids == te["ids"]
js = {}
for l in open(f"{B}/work/test/feat/new.jsonl"):
    r = json.loads(l); js[r["id"]] = r
P = {k: np.load(f"{OUT}/test_probs_{k}.npy") for k in ["strength", "heavy", "rebuild", "restore"]}
n = len(ids)
js_strength = np.array([["gentle", "balanced", "strong", "max"].index(js[i]["strength"]) for i in ids])
py_strength = P["strength"].argmax(1)
js_heavy = np.array([int(js[i]["heavy"]) for i in ids]); py_heavy = (P["heavy"] > 0.5).astype(int)
js_reb = np.array([int(js[i]["rebuild"]) for i in ids])
js_res = np.array([int(js[i]["restoreHighs"]) for i in ids])
cur_r = te["rules"]
py_reb_fb = np.where((P["rebuild"] >= 0.4) & (P["rebuild"] <= 0.6), [cur_rebuild(r) for r in cur_r], (P["rebuild"] > 0.5).astype(int))
py_res_fb = np.where((P["restore"] >= 0.4) & (P["restore"] <= 0.6), [cur_restore(r) for r in cur_r], (P["restore"] > 0.5).astype(int))
prob_gap = max(float(np.max(np.abs(np.array([js[i]["strengthProbs"] for i in ids]) - P["strength"]))),
               float(np.max(np.abs(np.array([js[i]["probs"]["heavy"] for i in ids]) - P["heavy"]))),
               float(np.max(np.abs(np.array([js[i]["probs"]["rebuild"] for i in ids]) - P["rebuild"]))),
               float(np.max(np.abs(np.array([js[i]["probs"]["restoreHighs"] for i in ids]) - P["restore"]))))
res = dict(n=n,
           agreement=dict(strength=round(float(np.mean(js_strength == py_strength)), 6), heavy=round(float(np.mean(js_heavy == py_heavy)), 6),
                          rebuild_with_fallback=round(float(np.mean(js_reb == py_reb_fb)), 6),
                          restore_with_fallback=round(float(np.mean(js_res == py_res_fb)), 6)),
           max_abs_probability_gap=prob_gap)
ys = te["y_strength"]
pesq = np.array([[l["pesq_12"], l["pesq_24"], l["pesq_40"], l["pesq_100"]] for l in te["labs"]], float)
pesq = np.where(np.isfinite(pesq), pesq, -1.0)
pin = np.array([l["pesq_in"] for l in te["labs"]], float)
ref = np.maximum(pesq.max(1), pin)  # best of the outputs and the input
cs = np.array([cur_strength(r) for r in cur_r])
def loss(p): return float(np.mean(ref - pesq[np.arange(n), p]))
res["pesq_regret_vs_best_of_outputs_and_input"] = dict(new=round(loss(py_strength), 4), current=round(loss(cs), 4), input_alone=round(float(np.mean(ref - pin)), 4))
res["pesq_regret_vs_best_of_outputs_and_input_mean_by_class"] = {}
json.dump(res, open(f"{OUT}/js_check.json", "w"), indent=1)
print(json.dumps(res, indent=1))
