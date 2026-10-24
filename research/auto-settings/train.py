"""Trains the automatic-settings heads on the measured features (the browser's Analyser.result() fields only):
  noise strength (4 classes: gentle, balanced, strong, max), heavy remover (yes/no), voice rebuild (yes/no), restore highs (yes/no).
Exports LightGBM trees to out/reco-model.json (version 2) and scores the held-out TEST split for the new model and for the
product's current rules (web/src/recommend.js, via src/rules.mjs). usage: train.py
reads work/{train,val,test}/feat/{stats,labels,rules}.jsonl (duplicates by id are read once)."""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import lightgbm as lgb

B = os.path.expanduser("~/audio-cleaner/research/auto-settings")
OUT = os.path.join(B, "out")
os.makedirs(OUT, exist_ok=True)
FEATS = ["floorDb", "levelDb", "peakDb", "silentFraction", "speechFraction", "bandDynDb", "hasSpeech",
         "bandwidthHz", "tiltDb", "dropDb", "clipFraction", "peak", "loudnessLufs", "noiseDb"]
STRENGTHS = ["gentle", "balanced", "strong", "max"]
STRENGTH_OF = {s: i for i, s in enumerate(STRENGTHS)}


def first_by_id(path, key="id"):
    out = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            out.setdefault(r[key], r)
    return out


def load(split):
    d = os.path.join(B, "work", split, "feat")
    stats = first_by_id(os.path.join(d, "stats.jsonl"))
    labels = first_by_id(os.path.join(d, "labels.jsonl"))
    rules = first_by_id(os.path.join(d, "rules.jsonl"))
    ids = [i for i in labels if i in stats and i in rules and labels[i]["label_heavy"] >= 0]
    X = np.array([[np.nan if stats[i][f] is None else float(stats[i][f]) for f in FEATS] for i in ids], np.float64)
    X[:, FEATS.index("hasSpeech")] = [float(bool(stats[i]["hasSpeech"])) for i in ids]
    labs = [labels[i] for i in ids]
    return dict(ids=ids, X=X, labs=labs, rules=[rules[i]["rules"] for i in ids],
                y_strength=np.array([l["label_strength"] for l in labs]),
                y_heavy=np.array([l["label_heavy"] for l in labs]),
                y_reb=np.array([l["label_rebuild"] for l in labs]),
                y_res=np.array([l["label_restore"] for l in labs]))


# current product rules, read from the rules file (rules.mjs output)
def cur_strength(r):  # index 0..3 ; a clip without a recommendation counts as gentle (never happens in practice)
    return STRENGTH_OF.get(r["strength"], 0) if r else 0


def cur_heavy(r): return int(bool(r and r.get("heavy")))
def cur_rebuild(r): return int(bool(r and r["rebuild"]))
def cur_restore(r): return int(bool(r and r["restoreHighs"]))


PARAMS = dict(learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, num_threads=1, verbose=-1, seed=7)
ROUNDS = 400


def fit(Xtr, ytr, Xva, yva, obj, num_class=None):
    p = dict(PARAMS, objective=obj)
    if num_class: p["num_class"] = num_class
    dtr = lgb.Dataset(Xtr, ytr, feature_name=FEATS); dva = lgb.Dataset(Xva, yva, reference=dtr)
    return lgb.train(p, dtr, ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])


def export_trees(m, num_class):
    dump = m.dump_model(num_iteration=m.best_iteration)

    def node(n):
        if "leaf_value" in n and "split_feature" not in n:
            return [round(n["leaf_value"], 6)]
        return [FEATS.index(FEATS[n["split_feature"]]), n["threshold"], int(n["default_left"]), node(n["left_child"]), node(n["right_child"])]
    trees = [node(t["tree_structure"]) for t in dump["tree_info"]]
    return dict(num_class=num_class, trees=trees)


def main():
    tr, va, te = load("train"), load("val"), load("test")
    print("sizes train/val/test", len(tr["ids"]), len(va["ids"]), len(te["ids"]), flush=True)
    m_s = fit(tr["X"], tr["y_strength"], va["X"], va["y_strength"], "multiclass", num_class=4)
    m_h = fit(tr["X"], tr["y_heavy"], va["X"], va["y_heavy"], "binary")
    m_r = fit(tr["X"], tr["y_reb"], va["X"], va["y_reb"], "binary")
    m_t = fit(tr["X"], tr["y_res"], va["X"], va["y_res"], "binary")
    P = dict(strength=m_s.predict(te["X"], num_iteration=m_s.best_iteration),
             heavy=m_h.predict(te["X"], num_iteration=m_h.best_iteration),
             rebuild=m_r.predict(te["X"], num_iteration=m_r.best_iteration),
             restore=m_t.predict(te["X"], num_iteration=m_t.best_iteration))
    model = dict(version=2, features=FEATS, strength_classes=STRENGTHS,
                 noise=export_trees(m_s, 4), heavy=export_trees(m_h, 1),
                 rebuild=export_trees(m_r, 1), restore=export_trees(m_t, 1),
                 rebuild_uncertain=[0.4, 0.6], restore_uncertain=[0.4, 0.6])
    json.dump(model, open(os.path.join(OUT, "reco-model.json"), "w"), separators=(",", ":"))
    print("model bytes", os.path.getsize(os.path.join(OUT, "reco-model.json")),
          "best iters strength/heavy/rebuild/restore", m_s.best_iteration, m_h.best_iteration, m_r.best_iteration, m_t.best_iteration, flush=True)
    for k, v in P.items():
        np.save(os.path.join(OUT, f"test_probs_{k}.npy"), v)
    json.dump(dict(ids=te["ids"]), open(os.path.join(OUT, "test_ids.json"), "w"))
    report(te, P)


def confusion(y, p, k):
    M = np.zeros((k, k), int)
    for a, b in zip(y, p): M[a, b] += 1
    return M.tolist()


def binary(y, pred):
    tn = int(np.sum((y == 0) & (pred == 0))); fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0))); tp = int(np.sum((y == 1) & (pred == 1)))
    return dict(accuracy=round((tp + tn) / len(y), 4), tp=tp, fp=fp, fn=fn, tn=tn,
                false_positive_rate=round(fp / max(1, fp + tn), 4), recall=round(tp / max(1, tp + fn), 4))


def report(te, P):
    labs = te["labs"]; rules = te["rules"]; n = len(labs)
    res = {"n_test": n}
    # 1. noise strength (4 levels)
    ps = P["strength"].argmax(1); cs = np.array([cur_strength(r) for r in rules]); ys = te["y_strength"]
    pesq = np.array([[l["pesq_12"], l["pesq_24"], l["pesq_40"], l["pesq_100"]] for l in labs], float)
    pesq = np.where(np.isfinite(pesq), pesq, -1.0)
    best = pesq.max(1); pin = np.array([l["pesq_in"] for l in labs], float)
    def regret(p):  # PESQ-WB points lost against the best of the four outputs (0 = as good as the best)
        return float(np.mean(best - pesq[np.arange(n), p]))
    def within1(p): return float(np.mean(np.abs(p - ys) <= 1))
    res["strength"] = dict(
        new=dict(exact_accuracy=round(float(np.mean(ps == ys)), 4), within_one_step=round(within1(ps), 4),
                 mean_pesq_loss_vs_best=round(regret(ps), 4), share_gentle=round(float(np.mean(ps == 0)), 4)),
        current=dict(exact_accuracy=round(float(np.mean(cs == ys)), 4), within_one_step=round(within1(cs), 4),
                     mean_pesq_loss_vs_best=round(regret(cs), 4), share_gentle=round(float(np.mean(cs == 0)), 4)),
        label_share=[round(float(np.mean(ys == k)), 4) for k in range(4)],
        mean_pesq_loss_of_leaving_input_alone=round(float(np.mean(best - pin)), 4),
        confusion_new=confusion(ys, ps, 4), confusion_current=confusion(ys, cs, 4), classes=STRENGTHS)
    # 2. heavy remover (yes/no)
    yh = te["y_heavy"]
    ph = (P["heavy"] > 0.5).astype(int); ch = np.array([cur_heavy(r) for r in rules])
    res["heavy"] = dict(new=binary(yh, ph), current=binary(yh, ch), label_share=round(float(np.mean(yh)), 4),
                        confusion_new=confusion(yh, ph, 2), confusion_current=confusion(yh, ch, 2))
    res["strength_and_heavy_joint"] = dict(
        new=round(float(np.mean((ps == ys) & (ph == yh))), 4), current=round(float(np.mean((cs == ys) & (ch == yh))), 4))
    # 3. voice rebuild (uncertain band 0.4-0.6 falls back to the current rule for the new model)
    yr = te["y_reb"]
    pr_new = (P["rebuild"] > 0.5).astype(int); pr_cur = np.array([cur_rebuild(r) for r in rules])
    unc = (P["rebuild"] >= 0.4) & (P["rebuild"] <= 0.6)
    pr_fb = np.where(unc, pr_cur, pr_new)
    res["rebuild"] = dict(new=binary(yr, pr_new), current=binary(yr, pr_cur),
                          new_with_fallback=dict(binary(yr, pr_fb), uncertain_share=round(float(np.mean(unc)), 4)),
                          confusion_new=confusion(yr, pr_new, 2), confusion_current=confusion(yr, pr_cur, 2))
    dull = np.array([bool(l["label_dull"]) for l in labs])
    no_reverb = np.array([not l["reverb"] for l in labs])
    no_noise = np.array([l["snr"] is None for l in labs])
    no_clip = np.array([l["clip_frac_set"] == 0 for l in labs])
    codec_none = np.array([l["codec"] == "none" for l in labs])
    strict = dull & (yr == 0) & no_reverb & no_noise & codec_none & no_clip  # muffled, nothing else wrong: the listening samples
    broad = dull & (yr == 0)                                                  # muffled and no other ground-truth damage
    res["rebuild_false_positives_muffled_clean"] = dict(
        strict=dict(n=int(strict.sum()), new=round(float(np.mean(pr_new[strict])), 4), current=round(float(np.mean(pr_cur[strict])), 4)),
        broad=dict(n=int(broad.sum()), new=round(float(np.mean(pr_new[broad])), 4), current=round(float(np.mean(pr_cur[broad])), 4)))
    clean_dry = (yr == 0) & no_reverb & no_noise & codec_none & no_clip & (~dull)
    res["rebuild_false_positives_clean_dry"] = dict(n=int(clean_dry.sum()), new=round(float(np.mean(pr_new[clean_dry])), 4),
                                                    current=round(float(np.mean(pr_cur[clean_dry])), 4))
    # 4. restore highs
    ys_ = te["y_res"]
    pt_new = (P["restore"] > 0.5).astype(int); pt_cur = np.array([cur_restore(r) for r in rules])
    res["restore"] = dict(new=binary(ys_, pt_new), current=binary(ys_, pt_cur),
                          confusion_new=confusion(ys_, pt_new, 2), confusion_current=confusion(ys_, pt_cur, 2))
    # 5. groups
    groups = {}
    for name, mask in [("dry_clean_no_noise", lambda l: not l["reverb"] and l["snr"] is None and l["codec"] == "none"),
                       ("reverb_rt60_under_0.5", lambda l: l["reverb"] and l["rt60"] < 0.5),
                       ("reverb_rt60_0.5_plus", lambda l: l["reverb"] and l["rt60"] >= 0.5),
                       ("noise_only", lambda l: l["snr"] is not None),
                       ("bandlimited_or_codec", lambda l: l["codec"] != "none")]:
        mk = np.array([mask(l) for l in labs])
        groups[name] = dict(n=int(mk.sum()),
                            rebuild_rate_new=round(float(np.mean(pr_new[mk])), 4), rebuild_rate_current=round(float(np.mean(pr_cur[mk])), 4),
                            gt_rebuild_rate=round(float(np.mean(yr[mk])), 4),
                            restore_rate_new=round(float(np.mean(pt_new[mk])), 4), restore_rate_current=round(float(np.mean(pt_cur[mk])), 4),
                            gt_restore_rate=round(float(np.mean(ys_[mk])), 4),
                            heavy_rate_new=round(float(np.mean(ph[mk])), 4), heavy_rate_current=round(float(np.mean(ch[mk])), 4),
                            strength_mean_new=round(float(np.mean(ps[mk])), 3), strength_mean_current=round(float(np.mean(cs[mk])), 3))
    res["groups"] = groups
    json.dump(res, open(os.path.join(OUT, "test_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("strength", "heavy", "rebuild", "restore")}, indent=1)[:3000])
    print("strength", {k: v for k, v in res["strength"].items() if not k.startswith("confusion")})
    print("heavy", res["heavy"]["new"], res["heavy"]["current"])
    print("rebuild", res["rebuild"]["new"], res["rebuild"]["current"], res["rebuild"]["new_with_fallback"])
    print("restore", res["restore"]["new"], res["restore"]["current"])


if __name__ == "__main__":
    main()
