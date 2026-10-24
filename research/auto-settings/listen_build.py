"""Listening page: 25 clips (13 simulated test clips, 12 real Common Voice clips). For each clip: the original (loudness-matched
like the outputs), the output with the CURRENT rules' settings and the output with the NEW model's settings, produced with the
repo's own processing code (common/acw_common: Sidon for the voice rebuild, AP-BWE for restored highs; DeepFilterNet3 through
the site's streaming noise remover at the chosen strength). Output: out/listening/index.html + MP3 files (self-contained, no scripts).
usage: listen_build.py choose | render"""
import html, json, os, subprocess, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
import numpy as np, soundfile as sf
B = "/home/silver/audio-cleaner/research/auto-settings"
REPO = "/home/silver/repos/acw-desktop-merge"
LIST = f"{B}/out/listening"
WORK = f"{B}/work/listen"
ATT_OF = {"gentle": 12, "balanced": 24, "strong": 40, "max": 100}
STRENGTH_LABEL = {"gentle": "Gentle", "balanced": "Balanced", "strong": "Strong", "max": "Maximum"}
mode = sys.argv[1]


def first(path):
    out = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l); out.setdefault(r["id"], r)
    return out


def settings_text(d):
    """the product's wording for a decision dict {rebuild, restoreHighs, strength}"""
    bits = ["Voice rebuild"] if d["rebuild"] else [f"Denoising ({STRENGTH_LABEL[d['strength']]})"]
    if not d["rebuild"] and d["restoreHighs"]:
        bits.append("restore the highs")
    bits.append("leveling")
    return ", ".join(bits)


if mode == "choose":
    rng = np.random.default_rng(3)
    tl = first(f"{B}/work/test/feat/labels.jsonl")
    tr = {k: v["rules"] for k, v in first(f"{B}/work/test/feat/rules.jsonl").items()}
    tn = first(f"{B}/work/test/feat/new.jsonl")
    cr = {k: v["rules"] for k, v in first(f"{B}/work/cv/feat/rules.jsonl").items()}
    cn = first(f"{B}/work/cv/feat/new.jsonl")
    muff = [i for i, l in tl.items() if l["label_dull"] and not l["reverb"] and l["snr"] is None and l["codec"] == "none"
            and l["clip_frac_set"] == 0 and l["label_rebuild"] == 0]
    fp_cur = sorted(i for i in muff if tr[i] and tr[i]["rebuild"])
    fp_new = sorted(i for i in muff if tn[i]["rebuild"] and i not in fp_cur)
    ctrl = sorted(i for i in muff if not (tr[i] and tr[i]["rebuild"]) and not tn[i]["rebuild"])
    reverb = sorted(i for i, l in tl.items() if l["reverb"] and l["rt60"] >= 0.5 and l["snr"] is None)
    band = sorted(i for i, l in tl.items() if l["codec"] in ("phone", "lowpass", "opus") and l["snr"] is None)
    noisy = sorted(i for i, l in tl.items() if l["snr"] is not None and l["snr"] < 5)
    pick = []
    def take(pool, n, tag):
        pool = list(pool); rng.shuffle(pool)
        for i in pool[:n]:
            if i not in [p[0] for p in pick]:
                pick.append((i, tag))
    take(fp_cur, 4, "sim-muffled-clean (rebuild false positive)")
    take(fp_new, 1, "sim-muffled-clean (new model rebuilds)")
    take(ctrl, 2, "sim-muffled-clean (neither rebuilds)")
    take(reverb, 3, "sim-echoey-room")
    take(band, 3, "sim-band-limited")
    take(noisy, 3, "sim-noisy")
    sim_n = len(pick)
    cv_ids = sorted(cr)
    rng.shuffle(cv_ids)
    flagged = [i for i in cv_ids if cr[i] and cr[i]["rebuild"]]
    noisy_cv = [i for i in cv_ids if cr[i] and not cr[i]["rebuild"] and cr[i]["strength"] != "gentle"]
    other = [i for i in cv_ids if i not in flagged[:4] and i not in noisy_cv[:4]]
    real = [(i, "real-rebuild-flagged") for i in flagged[:4]] + [(i, "real-noise") for i in noisy_cv[:4]] + [(i, "real-other") for i in other[:2]]
    pick += real
    pick = pick[:25]
    chosen = []
    for i, tag in pick:
        src = f"{B}/clips/test/{i}.wav" if i.startswith("test") else f"{B}/clips/cv/{i}.wav"
        chosen.append(dict(id=i, tag=tag, src=src))
    os.makedirs(WORK, exist_ok=True)
    json.dump(chosen, open(f"{WORK}/chosen.json", "w"), indent=1)
    with open(f"{WORK}/manifest.json", "w") as f:
        json.dump([{"id": c["id"], "path": c["src"]} for c in chosen], f)
    print(len(chosen), "chosen;", sim_n, "simulated; muffled-clean pool", len(muff), "current rebuilds", len(fp_cur), "new rebuilds", len(fp_new), flush=True)


if mode == "render":
    import pyloudnorm as pyln
    sys.path.insert(0, f"{REPO}/common")
    from acw_common import restore as sidon, bwe
    chosen = json.load(open(f"{WORK}/chosen.json"))
    rules = {k: v["rules"] for k, v in first(f"{WORK}/rules.jsonl").items()}
    new = first(f"{WORK}/new.jsonl")
    tl = first(f"{B}/work/test/feat/labels.jsonl")
    os.makedirs(LIST, exist_ok=True)
    for old in os.listdir(LIST):
        if old.endswith(".mp3"):
            os.remove(os.path.join(LIST, old))
    FF = subprocess.run([sys.executable, "-c", "import imageio_ffmpeg as i;print(i.get_ffmpeg_exe())"],
                        capture_output=True, text=True).stdout.strip()

    def level(y):
        """loudness to -16 LUFS, peak-limited (leveling is on for every setting, so every version gets it)"""
        y = y.astype(np.float64)
        try:
            lo = pyln.normalize.loudness(y, pyln.Meter(48000).integrated_loudness(y), -16.0)
        except Exception:
            return y.astype(np.float32)
        pk = np.max(np.abs(lo)) + 1e-12
        if pk > 0.89:
            lo = lo * (0.89 / pk)
        return lo.astype(np.float32)

    subs = []
    def process(x, raw, n, d):
        """the product's settings: voice rebuild (Sidon) replaces noise removal; else noise removal at the strength, then
        restore highs (AP-BWE) when the recording is band-limited."""
        if d["rebuild"]:
            y = sidon.restore(x, f"{B}/models/sidon", threads=2, device="cpu")
        else:
            k = [12, 24, 40, 100].index(ATT_OF[d["strength"]])
            y = raw[k * n:(k + 1) * n]
            if d["restoreHighs"]:
                try:
                    y = bwe.extend(y, f"{B}/models/ap-bwe", threads=2, device="cpu")
                except Exception as ex:  # the product's generator file is not on this machine: nearest one that is
                    sub = "16k" if os.path.exists(f"{B}/models/ap-bwe/16kto48k.onnx") else "8k"
                    subs.append(f"generator 12k/24k file missing, used {sub}")
                    y = bwe.extend(y, f"{B}/models/ap-bwe", threads=2, device="cpu", model=sub)
        return level(y)

    def encode(y, path):
        tmp = path + ".wav"
        sf.write(tmp, y, 48000, subtype="PCM_16")
        subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", tmp, "-ac", "1", "-c:a", "libmp3lame",
                        "-b:a", "128k", path], check=True)
        os.remove(tmp)

    meta = []
    for k, c in enumerate(chosen):
        cid = c["id"]
        x, _ = sf.read(c["src"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        raw = np.fromfile(f"{WORK}/raw/{cid}.f32", dtype=np.float32)
        n = len(x)
        cur = rules[cid]; nw = new[cid]
        cur_d = dict(rebuild=bool(cur["rebuild"]), restoreHighs=bool(cur["restoreHighs"]), strength=cur["strength"], heavy=bool(cur.get("heavy")))
        new_d = dict(rebuild=bool(nw["rebuild"]), restoreHighs=bool(nw["restoreHighs"]), strength=nw["strength"], heavy=bool(nw["heavy"]))
        base = f"{k:02d}_{cid}"
        subs.clear()
        encode(level(x), f"{LIST}/{base}_original.mp3")
        encode(process(x, raw, n, cur_d), f"{LIST}/{base}_current.mp3")
        encode(process(x, raw, n, new_d), f"{LIST}/{base}_new.mp3")
        t = tl.get(cid)
        meta.append(dict(k=k, id=cid, tag=c["tag"], cur=cur_d, new=new_d, base=base, sim=cid.startswith("test"), notes=sorted(set(subs)),
                         truth=None if t is None else dict(dull=bool(t["label_dull"]), reverb=bool(t["reverb"]), rt60=t["rt60"],
                                                           codec=t["codec"], snr=t["snr"], gt_rebuild=bool(t["label_rebuild"]),
                                                           label_strength=STRENGTH_LABEL[["gentle", "balanced", "strong", "max"][t["label_strength"]]],
                                                           label_heavy=bool(t["label_heavy"]))))
        print("rendered", k, cid, flush=True)
    json.dump(meta, open(f"{LIST}/meta.json", "w"), indent=1)
    cards = []
    for m in meta:
        tr_ = m["truth"]
        if m["sim"]:
            what = ("Simulated from clean LibriTTS-R speech: " + ("muffled microphone (high-shelf cut), " if tr_["dull"] else "")
                    + (f"room echo (RT60 {tr_['rt60']:.2f} s), " if tr_["reverb"] else "")
                    + (f"codec {tr_['codec']}, " if tr_["codec"] != "none" else "")
                    + (f"noise at {tr_['snr']:.0f} dB SNR" if tr_["snr"] is not None else "no added noise") + ".")
            truth = (f"Ground truth: rebuild {'yes' if tr_['gt_rebuild'] else 'no'}; noise strength by the PESQ label: {tr_['label_strength']}.")
        else:
            what = "Real Common Voice recording (CC0). No ground truth."
            truth = ""
        same = "Same settings for both outputs." if (m["cur"] == m["new"]) else ""
        cards.append(f"""<section class="card">
<h2>{html.escape(m['tag'])}</h2>
<p class="what">{html.escape(what)}</p>
<div class="row"><div class="col"><h3>Original</h3><audio controls preload="metadata" src="{m['base']}_original.mp3"></audio></div>
<div class="col"><h3>Current rules</h3><audio controls preload="metadata" src="{m['base']}_current.mp3"></audio>
<p class="set">{html.escape(settings_text(m['cur']))}</p></div>
<div class="col"><h3>New model</h3><audio controls preload="metadata" src="{m['base']}_new.mp3"></audio>
<p class="set">{html.escape(settings_text(m['new']))}</p></div></div>
<p class="note">{html.escape(truth)} {html.escape(same)}</p>
</section>""")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Automatic settings: listening check</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1d1b; --muted:#5f5e58; --card:#ffffff; --line:#e3e0d7; --accent:#2f5d8a; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#17171a; --fg:#ececea; --muted:#a3a29b; --card:#222227; --line:#34343a; --accent:#8ab4e0; }} }}
:root[data-theme="dark"] {{ --bg:#17171a; --fg:#ececea; --muted:#a3a29b; --card:#222227; --line:#34343a; --accent:#8ab4e0; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }}
main {{ max-width:1000px; margin:0 auto; }}
h1 {{ font-size:1.5rem; margin:.2em 0; }}
.intro {{ color:var(--muted); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:14px 0; }}
.card h2 {{ font-size:1.05rem; margin:0 0 .3em; color:var(--accent); }}
.what {{ margin:.2em 0 .6em; color:var(--muted); font-size:.92rem; }}
.row {{ display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:12px; }}
.col h3 {{ font-size:.95rem; margin:.2em 0; }}
.col audio {{ width:100%; }}
.set {{ font-size:.88rem; margin:.3em 0 0; }}
.note {{ font-size:.85rem; color:var(--muted); margin:.5em 0 0; }}
@media (max-width:720px) {{ .row {{ grid-template-columns:1fr; }} }}
</style></head>
<body><main>
<h1>Automatic settings: listening check</h1>
<p class="intro">Each card has the original, the output with the settings the current rules pick, and the output with the settings
the new model picks. Loudness is matched to -16 LUFS in all three, so only the cleaning differs. The voice rebuild (Sidon) and
noise removal (DeepFilterNet3, streaming, as the site runs it) come from the repository's own code. The heavy remover is not in
this project: the heavy flag is recorded, but the audio uses the same noise remover at the chosen strength. The
"muffled but clean" cards are the false-positive case: a dulled microphone with nothing else wrong, where a voice rebuild is not
needed. Listen before you judge: most of these are small differences.</p>
{''.join(cards)}
</main></body></html>
"""
    open(f"{LIST}/index.html", "w").write(page)
    print("page", f"{LIST}/index.html", len(meta), "clips", flush=True)
