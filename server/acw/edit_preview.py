"""What a job's block cache holds, the stretches of it a preview plays and the outlines drawn from it (acw/editing.py has the whole
picture)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import edit_store, jobs


# ---------------------------------------------------------------------------------------------------------------------
# What is cached, and the stretches of it a preview plays
# ---------------------------------------------------------------------------------------------------------------------
def probe_timeline(tl, layer: str, remover: str | None = None):
    """A timeline that asks for `layer` on the whole recording (with the echo parts of `tl`, which the blocks' identity includes):
    its plan lists every block of that layer, the cache says which of them exist."""
    t = edit_store.timeline_lib().Timeline(tl.seconds)
    for p in tl.layers["echo"]:
        t.paint("echo", p.start, p.end)
    if layer == "noise":
        t.paint("noise", 0.0, t.seconds, {"strength_db": 100.0, "remover": remover or "light"})
    elif layer == "rebuild":
        t.paint("rebuild", 0.0, t.seconds)
    elif layer == "echo":
        t.paint("echo", 0.0, t.seconds)
    return t


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(r for r in ranges if r[1] > r[0]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _intersect(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _subtract(whole: tuple[int, int], covered: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out, pos = [], whole[0]
    for a, b in _merge(covered):
        if b <= pos or a >= whole[1]:
            continue
        if a > pos:
            out.append((pos, a))
        pos = max(pos, b)
    if pos < whole[1]:
        out.append((pos, whole[1]))
    return out


def _plan_of(job: dict, params: dict, probe, inputs=None):
    ed = edit_store.editor_lib()
    try:
        return ed.plan(probe, inputs or edit_store.inputs_for(job, params), edit_store.cache_for(job["id"]))
    except ed.EditorError:
        return None  # (echo removal on a file without a second track: nothing of it exists)


def layer_coverage(job: dict, params: dict, tl, step: str, plan=None) -> dict:
    """What the cache holds of one step on the whole recording: {"channels": {channel: [(a, b)]}, "silent": {channel: [(a, b)]}}
    in samples (silent: stretches the noise removers skip, digital silence, that count as done)."""
    inputs = edit_store.inputs_for(job, params)
    layer = {"echo": "echo", "light": "noise", "heavy": "noise", "rebuild": "rebuild"}[step]
    plan = plan or _plan_of(job, params, probe_timeline(tl, layer, step if layer == "noise" else None), inputs)
    cache = edit_store.cache_for(job["id"])
    out: dict = {"channels": {}, "silent": {}}
    if plan is None:
        return out
    for t in plan.tasks:
        if t.step != step:
            continue
        out["channels"].setdefault(t.channel, [])
        if cache.has(t.key):
            out["channels"][t.channel].append((t.s, t.e))
    if layer == "noise":
        for c in out["channels"]:
            spans = inputs.spans(c)
            out["silent"][c] = _subtract((0, inputs.n), list(spans))
    return out


def covered(cov: dict) -> list[tuple[int, int]]:
    """The samples where every channel of the step has its block (or is silent there)."""
    sets = [_merge(cov["channels"][c] + cov["silent"].get(c, [])) for c in cov["channels"]]
    if not sets:
        return []
    out = sets[0]
    for s in sets[1:]:
        out = _intersect(out, s)
    return out


def cached_ranges(job: dict, params: dict, tl) -> dict[str, list[list[float]]]:
    """What the cache holds, per layer, as time ranges in seconds: `echo`, `noise` (either remover; `light` and `heavy` apart) and
    `rebuild`. Leveling is not cached (it is made again at every render)."""
    if not edit_store.blocks_dir(job["id"]).is_dir():
        return {"echo": [], "noise": [], "light": [], "heavy": [], "rebuild": []}
    secs = lambda rs: [[round(a / edit_store.SR, 3), round(b / edit_store.SR, 3)] for a, b in rs]  # noqa: E731
    by = {s: covered(layer_coverage(job, params, tl, s)) for s in ("echo", "light", "heavy", "rebuild")}
    return {"echo": secs(by["echo"]), "noise": secs(_merge(by["light"] + by["heavy"])), "light": secs(by["light"]), "heavy": secs(by["heavy"]),
            "rebuild": secs(by["rebuild"])}


def stretch_removers(tl, params: dict, resolved: dict | None, first: int, end: int, has: dict[str, list[tuple[int, int]]]) -> list[tuple[int, int, str]]:
    """Which remover's output a preview of [first, end) plays at each moment: the one of the noise part there (decided, or for a part
    still on "auto" the one whose blocks exist, light first), and outside the noise parts the one the job used (else light)."""
    fallback = params.get("noise_profile") if params.get("noise_profile") in ("light", "heavy") else "light"
    segs: list[tuple[int, int, str]] = []
    pos = first
    for p in sorted(tl.layers["noise"], key=lambda p: p.start):
        a, b = max(first, int(round(p.start * edit_store.SR))), min(end, int(round(p.end * edit_store.SR)))
        if b <= a:
            continue
        if a > pos:
            segs.append((pos, a, fallback))
        r = "browser" if p.settings.get("browser") else p.settings.get("remover")  # (a browser part has no model blocks: not cached here)
        if r == "auto":
            r = (resolved or {}).get(edit_store.part_key(p.start, p.end))
        if r != "browser" and r not in ("light", "heavy"):
            r = "light" if _intersect(has.get("light", []), [(a, b)]) or not _intersect(has.get("heavy", []), [(a, b)]) else "heavy"
        segs.append((a, b, r))
        pos = b
    if pos < end:
        segs.append((pos, end, fallback))
    merged: list[tuple[int, int, str]] = []
    for a, b, r in segs:
        if merged and merged[-1][1] == a and merged[-1][2] == r:
            merged[-1] = (merged[-1][0], b, r)
        else:
            merged.append((a, b, r))
    return merged


def plan_stretch(job: dict, params: dict, layer: str, first: int, count: int) -> dict:
    """What the assembler needs to make the stretch [first, first + count) samples of `layer` ("echo", "noise", "rebuild") from the
    cache, per output channel: the block files that overlap it (each with the part of the stretch it is used for: `from`, `to`), and
    which parts of the stretch the cache does not hold.
    {"channels": [[{"name", "at", "len", "from", "to"}] | None per output channel], "missing": [(a, b)] relative to `first`}"""
    f = edit_store.facts_of(job, params)
    seconds = edit_store.file_seconds(job, params)
    tl = edit_store.saved_timeline(job, params, seconds)
    cache = edit_store.cache_for(job["id"])
    end = first + count
    inputs = edit_store.inputs_for(job, params)
    chans: list[list[dict] | None] = [[] for _ in range(f["channels"])]
    have: list[list[tuple[int, int]]] = [[] for _ in range(f["channels"])]  # what each output channel holds of the stretch

    def take(plan, step: str, ranges: list[tuple[int, int]]) -> None:
        for t in plan.tasks:
            if t.step != step or not cache.has(t.key):
                continue
            for a, b in _intersect([(t.s, t.e)], ranges):
                outs = range(f["channels"]) if t.channel == "mid" else [c for c in range(f["channels"]) if (f["plan"][c] if step in ("light", "heavy") else c) == t.channel]
                for c in outs:
                    if chans[c] is not None:
                        chans[c].append({"name": cache.name(t.key), "at": t.s, "len": t.e - t.s, "from": a, "to": b})
                        have[c].append((a, b))

    if layer == "echo":
        plan = _plan_of(job, params, probe_timeline(tl, "echo"), inputs)
        if plan is not None:
            take(plan, "echo", [(first, end)])
    elif layer == "rebuild":
        plan = _plan_of(job, params, probe_timeline(tl, "rebuild"), inputs)
        if plan is not None:
            take(plan, "rebuild", [(first, end)])
    else:
        has = {s: covered(layer_coverage(job, params, tl, s)) for s in ("light", "heavy")}
        for a, b, r in stretch_removers(tl, params, (params.get("edit") or {}).get("resolved"), first, end, has):
            if r == "browser":
                continue
            plan = _plan_of(job, params, probe_timeline(tl, "noise", r), inputs)
            if plan is not None:
                take(plan, r, [(a, b)])
        for c in range(f["channels"]):  # a channel with nothing to clean (silent), or digital silence the removers skip
            src = f["plan"][c]
            if src is None:
                chans[c] = None
            else:
                have[c] += _intersect(_subtract((0, inputs.n), list(inputs.spans(src))), [(first, end)])
    active = [c for c in range(f["channels"]) if chans[c] is not None]
    got = [(first, end)] if not active else _merge(have[active[0]])
    for c in active[1:]:
        got = _intersect(got, _merge(have[c]))
    missing = _subtract((first, end), got) if active else []
    return {"channels": chans, "missing": [(a - first, b - first) for a, b in missing]}


# ---------------------------------------------------------------------------------------------------------------------
# Outlines: the loudness of what a layer holds, for drawing it
# ---------------------------------------------------------------------------------------------------------------------
PEAKS_RATE = 4  # values per second of the recording
PEAKS_WINDOW = edit_store.SR // PEAKS_RATE
PEAKS_BUDGET_SECONDS = 25.0  # how long one request works on blocks that have no outline yet (the rest follow on the next call)
PEAKS_PER_RUN = 24  # blocks one confined run reads


def _read_outline(path: Path) -> dict | None:
    data = edit_store.read_json(path, 4_000_000)
    if isinstance(data, dict) and isinstance(data.get("first"), int) and isinstance(data.get("p"), list) and all(isinstance(v, (int, float)) for v in data["p"]):
        return data
    return None


def _write_outline(path: Path, data: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    os.replace(tmp, path)


def peaks_of(job: dict, params: dict, layer: str) -> dict | None:
    """The outline of a layer (`source`, `echo`, `noise` or `rebuild`, the stems of the preview, on its sample grid): `{"rate": 4,
    "seconds", "peaks": [...]}`, one value per quarter second, the largest absolute sample over all channels (0 to 1, three decimals),
    `None` where the layer is not in the cache for that moment. Made from the cached blocks (decoded by the confined helper, never here)
    and kept per block next to it (`blocks/<name>.peaks`, the recording's in `kept-source.peaks`), so a block is read once. None when
    nothing of the layer is cached; `pending` is added when the time of this call ran out before every block had its outline."""
    from . import stretch

    f = (params.get("edit") or {}).get("facts")
    n = int(f["n"])
    windows = -(-n // PEAKS_WINDOW)
    d = jobs.job_dir(job["id"])
    out: list[float | None] = [None] * windows
    pending = False
    if layer == "source":
        entry = f.get("source")
        if not entry or not edit_store.regular_file(d / entry["file"]):
            return None
        side = d / (Path(entry["file"]).stem + ".peaks")
        got = _read_outline(side)
        if got is None:
            got = stretch.peaks(d, {"file": entry["file"], "channels": f["channels"], "scale": float(entry["scale"])}, timeout=240.0)
            if not isinstance(got.get("p"), list):
                raise stretch.Failed("the outline could not be read")
            _write_outline(side, {"first": 0, "p": got["p"]})
        for i, v in enumerate(got["p"][:windows]):
            out[i] = v
    else:
        plan = plan_stretch(job, params, layer, 0, n)
        wanted: dict[str, dict] = {}
        for chan in plan["channels"]:
            for b in chan or []:
                wanted.setdefault(b["name"], b)
        have: dict[str, dict] = {}
        todo: list[dict] = []
        for name, b in wanted.items():
            got = _read_outline(d / edit_store.BLOCKS / f"{name}.peaks")
            if got is None:
                todo.append({"name": name, "at": b["at"]})
            else:
                have[name] = got
        import time as _time

        t0 = _time.monotonic()
        for i in range(0, len(todo), PEAKS_PER_RUN):
            if _time.monotonic() - t0 > PEAKS_BUDGET_SECONDS:
                pending = True
                break
            made = stretch.peaks(d, {"blocks": todo[i:i + PEAKS_PER_RUN]})
            for name, got in made.items():
                if name in wanted and isinstance(got, dict) and isinstance(got.get("first"), int) and isinstance(got.get("p"), list):
                    _write_outline(d / edit_store.BLOCKS / f"{name}.peaks", {"first": got["first"], "p": got["p"]})
                    have[name] = got
        unknown = [False] * windows  # under a block that has no outline yet (the time ran out)
        for chan in plan["channels"]:
            for b in chan or []:
                got = have.get(b["name"])
                for w in range(b["from"] // PEAKS_WINDOW, min(windows, -(-b["to"] // PEAKS_WINDOW))):
                    if got is None:
                        unknown[w] = True
                        continue
                    i = w - got["first"]
                    if 0 <= i < len(got["p"]):
                        out[w] = got["p"][i] if out[w] is None else max(out[w], got["p"][i])
        # a window the cache lacks (in all of it) is absent; one with nothing missing but no block under it is silence the removers skip
        absent = [False] * windows
        for a, z in plan["missing"]:
            for w in range(-(-a // PEAKS_WINDOW), min(windows, z // PEAKS_WINDOW)):
                absent[w] = True
        for w in range(windows):
            if absent[w] or unknown[w]:
                out[w] = None
            elif out[w] is None:
                out[w] = 0.0
    if all(v is None for v in out):
        return None
    result = {"rate": PEAKS_RATE, "seconds": round(n / edit_store.SR, 3), "peaks": [None if v is None else round(min(1.0, float(v)), 3) for v in out]}
    if pending:
        result["pending"] = True
    return result
