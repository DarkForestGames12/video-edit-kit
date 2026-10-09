"""Stage 2 helper - a first-draft cut list from the phrase map. Drops fillers, noise and phrases that a
later phrase repeats (false starts / retakes), and tightens pauses. Claude still reviews every
flagged phrase and makes the creative calls (hook, order, title cover) before rendering."""
import json, re
from .common import Job
from .transcribe import norm


def draft(job, keep_pause=0.35, pad_in=0.04, pad_out=0.10):
    ph = job.load("phrases.json")["phrases"]
    keep = []
    dropped = []
    for i, p in enumerate(ph):
        f = " ".join(p["flags"])
        if "FILLER" in f or "noise/breath" in f or "REPEAT->next" in f:
            dropped.append((i, p["text"], f)); continue
        keep.append(i)
    segs = []
    cur = None
    for i in keep:
        p = ph[i]
        if cur and p["s"] - cur["out"] <= keep_pause and i - cur["last"] == 1:
            cur["out"] = p["e"]; cur["last"] = i; cur["text"] += " " + (p["ctx"] or p["text"])
        else:
            if cur: segs.append(cur)
            cur = dict(first=i, last=i, **{"in": p["s"], "out": p["e"]}, text=p["ctx"] or p["text"])
    if cur: segs.append(cur)
    out = []
    for k, s in enumerate(segs):
        nxt = ph[s["last"] + 1]["s"] if s["last"] + 1 < len(ph) else s["out"] + 0.3
        prv = ph[s["first"] - 1]["e"] if s["first"] > 0 else 0
        a = max(prv + 0.01, s["in"] - pad_in)
        b = min(nxt - 0.01, s["out"] + pad_out)
        out.append(dict(name=f"S{k+1}", **{"in": round(a, 2), "out": round(b, 2)}, text=s["text"]))
    removed = sum(p["e"] - p["s"] for p in ph) - 0
    kept = sum(s["out"] - s["in"] for s in out)
    job.save("cuts.draft.json", dict(segments=out, dropped=[dict(i=i, text=t, flags=f) for i, t, f in dropped]))
    print(f"[autocut] {len(out)} segments, {kept:.1f}s kept of {job.load('probe.json')['duration']:.1f}s source")
    for s in out:
        print(f"  {s['name']:>4} {s['in']:8.2f}-{s['out']:8.2f}  {s['text']}")
    if dropped:
        print("  dropped:", "; ".join(f"#{i} '{t}' ({f})" for i, t, f in dropped))
    return out


def main(job_dir):
    draft(Job(job_dir))
