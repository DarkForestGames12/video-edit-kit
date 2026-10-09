"""Stage 1b - speech map: split speech into phrases at pauses, transcribe each, flag fillers/false starts.

Why phrase-level: the recogniser's word timestamps run 0.1-0.3 s late and it silently drops
fillers ("um", "mm", false starts) when transcribing long stretches. Re-transcribing each
pause-delimited phrase on its own exposes them, and the phrase edges (from audio energy)
are accurate to ~10 ms, which is what cut points need.
"""
import re, json
import numpy as np
import soundfile as sf
from .common import Job, asr, energy_db, voiced_runs

FILLERS = {"um", "uh", "uhm", "umm", "mm", "mmm", "hmm", "er", "erm", "ah", "eh", "uh-huh", "mhm"}
SOFT_FILLERS = {"yeah", "okay", "ok", "right", "so", "like", "well", "no"}


def norm(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def phrase_map(job, gap=0.10, min_len=0.05):
    a, sr = sf.read(job.p("audio16.wav"), dtype="float32")
    db = energy_db(a, sr, 0.01, 3)
    floor = np.percentile(db, 10)
    speech = np.percentile(db[db > floor + 6], 90) if (db > floor + 6).any() else floor + 30
    thr = float(max(speech - 22, floor + 8))
    runs = voiced_runs(db, thr, merge=int(gap * 100), min_len=int(min_len * 100))
    phrases = []
    for s, e in runs:
        s0 = max(0, s / 100 - 0.04)
        e0 = min(len(a) / sr, e / 100 + 0.06)
        txt, _ = asr(a[int(s0 * sr):int(e0 * sr)])
        phrases.append(dict(s=round(s0, 2), e=round(e0, 2), text=txt))
    # context transcription in windows (better wording), words mapped back to phrases
    windows = []
    cur = []
    for i, p in enumerate(phrases):
        if cur and (p["e"] - phrases[cur[0]]["s"] > 25):
            windows.append(cur)
            cur = []
        cur.append(i)
    if cur:
        windows.append(cur)
    import difflib
    for win in windows:
        ws, we = phrases[win[0]]["s"], phrases[win[-1]]["e"]
        txt, words = asr(a[int(ws * sr):int(we * sr)])
        # align window words to the per-phrase transcripts (timestamps alone are unreliable)
        ref, rid = [], []
        for i in win:
            for t in norm(phrases[i]["text"]):
                ref.append(t); rid.append(i)
        hyp = [norm(w["w"])[0] if norm(w["w"]) else "" for w in words]

        def by_time(w):
            t = ws + w["s"] - 0.12
            return min(win, key=lambda i: 0 if phrases[i]["s"] <= t <= phrases[i]["e"] else
                       min(abs(t - phrases[i]["s"]), abs(t - phrases[i]["e"])))
        owner = [None] * len(words)
        sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op in ("equal", "replace") and i2 > i1:
                for j in range(j1, j2):
                    k = i1 + min(i2 - i1 - 1, int((j - j1) * (i2 - i1) / max(1, j2 - j1)))
                    owner[j] = rid[k]
        for j, w in enumerate(words):
            phrases[owner[j] if owner[j] is not None else by_time(w)].setdefault("ctx", []).append(w["w"])
    for i, p in enumerate(phrases):
        p["ctx"] = " ".join(p.get("ctx", []))
        p["gap_before"] = round(p["s"] - (phrases[i - 1]["e"] if i else 0), 2)
        toks = norm(p["text"])
        flags = []
        if not toks and not p["ctx"]:
            flags.append("noise/breath")
        elif toks and all(t in FILLERS for t in toks):
            flags.append("FILLER")
        elif all(t in SOFT_FILLERS for t in toks) and p["e"] - p["s"] < 0.8 and not p["ctx"]:
            flags.append("filler?")
        if i + 1 < len(phrases):
            nt = norm(phrases[i + 1]["text"])
            if toks and nt and len(toks) <= len(nt) and nt[:len(toks)] == toks:
                flags.append("REPEAT->next (false start?)")
            elif len(toks) >= 2 and nt[:2] == toks[:2]:
                flags.append("same opening as next (retake?)")
            elif toks and nt and toks[0] == nt[0] and p["e"] - p["s"] < 1.0:
                flags.append("starts like next - false start?")
        if p["gap_before"] > 0.6:
            flags.append(f"pause {p['gap_before']:.1f}s before")
        p["flags"] = flags
    job.save("phrases.json", dict(threshold_db=thr, phrases=phrases))
    print(f"[transcribe] speech threshold {thr:.1f} dBFS")
    write_transcript(job, phrases)
    return phrases


def write_transcript(job, phrases):
    lines = ["# Phrase map (times = source video clock, seconds)", "",
             "Each row is speech between pauses. `asr` = phrase transcribed alone (shows fillers),",
             "`ctx` = words from transcribing the surrounding window (better wording).", "",
             "| # | start | end | dur | asr | ctx | flags |", "|---|---|---|---|---|---|---|"]
    for i, p in enumerate(phrases):
        lines.append(f"| {i} | {p['s']:.2f} | {p['e']:.2f} | {p['e']-p['s']:.2f} | {p['text']} | {p['ctx']} | "
                     f"{', '.join(p['flags'])} |")
    with open(job.p("transcript.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def check_ranges(job, ranges, join=False):
    """Transcribe source ranges ("s:e"). join=True concatenates them (tests how a cut will sound)."""
    a, sr = sf.read(job.p("audio16.wav"), dtype="float32")
    parts = []
    for r in ranges:
        s, e = map(float, r.split(":"))
        parts.append(a[int(s * sr):int(e * sr)])
    if join:
        gap = np.zeros(int(0.06 * sr), np.float32)
        x = np.concatenate([p for q in parts for p in (q, gap)])
        print("JOIN:", asr(x)[0])
    else:
        for r, x in zip(ranges, parts):
            print(f"{r:>14}  {asr(x)[0]!r}")


def main(job_dir):
    job = Job(job_dir)
    ph = phrase_map(job)
    print(f"[transcribe] {len(ph)} phrases -> {job.p('transcript.md')}")
    full = " ".join(p["ctx"] or p["text"] for p in ph)
    print(full)
