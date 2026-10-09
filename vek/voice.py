"""Stage 2/4 - cut list -> output timeline, voice assembly, voice processing, caption word timings."""
import json, subprocess
import numpy as np
import soundfile as sf
from .common import Job, asr, energy_db, voiced_runs, Timeline


def build_edl(job):
    ed = job.load("edit.json")
    t = 0.0
    segs = []
    for s in ed["segments"]:
        d = float(s["out"]) - float(s["in"])
        segs.append(dict(name=s["name"], vin=float(s["in"]), vout=float(s["out"]), ostart=round(t, 4),
                         oend=round(t + d, 4), enter=s.get("enter", "cut"), shot=s.get("shot", {})))
        t += d
    edl = dict(segs=segs, speech_end=round(t, 4))
    ec = ed.get("end_card")
    tl = Timeline(dict(segs=segs, total=t), [], ed)
    if ec:
        ein = tl.t(ec.get("in", segs[-1]["name"] + ".end"))
        edl["endcard_in"] = round(ein, 4)
        edl["total"] = round(ein + float(ec.get("length", 5.4)), 4)
    else:
        edl["endcard_in"] = None
        edl["total"] = round(t + 0.4, 4)
    job.save("edl.json", edl)
    return edl


def assemble(job):
    edl = build_edl(job)
    a, sr = sf.read(job.p("audio48.wav"), dtype="float32")
    fade = int(0.006 * sr)
    r = np.linspace(0, 1, fade, dtype=np.float32)
    parts = []
    for s in edl["segs"]:
        x = a[int(round(s["vin"] * sr)):int(round(s["vout"] * sr))].copy()
        x[:fade] *= r
        x[-fade:] *= r[::-1]
        parts.append(x)
    v = np.concatenate(parts)
    sf.write(job.p("voice_raw.wav"), v, sr, subtype="FLOAT")
    return edl


VOICE_CHAIN = ("highpass=f=80,afftdn=nr=12:nf=-50:tn=1,"
               "equalizer=f=280:t=q:w=1.2:g=-3,equalizer=f=3200:t=q:w=1.0:g=3,equalizer=f=6500:t=q:w=1.5:g=1.5,"
               "deesser=i=0.35,acompressor=threshold=-24dB:ratio=3:attack=6:release=90:makeup=4dB,alimiter=limit=0.89")


def process(job, chain=VOICE_CHAIN):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", job.p("voice_raw.wav"), "-af", chain, "-ar", "48000",
                    "-c:a", "pcm_f32le", job.p("voice_chain.wav")], check=True)
    level(job)


def level(job, max_cut=3.0, max_boost=6.0):
    """Ride the fader: bring every segment to the median speech level (soft phrases like a quiet
    'goodbye' otherwise get buried under the music)."""
    edl = job.load("edl.json")
    v, sr = sf.read(job.p("voice_chain.wav"), dtype="float32")
    if v.ndim > 1:
        v = v.mean(1)
    hop = int(0.02 * sr)

    def act(x):
        if len(x) < hop * 3:
            return None
        n = len(x) // hop
        db = 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
        a = db[db > db.max() - 20]
        return 20 * np.log10(np.sqrt(np.mean((10 ** (a / 20)) ** 2)))
    vals = [act(v[int(s["ostart"] * sr):int(s["oend"] * sr)]) for s in edl["segs"]]
    ok = [x for x in vals if x is not None]
    tgt = float(np.median(ok)) if ok else -24
    gain = np.ones(len(v), np.float32)
    report = []
    for s, val in zip(edl["segs"], vals):
        g = 0.0 if val is None else float(np.clip(tgt - val, -max_cut, max_boost))
        gain[int(s["ostart"] * sr):int(s["oend"] * sr)] = 10 ** (g / 20)
        report.append((s["name"], round(g, 1)))
    k = int(0.03 * sr)
    gain = np.convolve(np.pad(gain, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid").astype(np.float32)
    out = v * gain[:len(v)]
    pk = np.abs(out).max()
    if pk > 0.98:
        out *= 0.98 / pk
    sf.write(job.p("voice.wav"), out, sr, subtype="FLOAT")
    print("[voice] level (dB):", " ".join(f"{n}:{g:+.1f}" for n, g in report))


def verify(job):
    import scipy.signal as ss
    v, sr = sf.read(job.p("voice.wav"), dtype="float32")
    v16 = ss.resample_poly(v, 1, 3).astype(np.float32)
    txt, words = asr(v16)
    job.save("voice_asr.json", dict(text=txt, words=words))
    print("[voice] edited voice reads:\n  " + txt)
    return txt, words


def word_times(job):
    """Caption timings on the output timeline via CTC forced alignment of the recognised words
    (accurate to ~40 ms). Long audio is aligned in windows split at pauses."""
    import scipy.signal as ss
    from .align import align_words
    v, sr = sf.read(job.p("voice.wav"), dtype="float32")
    v16 = ss.resample_poly(v, 1, 3).astype(np.float32)
    edl = job.load("edl.json")
    W = job.load("voice_asr.json")["words"]
    wins, cur = [], []
    for i, w in enumerate(W):
        if cur and (w["s"] - W[cur[-1]]["e"] > 0.35 and w["s"] - W[cur[0]]["s"] > 20):
            wins.append(cur); cur = []
        cur.append(i)
    if cur: wins.append(cur)
    times = [None] * len(W)
    for k, win in enumerate(wins):
        a0 = max(0.0, W[win[0]]["s"] - 0.6) if k else 0.0
        b0 = min(len(v16) / 16000, W[win[-1]]["e"] + 0.6) if k < len(wins) - 1 else len(v16) / 16000
        st = align_words(v16[int(a0 * 16000):int(b0 * 16000)], [W[i]["w"] for i in win], offset=a0)
        if st is None:
            st = [W[i]["s"] - 0.1 for i in win]
        for i, t in zip(win, st):
            times[i] = t - 0.02
    out = []
    for i, w in enumerate(W):
        t = max(0.0, times[i])
        if out and t <= out[-1]["t"] + 0.04:
            t = out[-1]["t"] + 0.06
        seg = next((s["name"] for s in edl["segs"] if s["ostart"] <= t < s["oend"]), edl["segs"][-1]["name"])
        out.append(dict(i=i, w=w["w"], t=round(t, 3), seg=seg))
    job.save("words.json", out)
    print("[voice] caption words (index: word @ time):")
    print("  " + "  ".join(f"{o['i']}:{o['w']}@{o['t']:.2f}" for o in out))
    return out


def main(job_dir):
    job = Job(job_dir)
    edl = assemble(job)
    print(f"[voice] {len(edl['segs'])} segments, speech {edl['speech_end']:.2f}s, total {edl['total']:.2f}s")
    for s in edl["segs"]:
        print(f"  {s['name']:>5}  src {s['vin']:7.2f}-{s['vout']:7.2f}  ->  out {s['ostart']:6.2f}-{s['oend']:6.2f}  {s['enter']}")
    process(job)
    verify(job)
    word_times(job)
