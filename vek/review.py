"""Stage 6 - review: contact sheets of the render, spot-check list, final-audio speech check, loudness."""
import os, subprocess, glob, re
import numpy as np, cv2, soundfile as sf
from .common import Job, asr, FPS
from .mix import loudness


def sheet(paths, out, cols=5, w=480, labels=None):
    ims = []
    for k, p in enumerate(paths):
        im = cv2.imread(p)
        h = int(im.shape[0] * w / im.shape[1])
        im = cv2.resize(im, (w, h))
        lab = labels[k] if labels else os.path.basename(p).split("_")[-1].rsplit(".", 1)[0]
        cv2.putText(im, lab, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ims.append(im)
    while len(ims) % cols:
        ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(out, np.vstack([np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def sheets_from_video(job, video, fps=4, per=40, cols=5, w=400):
    d = job.p("out", "review"); os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(d, "r_*.jpg")): os.remove(f)
    subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"fps={fps},scale={w}:-2", os.path.join(d, "r_%05d.jpg")], check=True)
    fs = sorted(glob.glob(os.path.join(d, "r_*.jpg")))
    outs = []
    for s in range(0, len(fs), per):
        blk = fs[s:s + per]
        outs.append(sheet(blk, job.p("out", f"review_{s // per:02d}.jpg"), cols, w, [f"{(s + k) / fps:.2f}" for k in range(len(blk))]))
    return outs


def speech_check(job, video):
    tmp = job.p("_final16.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le", tmp], check=True)
    a, sr = sf.read(tmp, dtype="float32"); os.remove(tmp)
    edl = job.load("edl.json")
    end = edl["speech_end"] + 0.3
    got = []
    t = 0.0
    while t < end:            # windows of ~20 s (long single passes drop trailing words)
        e = min(end, t + 20)
        txt, _ = asr(a[int(t * sr):int(e * sr)])
        got.append(txt); t = e - 0.0
    final = " ".join(got)
    ref = job.load("voice_asr.json")["text"]
    tok = lambda s: re.findall(r"[a-z']+", s.lower())
    rf, fn = tok(ref), tok(final)
    missing = [w for w in set(rf) if rf.count(w) > fn.count(w)]
    print("[review] final mix reads:\n  " + final)
    print("[review] words weaker/missing vs clean voice:", missing or "none")
    return final, missing


def spot_times(job):
    """Moments worth inspecting at full size: every boundary, transition and element edge."""
    ed = job.load("edit.json"); edl = job.load("edl.json")
    from .common import Timeline
    tl = Timeline(edl, job.load("words.json", []), ed)
    ts = []
    for s in edl["segs"][1:]:
        ts += [s["ostart"] - 0.04, s["ostart"] + 0.04]
    tc = ed.get("title_card")
    if tc: ts += [tl.t(tc["in"]) + 0.1, tl.t(tc["in"]) + 0.8, tl.t(tc["out"]) + 0.15]
    for c in ed.get("cards", []): ts += [tl.t(c["in"]) + 0.5]
    if ed.get("burst"): ts += [tl.t(ed["burst"]["in"]) + 0.6]
    if edl.get("endcard_in") is not None: ts += [edl["endcard_in"] + 0.25, edl["endcard_in"] + 2.0]
    return sorted(set(round(t, 2) for t in ts if 0 <= t < edl["total"]))


def main(job_dir, video):
    job = Job(job_dir)
    s = sheets_from_video(job, video)
    print("[review] sheets:", *s)
    speech_check(job, video)
    I, P = loudness(video)
    print(f"[review] loudness {I} LUFS, true peak {P} dBTP (target -14 / <= -1)")
    print("[review] spot-check times:", " ".join(str(t) for t in spot_times(job)))
