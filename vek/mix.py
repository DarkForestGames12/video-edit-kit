"""Stage 4d - mix + master. Music ducks under the voice (attack 40 ms / release 350 ms), lifts in
pauses, the title drop and the end card. Master: +gain, true-peak limiting, -14 LUFS / -1 dBTP."""
import json, subprocess, re
import numpy as np, soundfile as sf
from .common import Job, Timeline

SR = 48000
DUCK = dict(hook=-16.0, title=-12.5, speech=-20.5, gap=-9.0, endcard=-6.5, music_gain=0.62, sfx_gain=0.24)


def loudness(path, s=None, e=None):
    cmd = ["ffmpeg", "-hide_banner"]
    if s is not None: cmd += ["-ss", str(s), "-to", str(e)]
    cmd += ["-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True).stderr
    I = re.findall(r"I:\s+(-?[\d.]+) LUFS", r); P = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", r)
    return (float(I[-1]) if I else None, float(P[-1]) if P else None)


def mix(job, target=-14.0):
    ed = job.load("edit.json"); edl = job.load("edl.json")
    tl = Timeline(edl, job.load("words.json", []), ed)
    d = dict(DUCK); d.update(ed.get("mix", {}))
    N = int(edl["total"] * SR)

    def rd(name, ch):
        if not job.has(name): return np.zeros((N, ch))
        x, _ = sf.read(job.p(name), dtype="float32")
        if x.ndim == 1: x = x[:, None]
        if x.shape[1] != ch: x = np.repeat(x[:, :1], ch, 1)
        y = np.zeros((N, ch)); y[:min(N, len(x))] = x[:N]
        return y
    voice = rd("voice.wav", 1)[:, 0]; music = rd("music.wav", 2); sfx = rd("sfx.wav", 2)
    hop = int(0.01 * SR); n = N // hop
    env = np.sqrt((voice[:n * hop].reshape(n, hop) ** 2).mean(1))
    db = 20 * np.log10(env + 1e-9)
    act = (db > max(np.percentile(db, 95) - 24, -45)).astype(float)
    g = np.zeros(n); aa = 1 - np.exp(-1 / 4); ar = 1 - np.exp(-1 / 35)
    for i in range(n):
        p = g[i - 1] if i else 0
        g[i] = p + (act[i] - p) * (aa if act[i] > p else ar)
    tt = np.arange(n) * 0.01
    tc = ed.get("title_card")
    ti, te = (tl.t(tc["in"]), tl.t(tc["end"])) if tc else (-1, -1)
    whips = [s["ostart"] for s in edl["segs"] if s["enter"] == "whip"]
    hook_end = whips[0] if whips else (ti if ti > 0 else 0)
    sp = np.where(tt < hook_end, d["hook"], np.where((tt >= ti) & (tt < te), d["title"], d["speech"]))
    duck = sp * g + d["gap"] * (1 - g)
    if edl.get("endcard_in") is not None:
        rise = np.clip((tt - edl["endcard_in"]) / 0.6, 0, 1)
        duck = duck * (1 - rise) + d["endcard"] * rise
    gain = np.interp(np.arange(N) / SR, tt, 10 ** (duck / 20))
    sduck = np.interp(np.arange(N) / SR, tt, 10 ** (d.get("sfx_duck", -4.0) * g / 20))   # effects step back under speech
    stems = dict(voice=np.stack([voice, voice], 1), music=music * gain[:, None] * d["music_gain"],
                 sfx=sfx * sduck[:, None] * d["sfx_gain"])
    for k, v in stems.items():
        sf.write(job.p(f"stem_{k}.wav"), v, SR, subtype="FLOAT")
    # balance against the measured voice: recordings differ a lot in level
    spe = max(1, (edl.get("endcard_in") or edl["total"]) - 0.1)
    Lv = loudness(job.p("stem_voice.wav"), 0, spe)[0]
    for k, rel in (("music", d.get("music_rel", -12.5)), ("sfx", d.get("sfx_rel", -5.5))):
        L = loudness(job.p(f"stem_{k}.wav"), 0, spe)[0]
        if Lv is not None and L is not None and L > -70:
            g = float(np.clip(10 ** ((Lv + rel - L) / 20), 0.1, 3.0))
            stems[k] = stems[k] * g
            sf.write(job.p(f"stem_{k}.wav"), stems[k], SR, subtype="FLOAT")
    pre = stems["voice"] + stems["music"] + stems["sfx"]
    sf.write(job.p("mix_pre.wav"), pre, SR, subtype="FLOAT")
    I, P = loudness(job.p("mix_pre.wav"))
    gdb = target - I + 0.35
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", job.p("mix_pre.wav"), "-af",
                    f"volume={gdb:.2f}dB,alimiter=limit=0.85:attack=3:release=60:level=disabled,aresample=192000,"
                    "alimiter=limit=0.84:attack=1:release=40:level=disabled,aresample=48000",
                    "-c:a", "pcm_s24le", job.p("master.wav")], check=True)
    I2, P2 = loudness(job.p("master.wav"))
    print(f"[mix] master {I2} LUFS, true peak {P2} dBTP")
    rep = {}
    for k in ("voice", "music", "sfx"):
        rep[k] = loudness(job.p(f"stem_{k}.wav"), 0, max(1, (edl.get("endcard_in") or edl["total"]) - 0.1))[0]
    print(f"[mix] during speech: voice {rep['voice']} | music {rep['music']} | sfx {rep['sfx']} LUFS "
          f"(music should sit ~11-14 LU under voice)")
    return I2, P2


def main(job_dir):
    mix(Job(job_dir))
