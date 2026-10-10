"""Stage 4c - sound design. Every graphic/transition in edit.json gets its sound automatically,
so picture and sound always agree. Extra cues: edit.json "sfx_extra": [{"t":..,"kind":..,"gain":..,"pan":..}].
"""
import numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve
from .common import Job, Timeline

SR = 48000


class FX:
    def __init__(self, length, seed=11):
        self.rng = np.random.default_rng(seed)
        self.L = int((length + 0.5) * SR)
        self.out = np.zeros((self.L, 2))
        n = int(1.4 * SR); t = np.arange(n) / SR
        self.IR = np.stack([self.lp(self.rng.standard_normal(n), 5000) * np.exp(-t * 3.0) for _ in range(2)], 1) * 0.06

    def T(self, s): return int(round(s * SR))
    def lp(self, x, f, o=2): return sosfilt(butter(o, f, 'low', fs=SR, output='sos'), x)
    def hp(self, x, f, o=2): return sosfilt(butter(o, f, 'high', fs=SR, output='sos'), x)

    def add(self, x, t, g=1.0, pan=0.0):
        if x.ndim == 1:
            l = np.cos((pan + 1) * np.pi / 4) * 1.414; r = np.sin((pan + 1) * np.pi / 4) * 1.414
            x = np.stack([x * l, x * r], 1)
        i = self.T(t)
        if i < 0: x = x[-i:]; i = 0
        n = min(len(x), self.L - i)
        if n > 0: self.out[i:i + n] += x[:n] * g

    def verb(self, x, wet=0.3):
        if x.ndim == 1: x = np.stack([x, x], 1)
        y = np.stack([fftconvolve(x[:, c], self.IR[:, c]) for c in range(2)], 1)
        z = np.zeros_like(y); z[:len(x)] = x
        return z + y * wet

    def sweep(self, x, f0, f1, q=1.5, blk=256):
        y = np.zeros_like(x); n = len(x)
        for i in range(0, n, blk):
            fc = f0 * (f1 / f0) ** (i / n); lo = max(40, fc / q); hi = min(SR / 2 - 100, fc * q)
            sos = butter(2, [lo, hi], 'band', fs=SR, output='sos')
            seg = x[max(0, i - 1024):i + blk]
            y[i:i + blk] = sosfilt(sos, seg)[-len(x[i:i + blk]):]
        return y

    def whoosh(self, dur=0.45, f0=300, f1=3500, pan0=-0.6, pan1=0.6, peak=0.5):
        n = self.T(dur); t = np.arange(n) / n
        y = self.sweep(self.rng.standard_normal(n), f0, f1) * np.exp(-((t - peak) / 0.22) ** 2)
        y /= np.abs(y).max() + 1e-9
        pan = pan0 + (pan1 - pan0) * t
        return np.stack([y * np.cos((pan + 1) * np.pi / 4) * 1.414, y * np.sin((pan + 1) * np.pi / 4) * 1.414], 1)

    def impact(self):
        n = self.T(1.6); t = np.arange(n) / SR
        sub = np.sin(2 * np.pi * np.cumsum(34 + 70 * np.exp(-t / 0.12)) / SR) * np.exp(-t / 0.55)
        nz = self.lp(self.rng.standard_normal(n), 2500) * np.exp(-t / 0.06) * 0.6
        body = self.lp(self.rng.standard_normal(n), 600) * np.exp(-t / 0.35) * 0.35
        return np.tanh(1.3 * (sub + nz + body))

    def riser(self, dur=0.75):
        n = self.T(dur); t = np.arange(n) / n
        y = self.sweep(self.rng.standard_normal(n), 400, 7000, q=1.3)
        tone = np.sin(2 * np.pi * np.cumsum(220 + 900 * t ** 2) / SR) * 0.25
        return (y / np.abs(y).max() + tone) * t ** 2.2

    def pop(self, f0=520, f1=980):
        n = self.T(0.14); t = np.arange(n) / SR
        x = np.sin(2 * np.pi * np.cumsum(f0 + (f1 - f0) * (1 - np.exp(-t / 0.02))) / SR) * np.exp(-t / 0.035)
        return x + self.hp(self.rng.standard_normal(n), 3000) * np.exp(-t / 0.004) * 0.25

    def thump(self, f=60, dur=0.3):
        n = self.T(dur); t = np.arange(n) / SR
        return np.tanh(2 * np.sin(2 * np.pi * np.cumsum(f + 40 * np.exp(-t / 0.02)) / SR) * np.exp(-t / 0.07) * np.minimum(t / 0.004, 1))

    def heartbeat(self):
        x = np.zeros(self.T(0.6)); a = self.thump(58); b = self.thump(68) * 0.75
        x[:len(a)] += a; i = self.T(0.2); x[i:i + len(b)] += b
        return self.lp(x, 400)

    def beep(self, f=1000, dur=0.08):
        n = self.T(dur); t = np.arange(n) / SR
        return np.sin(2 * np.pi * f * t) * np.minimum(t / 0.005, 1) * np.minimum((dur - t) / 0.02, 1) * 0.5

    def bell(self, freqs, dur=1.3, spacing=0.06):
        x = np.zeros(self.T(dur + spacing * len(freqs)))
        for k, f in enumerate(freqs):
            m = self.T(dur); t = np.arange(m) / SR
            y = np.sin(2 * np.pi * f * t + 2.0 * np.exp(-t / 0.25) * np.sin(2 * np.pi * f * 3.5 * t)) * np.exp(-t / 0.5) * np.minimum(t / 0.002, 1)
            i = self.T(k * spacing); x[i:i + m] += y
        return x * 0.5

    def click(self):
        n = self.T(0.03); t = np.arange(n) / SR
        return self.hp(self.rng.standard_normal(n), 2000) * np.exp(-t / 0.003) * 0.8 + np.sin(2 * np.pi * 1800 * t) * np.exp(-t / 0.006) * 0.4

    def kind(self, k):
        return {"pop": lambda: self.verb(self.pop()), "whoosh": lambda: self.whoosh(),
                "impact": lambda: self.verb(self.impact(), 0.25), "thump": lambda: self.thump(52, 0.35),
                "ding": lambda: self.verb(self.bell([1568, 2093, 2637], 1.5), 0.5), "click": self.click,
                "heartbeat": self.heartbeat, "riser": self.riser,
                "chime": lambda: self.verb(self.bell([1318.5, 1760.0, 2637.0], 1.3, 0.07), 0.45)}[k]()


def build(job):
    ed = job.load("edit.json"); edl = job.load("edl.json")
    tl = Timeline(edl, job.load("words.json", []), ed)
    fx = FX(edl["total"])
    fmt_w = 1920 if ed.get("format", "16:9") == "16:9" else 1080
    ht = ed.get("hook_text")
    if ht:
        t_in = tl.t(ht.get("in", 0.06)); t_out = tl.t(ht.get("out", "w1"))
        fx.add(fx.whoosh(0.5, 250, 2500, -0.2, 0.2, 0.75), t_in - 0.24, 0.22)
        fx.add(fx.impact() * 0.35, t_in, 0.5)
        fx.add(fx.whoosh(0.45, 2500, 300, 0, 0, 0.4), t_out, 0.20)
    segs = edl["segs"]
    for i, s in enumerate(segs):
        if s["enter"] == "whip":
            fx.add(fx.whoosh(0.42, 300, 5000, -0.9, 0.9, 0.5), s["ostart"] - 0.21, 0.55)
        elif i and s["enter"] == "cut":
            z_prev = (segs[i - 1]["shot"].get("z") or [1, 1])[-1]; z_new = (s["shot"].get("z") or [1, 1])[0]
            if z_new - z_prev >= 0.1 and s["ostart"] < 12:
                fx.add(fx.whoosh(0.25, 600, 3000, 0.3, -0.3, 0.5), s["ostart"] - 0.13, 0.15)
    for p in ed.get("punches", []):
        fx.add(fx.thump(52, 0.35) * 0.8, tl.t(p["at"]), 0.45)
    for k, st in enumerate(ed.get("stickers", [])):
        f0 = [500, 560, 620, 520][k % 4]
        pan = 0.4 if st.get("side", "right") == "right" else -0.35
        fx.add(fx.verb(fx.pop(f0, f0 * 1.8)), tl.t(st["in"]), 0.45, pan)
        if st.get("mode") == "spin":
            fx.add(fx.verb(fx.bell([1568, 2093], 0.8, 0.05), 0.4), tl.t(st["in"]) + 0.02, 0.08, pan)
    tc = ed.get("title_card")
    if tc:
        ti, to = tl.t(tc["in"]), tl.t(tc["out"])
        fx.add(fx.riser(0.75), ti - 0.75, 0.20)
        fx.add(fx.verb(fx.impact(), 0.25), ti, 0.50)
        fx.add(fx.whoosh(0.35, 800, 6000, -0.8, 0.8, 0.45), ti - 0.05, 0.25)
        fx.add(fx.verb(fx.bell([1046.5, 1318.5, 1568.0], 1.2, 0.05), 0.4), ti + 0.6, 0.10)
        fx.add(fx.whoosh(0.5, 5000, 300, 0, 0, 0.6), to - 0.12, 0.28)
    for c in ed.get("cards", []):
        ci, co = tl.t(c["in"]), tl.t(c["out"])
        fx.add(fx.whoosh(0.4, 400, 3000, -0.9, -0.3, 0.55), ci - 0.15, 0.30)
        fx.add(fx.whoosh(0.3, 2500, 500, -0.3, -0.9, 0.4), co - 0.05, 0.22)
        if c.get("kind") == "pulse":
            for b in c.get("beats", []):
                fx.add(fx.heartbeat(), tl.t(b), 0.70)
                if c.get("ecg"): fx.add(fx.beep(1000, 0.08), tl.t(b) + 0.02, 0.05, -0.4)
        elif c.get("pop") is not None:
            pt = tl.t(c["pop"])
            fx.add(fx.verb(fx.bell([1318.5, 1760.0, 2637.0], 1.3, 0.07), 0.45), pt, 0.09, -0.3)
            fx.add(fx.verb(fx.pop(700, 1200)), pt + 0.06, 0.22, -0.3)
    b = ed.get("burst")
    if b:
        bi, bo = tl.t(b["in"]), tl.t(b["out"])
        for k in range(len(b.get("icons", []))):
            f = 500 * 2 ** (k * 2 / 12)
            fx.add(fx.verb(fx.pop(f, f * 1.8), 0.25), bi + 0.07 * k, 0.17, -0.7 + 1.4 * k / max(1, len(b["icons"]) - 1))
        fx.add(fx.whoosh(0.3, 3000, 600, 0, 0, 0.5), bo - 0.24, 0.15)
    ec = edl.get("endcard_in")
    if ec is not None:
        fx.add(fx.whoosh(0.6, 300, 3000, 0.3, -0.4, 0.5), ec - 0.12, 0.40)
        fx.add(fx.verb(fx.impact() * 0.5, 0.2), ec + 0.30, 0.30)
        if ed.get("end_card", {}).get("button", "SUBSCRIBE"):
            fx.add(fx.verb(fx.pop(600, 1000)), ec + 1.0, 0.40, 0.3)
            fx.add(fx.click(), ec + 2.45, 0.6, 0.3)
            fx.add(fx.verb(fx.bell([1568, 2093, 2637], 1.5, 0.06), 0.5), ec + 2.50, 0.13, 0.3)
    for c in ed.get("sfx_extra", []):
        fx.add(fx.kind(c["kind"]), tl.t(c["t"]), c.get("gain", 0.4), c.get("pan", 0))
    out = fx.out
    T = fx.T(edl["total"]); fo = fx.T(0.4)
    out[T - fo:T] *= np.linspace(1, 0, fo)[:, None]
    out = out[:T]
    sf.write(job.p("sfx.wav"), out, SR, subtype="FLOAT")
    print(f"[sfx] built, peak {np.abs(out).max():.2f}")


def main(job_dir):
    build(Job(job_dir))
