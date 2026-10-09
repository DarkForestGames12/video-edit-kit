"""Stage 4b - original music bed, synthesised and locked to the edit.

The tempo grid is anchored so a downbeat lands exactly on the title-card slam (the "drop").
Before the drop: filtered, sparse intro (builds). After: full groove. Last bars: cadence that
resolves on a downbeat, then rings out under the end card. No samples -> no copyright claims.
"""
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve, butter, sosfilt
from .common import Job, Timeline

SR = 48000
MOODS = {
    #            bpm  swing  progression (roman, quality)                                   minor
    "uplifting": (105, 0.58, [(5, "maj9"), (7, "6add9"), (4, "m9"), (9, "m9")], False),
    "chill":     (86, 0.62, [(2, "m9"), (7, "13"), (0, "maj9"), (9, "m9")], False),
    "energetic": (118, 0.52, [(0, "add9"), (7, "add9"), (9, "m9"), (5, "maj9")], False),
    "serious":   (92, 0.55, [(9, "m9"), (5, "maj9"), (0, "maj9"), (7, "add9")], True),
    "warm":      (96, 0.60, [(0, "maj9"), (9, "m9"), (5, "maj9"), (7, "13")], False),
}
QUAL = {"maj9": [4, 7, 11, 14], "6add9": [4, 7, 9, 14], "m9": [3, 7, 10, 14], "13": [4, 10, 14, 21],
        "add9": [4, 7, 14, 16], "maj7": [4, 7, 11], "m7": [3, 7, 10], "7": [4, 7, 10]}


def mtof(m): return 440 * 2 ** ((m - 69) / 12)


class Synth:
    def __init__(self, length_s, seed=7):
        self.L = int(length_s * SR)
        self.rng = np.random.default_rng(seed)

    def buf(self): return np.zeros((self.L, 2))

    def T(self, s): return int(round(s * SR))

    def add(self, buf, x, t, gain=1.0, pan=0.0):
        i = self.T(t)
        if i < 0:
            x = x[-i:]; i = 0
        n = min(len(x), buf.shape[0] - i)
        if n <= 0: return
        l = np.cos((pan + 1) * np.pi / 4) * 1.414; r = np.sin((pan + 1) * np.pi / 4) * 1.414
        buf[i:i + n, 0] += x[:n] * gain * l
        buf[i:i + n, 1] += x[:n] * gain * r

    def lp(self, x, f, o=2): return sosfilt(butter(o, f, 'low', fs=SR, output='sos'), x)
    def hp(self, x, f, o=2): return sosfilt(butter(o, f, 'high', fs=SR, output='sos'), x)
    def bp(self, x, lo, hi, o=2): return sosfilt(butter(o, [lo, hi], 'band', fs=SR, output='sos'), x)

    def ep(self, m, dur, vel=1.0):
        f = mtof(m); n = self.T(dur + 0.6); t = np.arange(n) / SR
        I = 1.4 * np.exp(-t / 0.30) + 0.25
        tine = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * t))
        bell = 0.18 * np.exp(-t / 0.07) * np.sin(2 * np.pi * f * 14.0 * t)
        det = 0.5 * np.sin(2 * np.pi * f * 1.003 * t + 0.7 * np.exp(-t / 0.3) * np.sin(2 * np.pi * f * 1.003 * t))
        env = np.minimum(t / 0.006, 1) * np.exp(-t / 1.6) * np.clip(1 - (t - dur) / 0.35, 0, 1)
        return ((tine + det) / 1.5 + bell) * env * vel * (1 - 0.12 * np.sin(2 * np.pi * 4.2 * t))

    def bass(self, m, dur, vel=1.0):
        f = mtof(m); n = self.T(dur + 0.1); t = np.arange(n) / SR
        x = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t) + 0.08 * np.sin(6 * np.pi * f * t)
        env = np.minimum(t / 0.008, 1) * np.exp(-t / 0.9) * np.clip(1 - (t - dur) / 0.08, 0, 1)
        return np.tanh(1.6 * x * env) * vel * 0.8

    def kick(self, vel=1.0):
        n = self.T(0.45); t = np.arange(n) / SR
        ph = 2 * np.pi * np.cumsum(48 + 110 * np.exp(-t / 0.045)) / SR
        x = np.sin(ph) * np.exp(-t / 0.28) + 0.075 * np.exp(-t / 0.004) * self.rng.standard_normal(n)
        return np.tanh(1.5 * x) * vel

    def snare(self, vel=1.0):
        n = self.T(0.35); t = np.arange(n) / SR
        nz = self.bp(self.rng.standard_normal(n), 900, 7000) * np.exp(-t / 0.12)
        tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.06) * 0.6
        clap = np.zeros(n)
        for d in (0, 0.011, 0.022):
            i = self.T(d); m = self.T(0.03)
            clap[i:i + m] += self.bp(self.rng.standard_normal(m), 1000, 4000) * np.exp(-np.arange(m) / SR / 0.008)
        return (nz * 0.8 + tone + clap * 0.9) * vel * 0.6

    def hat(self, vel=1.0, open_=False):
        d = 0.32 if open_ else 0.05; n = self.T(d + 0.05); t = np.arange(n) / SR
        return self.hp(self.rng.standard_normal(n), 7500, 4) * np.exp(-t / (d / 2.5)) * vel * 0.35

    def shaker(self, vel=1.0):
        n = self.T(0.09); t = np.arange(n) / SR
        return self.bp(self.rng.standard_normal(n), 4000, 10000) * np.sin(np.pi * np.minimum(t / 0.09, 1)) ** 2 * vel * 0.18

    def pluck(self, m, dur=1.0, vel=1.0):
        f = mtof(m); N = int(SR / f); n = self.T(dur)
        b = self.rng.uniform(-1, 1, N)
        out = np.empty(n); idx = 0
        for i in range(n):
            out[i] = b[idx]; nx = (idx + 1) % N
            b[idx] = 0.996 * 0.5 * (b[idx] + b[nx]); idx = nx
        return out * vel * np.minimum(np.arange(n) / SR / 0.002, 1)

    def ir(self, dur=1.8, decay=2.2, damp=4000):
        n = self.T(dur); t = np.arange(n) / SR
        out = np.zeros((n + self.T(0.02), 2))
        for c in range(2):
            out[self.T(0.02):, c] = self.lp(self.rng.standard_normal(n) * np.exp(-t * decay), damp)
        return out / np.abs(out).max() * 0.25

    def reverb(self, st, wet):
        irr = self.ir()
        y = np.stack([fftconvolve(st[:, c], irr[:, c])[:st.shape[0]] for c in range(2)], 1)
        return st + y * wet


def chord_notes(root_pc, qual, key):
    r = (root_pc + key) % 12
    root = 48 + r if r <= 7 else 36 + r          # root note for voicing reference
    notes = [root + i for i in QUAL[qual]]
    notes = [n + 12 if n < 52 else n for n in notes]
    notes = [n - 12 if n > 74 else n for n in notes]
    bass = 36 + r if 36 + r >= 38 else 48 + r
    return sorted(notes), bass


def make(job, length, drop, endcard=None, mood="uplifting", bpm=None, key=0, body=True, seed=7):
    bpm0, swing, prog, minor = MOODS.get(mood, MOODS["uplifting"])
    bpm = bpm or bpm0
    BEAT = 60 / bpm; BAR = 4 * BEAT
    n_pre = int(np.floor(drop / BAR)) + 1
    T0 = drop - n_pre * BAR                      # downbeat of bar 0 (<= 0)
    nbars = int(np.ceil((length - T0) / BAR)) + 1
    S = Synth(length + 2.5, seed)
    ep, bas, drm, plk = S.buf(), S.buf(), S.buf(), S.buf()
    S16 = BEAT / 4

    def db(n): return T0 + n * BAR

    def st(n, s):
        t = db(n) + s * S16
        return t + ((swing - 0.5) * 2 * S16 if s % 2 else 0)
    # final resolution bar: last downbeat with >= 0.9 s before the end
    final = max(n for n in range(nbars) if db(n) <= length - 0.9)
    tonic = (9, "m9") if minor else (0, "maj9")
    pre = [(5, "maj9"), (7, "13")] if not minor else [(5, "maj9"), (7, "add9")]
    drop_bar = n_pre
    end_bar = None
    if endcard is not None:
        end_bar = min(n for n in range(nbars) if db(n) >= endcard - BAR * 0.5)
    for n in range(nbars):
        t0 = db(n)
        if t0 > length + 0.1: break
        if n == final:
            notes, b = chord_notes(*tonic, key)
            for k, m in enumerate(notes): S.add(ep, S.ep(m, 2.0, 0.55), t0 + k * 0.018, pan=-0.3 + 0.15 * k)
            S.add(bas, S.bass(b, 1.6), t0); S.add(drm, S.kick(), t0); S.add(drm, S.hat(0.8, True), t0)
            break
        intro = n < drop_bar
        quiet = (not body) and drop_bar + 2 <= n and (end_bar is None or n < end_bar)
        if quiet:
            continue
        if n == final - 1:
            chords = [(0, 1.9, chord_notes(*pre[0], key)), (2, 1.9, chord_notes(*pre[1], key))]
        else:
            ch = chord_notes(*prog[(n - 1) % 4], key) if n >= 1 else chord_notes(*prog[-1], key)
            chords = [(0, 3.8, ch)] if intro else [(0, 1.75, ch), (2.5, 1.2, ch)]
        for hb, hd, (notes, b) in chords:
            v = 0.5 if intro else (0.55 if hb == 0 else 0.42)
            for k, m in enumerate(notes):
                S.add(ep, S.ep(m, hd * BEAT, v), t0 + hb * BEAT + k * 0.012, pan=-0.35 + 0.17 * k)
        if intro:
            if n >= 1:
                for s in range(0, 16, 2): S.add(drm, S.hat(0.35 if s % 4 else 0.5), st(n, s), pan=0.25)
                S.add(drm, S.kick(0.55), st(n, 0))
            continue
        for bb, bd, v in [(0, 1.4, 1.0), (1.5, 0.4, 0.7), (2.5, 1.0, 0.9), (3.5, 0.4, 0.6)]:
            notes, b = chords[0][2] if (len(chords) == 1 or bb < 2) else chords[1][2]
            S.add(bas, S.bass(b + (7 if bb == 3.5 else 0), bd * BEAT, v), t0 + bb * BEAT)
        four = mood == "energetic"
        for s in range(16):
            if s in ((0, 4, 8, 12) if four else (0, 8)): S.add(drm, S.kick(1.0), st(n, s))
            if s == 11 and not four: S.add(drm, S.kick(0.45), st(n, s))
            if s in (4, 12): S.add(drm, S.snare(0.9), st(n, s))
            if s % 2 == 0: S.add(drm, S.hat(0.8 if s % 4 == 0 else 0.55), st(n, s), pan=0.25)
            if s in (7, 15): S.add(drm, S.hat(0.3), st(n, s), pan=0.25)
            if s == 14 and n % 2 == 0: S.add(drm, S.hat(0.45, True), st(n, s), pan=0.25)
            if mood != "serious" and n % 8 in (4, 5) and s % 2 == 1: S.add(drm, S.shaker(0.9), st(n, s), pan=-0.4)
    # pluck motif: pentatonic, on the drop bar and the end-card bar
    base = 72 + key if not minor else 69 + key
    pent = [base + i for i in ([0, 4, 7, 9, 12] if not minor else [0, 3, 7, 10, 12])]
    motif = [(0, pent[1]), (3, pent[2]), (6, pent[3]), (8, pent[4]), (11, pent[3]), (14, pent[2])]
    for n in [drop_bar] + ([end_bar] if end_bar is not None and end_bar < final else []):
        for s, m in motif: S.add(plk, S.pluck(m, 1.0, 0.5), st(n, s), pan=0.3)
    S.add(plk, S.pluck(pent[4], 2.0, 0.5), db(final) + 0.02, pan=0.3)
    ep = S.reverb(ep, 0.35); plk = S.reverb(plk, 0.45)
    # intro filter sweep (builds into the drop)
    ce = S.T(drop)
    if ce > 0:
        seg = ep[:ce].copy(); out = np.zeros_like(seg); blk = S.T(0.05)
        for i in range(0, ce, blk):
            fc = 700 * (5000 / 700) ** (i / ce)
            sos = butter(2, fc, 'low', fs=SR, output='sos'); j = min(i + blk, ce); s0 = max(0, i - S.T(0.05))
            y = np.stack([sosfilt(sos, seg[s0:j, c]) for c in range(2)], 1); out[i:j] = y[i - s0:]
        ep[:ce] = out
        drm[:ce] = np.stack([S.lp(drm[:ce, c], 3500) for c in range(2)], 1)
    pump = np.ones(S.L)
    for n in range(drop_bar, final + 1):
        for s in (0, 8):
            i = S.T(st(n, s)); m = S.T(0.25)
            if 0 <= i and i + m < S.L: pump[i:i + m] = np.minimum(pump[i:i + m], 1 - 0.35 * np.exp(-np.arange(m) / SR / 0.08))
    mus = ep * 0.55 * pump[:, None] + bas * 0.55 * pump[:, None] + drm * 0.5 + plk * 0.5
    cr = np.zeros(S.L); idx = S.rng.integers(0, S.L, size=int(length * 14)); cr[idx] = S.rng.standard_normal(len(idx)) * 0.08
    cr = S.hp(cr, 1500); mus += np.stack([cr, np.roll(cr, 37)], 1) * 0.6
    mus = np.stack([S.lp(mus[:, c], 11000) for c in range(2)], 1)
    if not body:   # smooth fades around the silent body
        g = np.ones(S.L)
        a0 = S.T(db(drop_bar + 2)); g[a0:a0 + S.T(BAR)] = np.linspace(1, 0, len(g[a0:a0 + S.T(BAR)]))
        if end_bar is not None:
            b0 = S.T(db(end_bar)); g[a0 + S.T(BAR):b0] = 0
        else:
            g[a0 + S.T(BAR):] = 0
        mus *= g[:, None]
    fi = S.T(0.25); mus[:fi] *= np.linspace(0, 1, fi)[:, None]
    end = S.T(length); mus[end:] = 0
    fo = S.T(0.5); mus[end - fo:end] *= (np.linspace(1, 0, fo) ** 2)[:, None]
    mus = mus[:end]
    mus /= np.abs(mus).max() + 1e-9
    mus *= 0.9
    sf.write(job.p("music.wav"), mus, SR, subtype="FLOAT")
    return dict(bpm=bpm, bar=BAR, t0=T0, drop_bar=drop_bar, final=db(final))


def from_file(job, path, length):
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-stream_loop", "-1", "-i", path, "-t", str(length), "-ac", "2",
                    "-ar", str(SR), "-af", f"afade=t=out:st={max(0, length - 1.5)}:d=1.5", "-c:a", "pcm_f32le",
                    job.p("music.wav")], check=True)


def main(job_dir):
    job = Job(job_dir)
    ed = job.load("edit.json"); edl = job.load("edl.json")
    from .common import load_brand
    brand = load_brand(ed["brand"])
    mc = dict(brand.get("music", {})); mc.update(ed.get("music", {}))
    tl = Timeline(edl, job.load("words.json", []), ed)
    if mc.get("file"):
        from_file(job, mc["file"], edl["total"]); print("[music] using provided track", mc["file"]); return
    tc = ed.get("title_card")
    drop = tl.t(tc["in"]) if tc else min(2 * 60 / (mc.get("bpm") or 100) * 4, edl["total"] / 3)
    info = make(job, edl["total"], drop, edl.get("endcard_in"), mc.get("mood", "uplifting"), mc.get("bpm"),
                int(mc.get("key", 0)), body=mc.get("body", ed.get("mode", "full") != "long"))
    print(f"[music] {mc.get('mood','uplifting')} {info['bpm']} bpm, drop on downbeat at {drop:.3f}s, "
          f"final chord at {info['final']:.2f}s")
