"""Shared paths, job-folder layout and small helpers."""
import json, os, re, subprocess, glob
from functools import lru_cache

VEK_HOME = os.path.expanduser(os.environ.get("VEK_HOME", "~/.vek"))
MODELS = os.path.join(VEK_HOME, "models")
ASSETS = os.path.join(VEK_HOME, "assets")
PARAKEET = os.path.join(MODELS, "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8")
RVM = os.path.join(MODELS, "rvm_mobilenetv3_fp32.onnx")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRANDS = os.path.join(REPO, "brands")
FPS = 30


class Job:
    """A job folder holds everything for one video. Files:
    probe.json      source facts (size, duration, audio offset)
    audio48.wav     source audio, padded so t=0 == first video frame (48 kHz mono)
    audio16.wav     same at 16 kHz for speech recognition
    head.json       per-frame head track (source px): [cx, top_y, width]
    phrases.json    speech phrases split at pauses, each re-transcribed
    transcript.md   human-readable phrase map (what Claude reads to plan cuts)
    edit.json       THE EDIT SPEC (written by Claude): cuts, shots, graphics
    edl.json        computed segment timing (output timeline)
    voice*.wav      assembled / processed voice
    words.json      caption word timings on the output timeline
    music.wav sfx.wav stem_*.wav master.wav
    out/            renders, review sheets, deliverables
    """

    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.p("out"), exist_ok=True)

    def p(self, *a):
        return os.path.join(self.root, *a)

    def has(self, name):
        return os.path.exists(self.p(name))

    def load(self, name, default=None):
        if not self.has(name):
            if default is not None:
                return default
            raise FileNotFoundError(self.p(name))
        with open(self.p(name)) as f:
            return json.load(f)

    def save(self, name, obj):
        with open(self.p(name), "w") as f:
            json.dump(obj, f, indent=1)

    @property
    def source(self):
        pr = self.load("probe.json")
        return pr["source"]


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def load_brand(name_or_path):
    p = name_or_path
    if not os.path.exists(p) and os.path.exists(os.path.join(REPO, p)):   # e.g. examples/<slug>/brand.json
        p = os.path.join(REPO, p)
    if not os.path.exists(p):
        p = os.path.join(BRANDS, name_or_path if name_or_path.endswith(".json") else name_or_path + ".json")
    with open(p) as f:
        b = json.load(f)
    with open(os.path.join(BRANDS, "TEMPLATE.json")) as f:
        base = json.load(f)
    return deep_merge(base, b)


def deep_merge(a, b):
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(None)
def recognizer():
    import sherpa_onnx
    d = PARAKEET + "/"
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=d + "encoder.int8.onnx", decoder=d + "decoder.int8.onnx", joiner=d + "joiner.int8.onnx",
        tokens=d + "tokens.txt", model_type="nemo_transducer", num_threads=os.cpu_count() or 2)


def asr(samples16k, pad=0.3):
    """Transcribe float32 16 kHz mono samples. Returns (text, words[{w,s,e}])."""
    import numpy as np
    z = np.zeros(int(pad * 16000), np.float32)
    x = np.concatenate([z, samples16k.astype(np.float32), z])
    st = recognizer().create_stream()
    st.accept_waveform(16000, x)
    recognizer().decode_stream(st)
    r = st.result
    words = []
    durs = list(getattr(r, "durations", []) or [])
    for i, (tok, t) in enumerate(zip(r.tokens, r.timestamps)):
        d = durs[i] if i < len(durs) else 0.08
        t -= pad
        if tok.startswith(" ") or not words:
            words.append({"w": tok.strip(), "s": round(t, 3), "e": round(t + d, 3)})
        else:
            words[-1]["w"] += tok
            words[-1]["e"] = round(t + d, 3)
    return r.text.strip(), words


def energy_db(x, sr, hop_s=0.01, smooth=3):
    import numpy as np
    hop = int(sr * hop_s)
    n = len(x) // hop
    fr = x[:n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt((fr.astype(np.float64) ** 2).mean(1)) + 1e-9)
    if smooth > 1:
        db = np.convolve(db, np.ones(smooth) / smooth, "same")
    return db


def voiced_runs(db, thr=-41.0, merge=3, min_len=5):
    """Runs of frames above threshold (10 ms frames). Returns [[start,end],...] in frames."""
    v = db > thr
    runs = []
    i = 0
    n = len(v)
    while i < n:
        if v[i]:
            j = i
            while j < n and v[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    m = []
    for s, e in runs:
        if m and s - m[-1][1] < merge:
            m[-1][1] = e
        else:
            m.append([s, e])
    return [x for x in m if x[1] - x[0] >= min_len]


class Timeline:
    """Resolves time references used in edit.json.

    A time can be a number (output seconds) or a string:
      "w12"          start of caption word 12 (words.json index)
      "w12+0.2"      ...plus an offset (also "-")
      "B1.start"     output start of segment B1 ("B1.end" for its end)
      "title.in" / "title.out" / "title.end"   title card times
      "end"          total video length
    """

    def __init__(self, edl, words=None, edit=None):
        self.edl = edl
        self.segs = {s["name"]: s for s in edl["segs"]}
        self.words = words or []
        self.edit = edit or {}

    def t(self, ref):
        if ref is None:
            return None
        if isinstance(ref, (int, float)):
            return float(ref)
        s = str(ref).replace(" ", "")
        m = re.match(r"^(.+?)([+-]\d*\.?\d+)?$", s)
        base, off = m.group(1), float(m.group(2) or 0)
        if re.match(r"^w\d+$", base):
            v = self.words[int(base[1:])]["t"]
        elif base == "end":
            v = self.edl["total"]
        elif base.startswith("title."):
            tc = self.edit.get("title_card", {})
            v = self.t(tc[base.split(".")[1]])
        elif "." in base:
            name, edge = base.rsplit(".", 1)
            v = self.segs[name]["ostart"] if edge == "start" else self.segs[name]["oend"]
        else:
            v = float(base)
        return v + off
