"""Streaming source frames (+ person matte) on demand.

Nothing is cached to disk: each request is served from a forward-only ffmpeg stream that is
(re)opened near the requested frame, denoised (hqdn3d) and matted with Robust Video Matting.
RVM is recurrent, so every stream starts a few frames early to warm its state up."""
import subprocess, collections
import numpy as np
from .common import RVM, FPS

WARM = 12


class Stream:
    def __init__(self, src, start, W, H, matte, sess):
        self.W, self.H = W, H
        self.start = max(0, start)
        self.next = self.start
        self.cache = collections.OrderedDict()
        self.matte = matte
        self.sess = sess
        self.rec = [np.zeros([1, 1, 1, 1], np.float32)] * 4
        self.dr = np.array([min(0.5, 400 / max(W, H))], np.float32)
        t = self.start / FPS
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.4f}", "-i", src, "-vf",
               f"fps={FPS},scale={W}:{H}:flags=lanczos,hqdn3d=2.5:2:5:4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.done = False

    def _read(self):
        n = self.W * self.H * 3
        b = self.p.stdout.read(n)
        if len(b) < n:
            self.done = True
            return None
        f = np.frombuffer(b, np.uint8).reshape(self.H, self.W, 3)
        a = None
        if self.matte:
            src = (f.astype(np.float32) / 255).transpose(2, 0, 1)[None]
            fgr, pha, *self.rec = self.sess.run(None, {"src": src, "r1i": self.rec[0], "r2i": self.rec[1],
                                                        "r3i": self.rec[2], "r4i": self.rec[3], "downsample_ratio": self.dr})
            a = (pha[0, 0] * 255).clip(0, 255).astype(np.uint8)
        return f, a

    def get(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        while self.next <= idx and not self.done:
            r = self._read()
            if r is None:
                break
            self.cache[self.next] = r
            self.next += 1
            while len(self.cache) > 90:
                self.cache.popitem(last=False)
        if idx in self.cache:
            return self.cache[idx]
        return self.cache[next(reversed(self.cache))] if self.cache else None

    def can_serve(self, idx):
        return idx in self.cache or (not self.done and self.next <= idx <= self.next + 45)

    def close(self):
        try:
            self.p.kill()
        except Exception:
            pass


class FramePool:
    def __init__(self, probe, matte=True, max_streams=3):
        self.src = probe["source"]
        self.W, self.H = probe["work_w"], probe["work_h"]
        self.N = int(probe["duration"] * FPS) - 1
        self.matte = matte
        self.sess = None
        if matte:
            import onnxruntime as ort
            so = ort.SessionOptions(); so.intra_op_num_threads = 1
            self.sess = ort.InferenceSession(RVM, so, providers=["CPUExecutionProvider"])
        self.streams = []
        self.max = max_streams

    def get(self, idx):
        idx = int(max(0, min(self.N, idx)))
        for s in self.streams:
            if s.can_serve(idx):
                self.streams.remove(s); self.streams.append(s)
                r = s.get(idx)
                if r is not None:
                    return r
        s = Stream(self.src, idx - WARM, self.W, self.H, self.matte, self.sess)
        self.streams.append(s)
        while len(self.streams) > self.max:
            self.streams.pop(0).close()
        r = s.get(idx)
        if r is None:   # past the end: step back
            return self.get(idx - 1)
        return r
