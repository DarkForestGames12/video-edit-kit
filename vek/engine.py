"""Stage 5 - render engine. Every frame is composed from edit.json:
footage (streamed, matted, graded, reframed per shot) -> hook text behind the subject ->
stickers / cards / icon burst -> captions -> title card -> end card -> vignette + grain.
Formats: 16:9 (1920x1080) and 9:16 (1080x1920)."""
import math, os, sys, subprocess, time, json, re
from functools import lru_cache
import numpy as np, cv2
from .common import Job, Timeline, load_brand, FPS
from .gfx import (text_sprite, icon_sprite, add_shadow, rounded_rect, circle_sprite, ring_sprite, blit, over, font,
                  hexc, rgbf, clamp, lin, e_out, e_in, e_io, e_back, smooth)
from .grade import Grader
from .reader import FramePool

cv2.setNumThreads(1)

LAYOUT = {
    "16:9": dict(W=1920, H=1080, py=0.40, cap_y=905, cap_chars=22, card=(110, 175, 600, 600),
                 card_shot=dict(px=0.71, zmin=1.15), hook="side", title="left", end="side"),
    "9:16": dict(W=1080, H=1920, py=0.44, cap_y=1300, cap_chars=15, card=(240, 150, 600, 540),
                 card_shot=dict(py=0.66, zmin=1.0), hook="top", title="center", end="stack"),
}
# 9:16 from a landscape source: the picture sits in a square window over a blurred copy of itself;
# hook text lives in the band above, captions below the chin, never over the mouth.
STACK = dict(view=(0, 330, 1080, 1080), py=0.40, cap_y=1545, cap_chars=15, card=(50, 372, 400, 400),
             card_shot=dict(px=0.72, zmin=1.12), hook="band")
STARTERS = {"so", "and", "but", "because", "in", "to", "with", "for", "when", "how", "what", "about", "then", "which", "that", "if", "or"}
WEAK_END = {"very", "the", "a", "an", "to", "in", "of", "my", "your", "our", "and", "so", "for", "at", "on", "with", "is", "it's", "i"}


class Engine:
    def __init__(self, job, simple_grade=False, matte=True):
        self.job = job
        self.ed = job.load("edit.json")
        self.edl = job.load("edl.json")
        self.words = job.load("words.json", [])
        self.brand = load_brand(self.ed["brand"])
        self.C = self.brand["colors"]
        self.F = self.brand["fonts"]
        self.probe = job.load("probe.json")
        hd = job.load("head.json")
        self.head = np.array(hd["rows"], np.float64)
        self.fmt = self.ed.get("format", "16:9")
        self.L = dict(LAYOUT[self.fmt])
        self.W, self.H = self.L["W"], self.L["H"]
        self.SW, self.SH = self.probe["work_w"], self.probe["work_h"]
        self.vlayout = None
        if self.fmt == "9:16":
            self.vlayout = self.ed.get("vertical_layout") or ("crop" if self.SH > self.SW else "stack")
            if self.vlayout == "stack":
                self.L.update(STACK)
        self.view = self.L.get("view", (0, 0, self.W, self.H))
        self.grader = Grader(self.brand.get("grade"))
        self.simple = simple_grade
        self.pool = FramePool(self.probe, matte=matte)
        self.tl = Timeline(self.edl, self.words, self.ed)
        self.total = self.edl["total"]
        self.endcard = self.edl.get("endcard_in")
        self.segs = self.edl["segs"]
        self.seg = {s["name"]: s for s in self.segs}
        self.order = [s["name"] for s in self.segs]
        self._setup_shots()
        self._setup_elements()
        self.gcache = {}
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        d = np.sqrt(((xx - self.W / 2) / (self.W / 2)) ** 2 + ((yy - self.H / 2) / (self.H / 2)) ** 2) / 1.414
        vs = self.brand["finish"]["vignette"]
        self.vig = (1 - vs * np.clip((d - 0.45) / 0.55, 0, 1) ** 1.6)[..., None].astype(np.float32)
        self.rng = np.random.default_rng(3)
        self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self._gx, self._gy = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32))

    # ------------------------------------------------------------------ shots / camera
    def _setup_shots(self):
        cards = [(self.tl.t(c["in"]), self.tl.t(c["out"])) for c in self.ed.get("cards", [])]
        for s in self.segs:
            sh = dict(s.get("shot") or {})
            z = sh.get("z", [1.0, 1.03])
            if isinstance(z, (int, float)): z = [z, z]
            px = sh.get("px", 0.5); py = sh.get("py", self.L["py"])
            overl = any(a < s["oend"] and b > s["ostart"] for a, b in cards)
            if overl and "px" not in sh and "py" not in sh:
                cs = self.L["card_shot"]
                px = cs.get("px", px); py = cs.get("py", py)
                z = [max(z[0], cs["zmin"]), max(z[1], cs["zmin"] + 0.03)]
            s["_z"], s["_px"], s["_py"] = z, px, py
            i0 = int(s["vin"] * FPS); i1 = max(i0 + 1, int(s["vout"] * FPS))
            rows = self.head[min(i0, len(self.head) - 1):min(i1, len(self.head))]
            if len(rows) == 0: rows = self.head[-1:]
            s["_anchor"] = np.median(rows, 0)          # cx, top, width, eye_y (work px)
        self.punches = [dict(t=self.tl.t(p["at"]), z=float(p["z"]), dur=float(p.get("dur", 0.16)))
                        for p in self.ed.get("punches", [])]

    def seg_at(self, t):
        for s in self.segs:
            if s["ostart"] - 1e-6 <= t < s["oend"] - 1e-6:
                return s
        return self.segs[-1] if t >= self.segs[-1]["ostart"] else self.segs[0]

    def cam(self, s, t):
        z0, z1 = s["_z"]
        u = clamp((t - s["ostart"]) / max(1e-3, s["oend"] - s["ostart"]))
        Z = z0 + (z1 - z0) * smooth(u)
        for p in self.punches:
            if s["ostart"] <= p["t"] < s["oend"] and t >= p["t"]:
                Z = Z * (1 + (p["z"] / z0 - 1) * e_out((t - p["t"]) / p["dur"]))
        cx, top, hw, eye = s["_anchor"]
        vx, vy, vw, vh = self.view
        asp = vw / vh
        bw = min(self.SW, self.SH * asp)
        cw = bw / Z; ch = cw / asp
        x0 = clamp(cx - s["_px"] * cw, 0, self.SW - cw); y0 = clamp(eye - s["_py"] * ch, 0, self.SH - ch)
        sc = vw / cw
        return np.array([[sc, 0, -sc * x0], [0, sc, -sc * y0]], np.float32), Z, sc   # window coords

    def head_screen(self, s, t):
        M, Z, sc = self.cam(s, t)
        cx, top, hw, eye = s["_anchor"]
        vx, vy = self.view[:2]
        return dict(x=M[0, 0] * cx + M[0, 2] + vx, y=M[1, 1] * eye + M[1, 2] + vy, w=hw * sc,
                    top=M[1, 1] * top + M[1, 2] + vy)

    # ------------------------------------------------------------------ footage
    def graded(self, idx):
        if idx not in self.gcache:
            if len(self.gcache) > 30:
                self.gcache.pop(next(iter(self.gcache)))
            fr, al = self.pool.get(idx)
            r = min(len(self.head) - 1, idx)
            if self.simple or al is None:
                g = self.grader.simple(fr)
            else:
                cx, top, hw, eye = self.head[r]
                g = self.grader.full(fr, al, cx, eye - 0.1 * hw)
            if al is None:
                al = np.zeros(fr.shape[:2], np.uint8)
            self.gcache[idx] = (g, al)
        return self.gcache[idx]

    def footage(self, s, t, src_t=None):
        st = s["vin"] + (t - s["ostart"]) if src_t is None else src_t
        g, al = self.graded(int(round(st * FPS)))
        M, Z, sc = self.cam(s, t)
        vx, vy, vw, vh = self.view
        f = cv2.warpAffine(g, M, (vw, vh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT).astype(np.float32) / 255
        a = cv2.warpAffine(al, M, (vw, vh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255
        bl = cv2.GaussianBlur(f, (0, 0), 1.2)
        f = np.clip(f + 0.25 * (f - bl), 0, 1)
        if (vw, vh) == (self.W, self.H):
            return f, a[..., None]
        # window over a blurred, darkened copy of itself
        k = 8
        small = cv2.resize(f, (vw // k, vh // k), interpolation=cv2.INTER_AREA)
        cover = max(self.W / vw, self.H / vh)
        sw_, sh_ = int(self.W / k), int(self.H / k)
        big = cv2.resize(small, (int(vw * cover / k) + 2, int(vh * cover / k) + 2), interpolation=cv2.INTER_LINEAR)
        oy = (big.shape[0] - sh_) // 2; ox = (big.shape[1] - sw_) // 2
        bgs = cv2.GaussianBlur(big[oy:oy + sh_, ox:ox + sw_], (0, 0), 5)
        bg = cv2.resize(bgs, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        bg = bg * 0.42 + rgbf(self.C["bg_edge"]) * 0.25
        bg[vy:vy + vh, vx:vx + vw] = f
        A = np.zeros((self.H, self.W, 1), np.float32)
        A[vy:vy + vh, vx:vx + vw, 0] = a
        return bg.astype(np.float32), A

    def morph(self, fa, fb, w):
        k = 4 if self.W >= 1920 or self.H >= 1920 else 2
        sz = (self.W // k, self.H // k)
        sa = cv2.cvtColor((cv2.resize(fa, sz, interpolation=cv2.INTER_AREA) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        sb = cv2.cvtColor((cv2.resize(fb, sz, interpolation=cv2.INTER_AREA) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        fl = cv2.GaussianBlur(self._dis.calc(sa, sb, None), (0, 0), 3)
        fl = cv2.resize(fl, (self.W, self.H), interpolation=cv2.INTER_LINEAR) * k
        wa = cv2.remap(fa, self._gx - w * fl[..., 0], self._gy - w * fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        wb = cv2.remap(fb, self._gx + (1 - w) * fl[..., 0], self._gy + (1 - w) * fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return wa * (1 - w) + wb * w

    # ------------------------------------------------------------------ elements setup
    def col(self, c):
        if c is None: return self.C["text"]
        return self.C.get(c, c)

    def _setup_elements(self):
        tl = self.tl
        ed = self.ed
        self.whips = [s["ostart"] for s in self.segs if s["enter"] == "whip"]
        # captions ------------------------------------------------------
        cap = ed.get("captions", {})
        for k, v in cap.get("fix", {}).items():          # caption spelling fixes: {"55": "advantages"}
            if int(k) < len(self.words): self.words[int(k)]["w"] = v
        skip = set(int(i) for i in cap.get("skip", []))
        kw = {}
        for k, v in cap.get("keywords", {}).items():
            if str(k).isdigit():
                kw[int(k)] = v
            else:
                for w in self.words:
                    if re.sub(r"[^\w']", "", w["w"]).upper() == re.sub(r"[^\w']", "", k).upper():
                        kw[w["i"]] = v
        self.kw = kw
        hide = []
        tc = ed.get("title_card")
        if tc: hide.append((tl.t(tc["in"]), tl.t(tc["end"])))
        hide += [(tl.t(a), tl.t(b)) for a, b in cap.get("hide", [])]
        self.cap_hide = hide
        words = [w for w in self.words if w["i"] not in skip]
        if cap.get("groups"):
            groups = [[self.words[i] for i in g if i not in skip] for g in cap["groups"]]
            groups = [g for g in groups if g]
        else:
            groups = self._auto_groups(words)
        self.chunks = []
        for gi, g in enumerate(groups):
            st = g[0]["t"] - 0.07
            en = groups[gi + 1][0]["t"] - 0.07 if gi + 1 < len(groups) else (self.endcard or self.total) - 0.02
            en = min(en, g[-1]["t"] + 0.95)
            for a, b in hide + ([(self.endcard, self.total)] if self.endcard else []):
                if st < a < en: en = a
            self.chunks.append(dict(words=g, st=st, en=en))
        # stickers / cards / burst / hook / title / end ------------------
        self.stickers = []
        for k, s in enumerate(ed.get("stickers", [])):
            ti, to = tl.t(s["in"]), tl.t(s["out"])
            tm = min(to, ti + 0.3)                       # place by the framing it will mostly sit on
            sg = self.seg_at(tm)
            h = self.head_screen(sg, tm)
            side = s.get("side", "right")
            sgn = 1 if side == "right" else -1
            if self.fmt == "16:9" or self.vlayout == "stack":
                x = h["x"] + sgn * 0.9 * h["w"]; y = h["y"] - 0.2 * h["w"]
            else:
                x = h["x"] + sgn * 0.42 * h["w"]; y = h["top"] - 20
            x += s.get("dx", 0); y += s.get("dy", 0)
            vx, vy, vw, vh = self.view
            x = clamp(x, vx + 100, vx + vw - 100); y = clamp(y, max(110, vy + 100), min(self.H - 300, vy + vh - 100))
            self.stickers.append(dict(ti=ti, to=to, icon=s["icon"], color=s.get("color", self.C["accent"]),
                                      x=x, y=y, mode=s.get("mode", "pop"), size=s.get("size", 172)))
        self.cards = []
        for c in ed.get("cards", []):
            self.cards.append(dict(kind=c.get("kind", "plain"), icon=c["icon"], label=c.get("label", ""),
                                   color=self.col(c.get("color", "accent2")), ti=tl.t(c["in"]), to=tl.t(c["out"]),
                                   beats=[tl.t(b) for b in c.get("beats", [])], ecg=c.get("ecg", False),
                                   pop=tl.t(c["pop"]) if c.get("pop") is not None else None, check=c.get("check", True)))
        b = ed.get("burst")
        self.burst = None
        if b:
            self.burst = dict(ti=tl.t(b["in"]), to=tl.t(b["out"]), icons=b["icons"])
        self.hook = None
        ht = ed.get("hook_text")
        if ht:
            self.hook = dict(lines=ht["lines"], ti=tl.t(ht.get("in", 0.06)), to=tl.t(ht.get("out")),
                             tuck=float(ht.get("tuck", 0.22)))
            self._layout_hook()
        self.title = None
        if tc:
            self.title = dict(ti=tl.t(tc["in"]), to=tl.t(tc["out"]), te=tl.t(tc["end"]), kicker=tc.get("kicker", ""),
                              text=tc.get("title", ""), sub=tc.get("sub", ""))
        ec = ed.get("end_card")
        self.ecard = None
        if ec and self.endcard is not None:
            d = dict(self.brand["end_card"]); d.update(ec)
            self.ecard = d

    def _auto_groups(self, words):
        """Caption lines: split at pauses/punctuation first, then split long phrases where it reads
        best (balanced, never ending a line on a weak word like 'very', 'your', 'to')."""
        maxc, maxw = self.L["cap_chars"], 4
        clean = lambda w: re.sub(r"[^\w']", "", w["w"]).lower()
        bounds = list(self.whips) + [x for ab in self.cap_hide for x in ab]
        crosses = lambda t1, t2: any(t1 < x <= t2 for x in bounds)
        phrases, cur = [], []
        for w in words:
            if cur:
                prev = cur[-1]
                cut = (w["t"] - prev["t"] > 0.42 or re.search(r"[.?!,;:]$", prev["w"]) or crosses(prev["t"], w["t"]))
                if cut:
                    phrases.append(cur); cur = []
            cur.append(w)
        if cur: phrases.append(cur)

        def fits(g): return len(g) <= maxw and len(" ".join(x["w"] for x in g)) <= maxc

        def split(g):
            if fits(g) or len(g) == 1: return [g]
            best, bk = None, 1
            for k in range(1, len(g)):
                L_, R_ = g[:k], g[k:]
                lc, rc = len(" ".join(x["w"] for x in L_)), len(" ".join(x["w"] for x in R_))
                cost = abs(lc - rc) * 0.5 + (30 if clean(L_[-1]) in WEAK_END else 0) - 20 * (g[k]["t"] - g[k - 1]["t"])
                cost += 15 * (not fits(L_)) + 15 * (not fits(R_)) - (6 if clean(R_[0]) in STARTERS else 0)
                if best is None or cost < best: best, bk = cost, k
            return split(g[:bk]) + split(g[bk:])
        out = []
        for p in phrases: out += split(p)
        # a lone short/weak word ("to", "it") joins its neighbour when the line still fits
        k = 0
        while k < len(out):
            g = out[k]
            if len(g) == 1 and (clean(g[0]) in WEAK_END or len(clean(g[0])) <= 3):
                if k + 1 < len(out) and fits(g + out[k + 1]) and out[k + 1][0]["t"] - g[0]["t"] < 1.0 \
                        and not re.search(r"[.?!]$", g[0]["w"]) and not crosses(g[0]["t"], out[k + 1][0]["t"]):
                    out[k + 1] = g + out[k + 1]; out.pop(k); continue
                if k and fits(out[k - 1] + g) and g[0]["t"] - out[k - 1][-1]["t"] < 1.0 \
                        and not re.search(r"[.?!]$", out[k - 1][-1]["w"]) and not crosses(out[k - 1][-1]["t"], g[0]["t"]):
                    out[k - 1] = out[k - 1] + g; out.pop(k); continue
            k += 1
        return out

    # ------------------------------------------------------------------ hook text behind subject
    def _layout_hook(self):
        hk = self.hook
        s = self.seg_at(hk["ti"])
        h = self.head_screen(s, hk["ti"] + 0.3)
        fspec = self.F["display"]
        n = len(hk["lines"])
        if self.L["hook"] == "side":
            XR = clamp(h["x"] - hk.get("tuck", 0.22) * h["w"], 0.35 * self.W, 0.72 * self.W)
            top, bottom = 60, self.L["cap_y"] - 95
            pitch = (bottom - top) / n
            size = int(pitch * 1.06)
            while size > 60:
                f = font(fspec, size)
                wid = max(sum(f.getlength(c) - size * 0.015 for c in ln) for ln in hk["lines"])
                if wid <= XR - 60: break
                size -= 8
            hk.update(size=size, base=[top + pitch * (k + 1) - 5 for k in range(n)], xr=XR, align="right")
        elif self.L["hook"] == "band":
            hk["lines"] = ["".join(hk["lines"])]
            size = 270
            while size > 60:
                f = font(fspec, size)
                if sum(f.getlength(c) - size * 0.015 for c in hk["lines"][0]) <= self.W - 100: break
                size -= 6
            vy = self.view[1]
            hk.update(size=size, base=[vy + 0.16 * size], xr=self.W / 2, align="center", color=self.C["accent"])
        else:
            size = 400
            while size > 60:
                f = font(fspec, size)
                wid = max(sum(f.getlength(c) - size * 0.015 for c in ln) for ln in hk["lines"])
                if wid <= self.W - 90: break
                size -= 8
            pitch = size * 0.94
            last = h["top"] + 0.30 * size
            first = last - pitch * (n - 1)
            asc = font(fspec, size).getmetrics()[0]
            if first - asc < 70:
                d = 70 - (first - asc); first += d; last += d
            hk.update(size=size, base=[first + pitch * k for k in range(n)], xr=self.W / 2, align="center")

    @lru_cache(None)
    def letter(self, ch, size, col):
        return text_sprite(ch, self.F["display"], size, col, pad=8)

    def hook_layer(self, t):
        hk = self.hook
        if not hk or t > hk["to"] + 0.45:
            return None
        L = np.zeros((self.H, self.W, 4), np.float32)
        size = hk["size"]
        f = font(self.F["display"], size); asc, desc = f.getmetrics()
        k = 0
        col = hk.get("color", self.C["behind_text"])
        first_x0 = None
        for ln, by in zip(hk["lines"], hk["base"]):
            adv = [f.getlength(ch) - size * 0.015 for ch in ln]
            wid = sum(adv)
            cx = hk["xr"] - wid if hk["align"] == "right" else hk["xr"] - wid / 2
            if first_x0 is None: first_x0 = cx
            for ch, a in zip(ln, adv):
                t0 = hk["ti"] + 0.05 * k
                u = e_out(lin(t, t0, t0 + 0.42))
                ex = e_in(lin(t, hk["to"] + 0.03 * k, hk["to"] + 0.35 + 0.03 * k))
                top = by - asc - 8
                blit(L, self.letter(ch, size, col), cx, top + (1 - u) * size * 0.6 + ex * size * 0.4, ax=0, ay=0,
                     op=u * (1 - ex), clip=(0, by - asc - 40, self.W, by + desc + (size if ex > 0 else 10)))
                cx += a
                k += 1
        u = e_out(lin(t, hk["ti"] + 0.5, hk["ti"] + 0.9)); ex = e_in(lin(t, hk["to"], hk["to"] + 0.3))
        if u > 0 and hk["align"] == "right":
            bar = rounded_rect(int(10 + size * 0.65 * u), 20, 10, self.C["accent"])
            blit(L, bar, first_x0 + 8, hk["base"][0] + 32, ax=0, ay=0.5, op=1 - ex)
        return L

    # ------------------------------------------------------------------ captions
    @lru_cache(None)
    def cap_word(self, txt, col):
        size = self.brand["captions"]["size_169" if self.fmt == "16:9" else "size_916"]
        st = self.brand["captions"]["stroke"]
        return text_sprite(txt, self.F["caption"], size, col, stroke=st, stroke_fill="#06110F", pad=14,
                           shadow=(12, 0, 7, 0.55))

    def captions(self, dst, t):
        up = self.brand["captions"]["uppercase"]
        for c in self.chunks:
            if not (c["st"] <= t < c["en"]):
                continue
            if any(a <= t < b for a, b in self.cap_hide):
                continue
            ws = [(w["w"].upper() if up else w["w"]) for w in c["words"]]
            sprs = [self.cap_word(w, self.C["text"]) for w in ws]
            sp = 34
            widths = [s.shape[1] - 28 - 62 for s in sprs]
            total = sum(widths) + sp * (len(ws) - 1)
            scale_fit = min(1.0, (self.W - 120) / max(1, total))
            x = self.W / 2 - total * scale_fit / 2
            outk = e_in(lin(t, c["en"] - 0.07, c["en"]))
            for k, w in enumerate(c["words"]):
                ts = w["t"] - 0.06 if k else min(w["t"] - 0.06, c["st"] - 0.03)
                pop = lin(t, ts, ts + 0.16)
                if pop > 0:
                    sc = (0.55 + 0.45 * e_back(pop, 2.2)) * scale_fit
                    nxt = c["words"][k + 1]["t"] - 0.06 if k + 1 < len(c["words"]) else c["en"]
                    active = t < nxt
                    key = self.kw.get(w["i"])
                    colr = self.col(key) if key else (self.C["accent"] if active else self.C["text"])
                    if active: sc *= 1.04
                    sc *= (1 - 0.25 * outk)
                    blit(dst, self.cap_word(ws[k], colr), x + widths[k] * scale_fit / 2, self.L["cap_y"], scale=sc,
                         op=(1 - outk) * clamp(pop * 3), ax=0.5, ay=0.62)
                x += (widths[k] + sp) * scale_fit

    # ------------------------------------------------------------------ stickers
    @lru_cache(None)
    def sticker(self, name, colr, d):
        base = circle_sprite(d, self.C["text"] if self.C["text"].upper() != "#000000" else "#FFFFFF")
        blit(base, icon_sprite(name, int(d * 0.50), colr), d / 2, d / 2)
        return add_shadow(base, 14, 0, 8, 0.45)

    def draw_stickers(self, dst, t):
        for s in self.stickers:
            if not (s["ti"] <= t <= s["to"]): continue
            u = lin(t, s["ti"], s["ti"] + 0.28)
            sc = e_back(u, 2.6) * (1 - e_in(lin(t, s["to"] - 0.12, s["to"])))
            rot = 0.0; y = s["y"]
            dt = t - s["ti"]
            if s["mode"] == "wave":
                rot = 18 * math.sin(dt * 2 * math.pi * 2.2) * (1 - 0.5 * lin(t, s["ti"], s["to"]))
            elif s["mode"] == "bob":
                y -= 10 * math.sin(dt * 6); rot = -8 + 6 * math.sin(dt * 5)
            elif s["mode"] == "spin":
                rot = -25 * (1 - e_out(u)) + 6 * math.sin(dt * 6)
            blit(dst, self.sticker(s["icon"], s["color"], s["size"]), s["x"], y + (1 - e_out(u)) * 40, scale=sc,
                 rot=rot, op=clamp(u * 4))

    # ------------------------------------------------------------------ cards
    @lru_cache(None)
    def card_base(self, accent):
        cx, cy, cw, ch = self.L["card"]
        ink = hexc(self.C["ink"], 0.87)
        base = rounded_rect(cw, ch, 46, ink, border=3, border_fill=hexc(accent, 0.55))
        return add_shadow(base, 30, 0, 18, 0.55)

    @lru_cache(None)
    def card_icon(self, name, colr, size):
        return icon_sprite(name, size, colr)

    @lru_cache(None)
    def card_glow(self, name, colr, size):
        ic = icon_sprite(name, size, colr); m = 60
        out = np.zeros((ic.shape[0] + 2 * m, ic.shape[1] + 2 * m, 4), np.float32); out[m:-m, m:-m] = ic
        return cv2.GaussianBlur(out, (0, 0), 22) * 1.6

    @lru_cache(None)
    def card_label(self, txt):
        size = int(self.L["card"][3] * 0.087)
        while size > 18 and font(self.F["caption"], size).getlength(txt) + 2 * len(txt) > self.L["card"][2] - 70:
            size -= 2
        return text_sprite(txt, self.F["caption"], size, self.C["text"], tracking=2, pad=6)

    @lru_cache(None)
    def check_badge(self):
        b = circle_sprite(78, self.C["accent"])
        blit(b, icon_sprite("check", 40, self.C["ink"]), 39, 40)
        return add_shadow(b, 8, 0, 4, 0.5)

    @staticmethod
    def ecg_y(ph):
        p = ph % 1.0
        if p < 0.10: return 0.10 * math.sin(p / 0.10 * math.pi)
        if p < 0.16: return 0
        if p < 0.19: return -0.25 * (p - 0.16) / 0.03
        if p < 0.23: return -0.25 + 1.35 * (p - 0.19) / 0.04
        if p < 0.27: return 1.10 - 1.7 * (p - 0.23) / 0.04
        if p < 0.30: return -0.60 + 0.60 * (p - 0.27) / 0.03
        if 0.48 <= p < 0.62: return 0.22 * math.sin((p - 0.48) / 0.14 * math.pi)
        return 0

    def draw_card(self, dst, c, t, ox, op):
        X, Y, CW, CH = self.L["card"]
        base = self.card_base(c["color"])
        m = (base.shape[1] - CW) // 2
        blit(dst, base, X + ox - m, Y - m + 4, ax=0, ay=0, op=op)
        cx = X + ox + CW / 2; cy = Y + 0.36 * CH
        isz = int(CH * 0.32)
        s = 1.0
        if c["kind"] == "pulse":
            for bt in c["beats"]:
                if bt <= t < bt + 0.5:
                    s += 0.16 * math.exp(-(t - bt) / 0.08) * (1 if t - bt < 0.25 else 0.6)
                    if t - bt > 0.2: s += 0.10 * math.exp(-(t - bt - 0.2) / 0.08)
        elif c["pop"] is not None and t >= c["pop"]:
            s += 0.18 * math.exp(-(t - c["pop"]) / 0.12)
            if c["kind"] == "rings":
                for k in range(3):
                    rt = c["pop"] + 0.12 * k
                    if rt <= t < rt + 0.9:
                        u = (t - rt) / 0.9
                        blit(dst, ring_sprite(int(CH * 0.7), c["color"]), cx, cy, scale=0.45 + 0.75 * e_out(u), op=op * (1 - u) * 0.8)
        if c["kind"] != "pulse":
            s *= 1 + 0.025 * math.sin((t - c["ti"]) * 4)
        blit(dst, self.card_glow(c["icon"], c["color"], isz), cx, cy, scale=s, op=op * 0.55)
        blit(dst, self.card_icon(c["icon"], c["color"], isz), cx, cy, scale=s, op=op)
        if c["kind"] == "pulse" and c["ecg"]:
            x0, x1 = X + ox + 50, X + ox + CW - 50
            yb = Y + 0.68 * CH
            N = 140
            reveal = e_out(lin(t, c["ti"] + 0.1, c["ti"] + 0.6))
            nv = int(N * reveal)
            if nv > 2 and op > 0:
                pts = []
                for i in range(nv):
                    u = i / N
                    pts.append((x0 + (x1 - x0) * u, yb - 70 * self.ecg_y(u * 1.6 - (t - c["ti"]) * 1.25)))
                layer = np.zeros((220, CW, 4), np.float32)
                p = np.array([[px - (X + ox), py - (yb - 120)] for px, py in pts], np.float32)
                cl = tuple(float(v) for v in rgbf(c["color"])) + (1.0,)
                cv2.polylines(layer, [np.round(p * 4).astype(np.int32)], False, cl, 6, cv2.LINE_AA, shift=2)
                blit(dst, cv2.GaussianBlur(layer, (0, 0), 6) * 0.9, X + ox, yb - 120, ax=0, ay=0, op=op)
                blit(dst, layer, X + ox, yb - 120, ax=0, ay=0, op=op)
                blit(dst, circle_sprite(16, "#FFFFFF"), pts[-1][0], pts[-1][1], op=op)
        if c["kind"] == "rings" and c["check"] and c["pop"] is not None:
            u = lin(t, c["pop"] + 0.05, c["pop"] + 0.35)
            if u > 0: blit(dst, self.check_badge(), cx + isz * 0.48, cy + isz * 0.38, scale=e_back(u, 2.5), op=op)
        if c["label"]:
            blit(dst, self.card_label(c["label"]), cx, Y + 0.875 * CH, op=op)

    def draw_cards(self, dst, t):
        X, Y, CW, CH = self.L["card"]
        far = X + CW + 60
        for c in self.cards:
            if not (c["ti"] - 0.01 <= t < c["to"] + 0.25): continue
            u = e_out(lin(t, c["ti"], c["ti"] + 0.38)); ex = e_in(lin(t, c["to"] - 0.04, c["to"] + 0.18))
            if self.fmt == "16:9" or self.vlayout == "stack":
                ox = -(1 - u) * far - ex * far
                self.draw_card(dst, c, t, ox, clamp(u * 2) * (1 - ex))
            else:
                lay = np.zeros((self.H, self.W, 4), np.float32)
                self.draw_card(lay, c, t, 0, 1.0)
                sc = 0.6 + 0.4 * e_back(lin(t, c["ti"], c["ti"] + 0.38), 1.8)
                sc *= (1 - 0.3 * ex)
                M = cv2.getRotationMatrix2D((X + CW / 2, Y + CH / 2), 0, sc)
                lay = cv2.warpAffine(lay, M, (self.W, self.H))
                over(dst, lay, clamp(u * 2) * (1 - ex))

    # ------------------------------------------------------------------ icon burst
    @lru_cache(None)
    def mini(self, name, colr):
        d = 128
        base = circle_sprite(d, "#FFFFFF")
        blit(base, icon_sprite(name, 62, colr), d / 2, d / 2)
        return add_shadow(base, 12, 0, 7, 0.45)

    def draw_burst(self, dst, t):
        b = self.burst
        if not b or not (b["ti"] - 0.02 <= t < b["to"]): return
        s = self.seg_at(t)
        h = self.head_screen(s, t)
        R = min(0.84 * h["w"], 0.42 * self.W)
        n = len(b["icons"])
        for k, (ic, colr) in enumerate(b["icons"]):
            ang = math.radians(192 + k * (156 / max(1, n - 1)))
            ti = b["ti"] + 0.07 * k
            u = lin(t, ti, ti + 0.3)
            if u <= 0: continue
            ex = e_in(lin(t, b["to"] - 0.19, b["to"] - 0.02))
            rr = R * (0.5 + 0.5 * e_out(u)) + 10 * math.sin((t - ti) * 3 + k)
            x = h["x"] + rr * math.cos(ang) * 1.22
            y = max(self.view[1] + 95, h["y"] - 10 + rr * math.sin(ang) * 0.70)
            x = clamp(x, 80, self.W - 80)
            blit(dst, self.mini(ic, self.col(colr)), x, y, scale=e_back(u, 2.4) * (1 - ex),
                 rot=8 * math.sin((t - ti) * 4 + k), op=clamp(u * 4) * (1 - ex))

    # ------------------------------------------------------------------ full-screen card pieces
    @lru_cache(None)
    def bg_plate(self):
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        cxr = 0.35 if self.fmt == "16:9" else 0.5
        d = np.sqrt(((xx - self.W * cxr) / self.W) ** 2 + ((yy - self.H * 0.45) / self.H) ** 2)
        k = np.clip(d / 0.75, 0, 1)[..., None]
        return (rgbf(self.C["bg_center"]) * (1 - k) + rgbf(self.C["bg_edge"]) * k).astype(np.float32)

    def grid(self, dst, t):
        off = (t * 18) % 80
        g = np.zeros((self.H, self.W), np.float32)
        for x in np.arange(-80, self.W + 80, 80):
            cv2.line(g, (int(x + off), 0), (int(x + off), self.H), 1.0, 1, cv2.LINE_AA)
        for y in np.arange(-80, self.H + 80, 80):
            cv2.line(g, (0, int(y + off * 0.5)), (self.W, int(y + off * 0.5)), 1.0, 1, cv2.LINE_AA)
        a = g[..., None] * 0.06
        dst[...] = dst * (1 - a) + rgbf(self.C["accent"]) * a

    @lru_cache(None)
    def floater(self, name, size): return icon_sprite(name, size, self.C["accent"])

    def floaters(self, dst, t):
        names = self.brand.get("floaters", [])
        pos = [(0.80, 0.16), (0.90, 0.82), (0.14, 0.83), (0.06, 0.13), (0.56, 0.89)] if self.fmt == "16:9" else \
              [(0.80, 0.08), (0.88, 0.90), (0.12, 0.92), (0.10, 0.07), (0.50, 0.95)]
        for k, ic in enumerate(names[:5]):
            px, py = pos[k]; ph = k * 0.7 + 0.3
            blit(dst, self.floater(ic, [150, 120, 110, 100, 90][k]), px * self.W + 14 * math.sin(t * 0.8 + ph),
                 py * self.H + 18 * math.sin(t * 0.6 + ph * 2), rot=10 * math.sin(t * 0.5 + ph), op=0.07)

    def motif(self, dst, t, cx, cy, length, amp, op, scale=1.0, horizontal=False):
        kind = self.brand.get("title_motif", "helix")
        if kind == "none" or op <= 0: return
        lay = np.zeros((self.H, self.W, 4), np.float32)
        A = tuple(float(v) for v in rgbf(self.C["accent"])); B = tuple(float(v) for v in rgbf(self.C["accent2"]))
        P = lambda x, y: (int(x * 4), int(y * 4))
        if kind == "helix":
            n = 26; pts = []
            for i in range(n):
                v = i / (n - 1); ang = v * 3.4 * math.pi + t * 1.6
                along = -length / 2 + v * length
                for strand in (0, 1):
                    a = ang + strand * math.pi
                    across = amp * math.sin(a) * scale
                    x, y = (cx + along, cy + across) if horizontal else (cx + across, cy + along)
                    pts.append((math.cos(a), x, y, strand, i))
            for i in range(n):
                (z0, x0, y0, _, _), (z1, x1, y1, _, _) = [p for p in pts if p[4] == i]
                c = 0.35 + 0.25 * (min(z0, z1) + 1) / 2
                cv2.line(lay, P(x0, y0), P(x1, y1), (A[0] * c, A[1] * c, A[2] * c, c), 3, cv2.LINE_AA, shift=2)
            for z, x, y, strand, i in sorted(pts, key=lambda p: p[0]):
                r = 9 + 7 * (z + 1) / 2; b = 0.45 + 0.55 * (z + 1) / 2
                cc = A if strand == 0 else B
                cv2.circle(lay, P(x, y), int(r * 4), (cc[0] * b, cc[1] * b, cc[2] * b, 1.0), -1, cv2.LINE_AA, shift=2)
        elif kind == "orbit":
            R = amp * 1.6 * scale
            for k in range(3):
                ang = k * math.pi / 3 + t * 0.4
                axes = (int(R * 4), int(R * 0.38 * 4))
                cv2.ellipse(lay, P(cx, cy), axes, math.degrees(ang), 0, 360, (A[0] * .6, A[1] * .6, A[2] * .6, .6), 3, cv2.LINE_AA, shift=2)
                ph = t * (1.5 + 0.4 * k) + k
                ex, ey = R * math.cos(ph), R * 0.38 * math.sin(ph)
                x = cx + ex * math.cos(ang) - ey * math.sin(ang); y = cy + ex * math.sin(ang) + ey * math.cos(ang)
                cv2.circle(lay, P(x, y), 4 * 14, B + (1.0,), -1, cv2.LINE_AA, shift=2)
            cv2.circle(lay, P(cx, cy), 4 * 30, A + (1.0,), -1, cv2.LINE_AA, shift=2)
        elif kind == "bars":
            nb = 18; bw = length / nb
            for i in range(nb):
                hgt = amp * 1.6 * (0.25 + 0.75 * abs(math.sin(t * 2.2 + i * 0.7) * math.sin(t * 1.3 + i * 0.31))) * scale
                x = cx - length / 2 + i * bw
                cc = A if i % 2 == 0 else B
                cv2.rectangle(lay, P(x + 4, cy - hgt / 2), P(x + bw - 4, cy + hgt / 2), cc + (1.0,), -1, cv2.LINE_AA, shift=2)
        over(dst, cv2.GaussianBlur(lay, (0, 0), 14), op * 0.5)
        over(dst, lay, op)

    # ------------------------------------------------------------------ title card
    @lru_cache(None)
    def title_sprites(self, kicker, sub):
        return dict(kicker=text_sprite(kicker, self.F["label"], 44, self.C["accent"], tracking=7, pad=6) if kicker else None,
                    sub=text_sprite(sub, self.F["sub"], 60, self.C["soft"], tracking=3, pad=6) if sub else None)

    def draw_title(self, dst, t):
        T = self.title
        if not T or not (T["ti"] - 0.001 <= t < T["te"]): return
        ti, to, te = T["ti"], T["to"], T["te"]
        card = self.bg_plate().copy()
        self.grid(card, t); self.floaters(card, t)
        S = self.title_sprites(T["kicker"], T["sub"])
        word = T["text"]
        size = 300
        f = font(self.F["display"], size)
        maxw = (0.62 if self.fmt == "16:9" else 0.88) * self.W
        while f.getlength(word) > maxw and size > 80:
            size -= 10; f = font(self.F["display"], size)
        asc, desc = f.getmetrics()
        tw = f.getlength(word)
        if self.fmt == "16:9":
            x0 = 132; ytop = 350; kx = 140; ky = 335; sy = 870; ax = 0
            self.motif(card, t, 1540, 540, 760, 150, e_out(lin(t, ti + 0.15, ti + 0.6)), e_out(lin(t, ti + 0.1, ti + 0.7)))
        else:
            x0 = (self.W - tw) / 2; ytop = 760; kx = self.W / 2; ky = 745; sy = 760 + asc + desc + 150; ax = 0.5
            self.motif(card, t, self.W / 2, 1520, 760, 120, e_out(lin(t, ti + 0.15, ti + 0.6)),
                       e_out(lin(t, ti + 0.1, ti + 0.7)), horizontal=True)
        u = e_out(lin(t, ti + 0.12, ti + 0.5))
        if S["kicker"] is not None:
            blit(card, S["kicker"], kx, ky + (1 - u) * 50, ax=ax, ay=1, op=u, clip=(0, ky - 65, self.W, ky + 10))
        x = x0
        for k, ch in enumerate(word):
            uu = e_out(lin(t, ti + 0.18 + 0.04 * k, ti + 0.62 + 0.04 * k))
            blit(card, self.letter(ch, size, self.C["text"]), x, ytop + (1 - uu) * size * 1.1, ax=0, ay=0, op=1,
                 clip=(0, ytop, self.W, ytop + asc + desc + 16))
            x += f.getlength(ch) - 2
        bw = x - x0
        ul_y = ytop + asc + desc * 0.2 + 18
        u = e_out(lin(t, ti + 0.55, ti + 0.95))
        if u > 0:
            blit(card, rounded_rect(int(8 + (bw - 10) * u), 18, 9, self.C["accent"]), x0 + 8 if ax == 0 else self.W / 2,
                 ul_y, ax=ax, ay=0.5)
        u = e_out(lin(t, ti + 0.62, ti + 1.0))
        if S["sub"] is not None:
            blit(card, S["sub"], kx, sy + (1 - u) * 70, ax=ax, ay=1, op=u, clip=(0, sy - 90, self.W, sy + 10))
        w_in = e_io(lin(t, ti, ti + 0.32))
        ex = e_in(lin(t, to, te))
        if ex > 0:
            M = cv2.getRotationMatrix2D((self.W * (0.42 if self.fmt == "16:9" else 0.5), self.H * 0.5), 0, 1 + 0.5 * ex)
            card = cv2.warpAffine(card, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if w_in < 1:
            yy, xx = self._gy, self._gx
            edge = (xx + (self.H - yy) * 0.35) / (self.W + self.H * 0.35)
            m_card = np.clip((w_in * 1.25 - 0.1 - edge) / 0.002, 0, 1)[..., None]
            m_bar = np.clip((w_in * 1.25 - edge) / 0.002, 0, 1)[..., None]
            dst[...] = dst * (1 - m_bar) + rgbf(self.C["accent"]) * m_bar
            dst[...] = dst * (1 - m_card) + card * m_card
        else:
            dst[...] = dst * ex + card * (1 - ex)

    # ------------------------------------------------------------------ end card
    @lru_cache(None)
    def end_sprites(self):
        E = self.ecard
        lines = E["lines"] + [""] * (3 - len(E["lines"]))
        big = 1.0 if self.fmt == "16:9" else 0.92
        d = dict(l1=text_sprite(lines[0], self.F["display"], int(138 * big), self.C["text"], pad=8) if lines[0] else None,
                 l2=text_sprite(lines[1], self.F["display"], int(176 * big), self.C["accent"], pad=8) if lines[1] else None,
                 l3=text_sprite(lines[2], self.F["label"], 40, self.C["soft"], tracking=5, pad=6) if lines[2] else None,
                 cursor=add_shadow(icon_sprite("arrow-pointer", 74, "#FFFFFF", stroke=5, stroke_fill="#0A0A0A"), 6, 2, 4, 0.5))
        if E.get("button"):
            d["btn"] = self.button(E["button"], self.C["accent"], self.C["ink"], False)
            d["btn2"] = self.button(E.get("button_done", E["button"] + "D" if E["button"].upper() == "SUBSCRIBE" else "DONE"),
                                    self.C["bg_center"], self.C["accent"], True)
        return d

    def button(self, txt, bg, fg, check):
        w, h = 520, 118
        b = rounded_rect(w, h, 59, bg, border=3 if check else 0, border_fill=hexc(self.C["accent"], 0.6) if check else None)
        ic = icon_sprite("check" if check else "bell", 46, fg)
        tx = text_sprite(txt, self.F["caption"], 50, fg, tracking=2, pad=4)
        while tx.shape[1] + ic.shape[1] + 22 > w - 50:
            tx = cv2.resize(tx, (int(tx.shape[1] * 0.92), int(tx.shape[0] * 0.92)))
        x0 = (w - (ic.shape[1] + 22 + tx.shape[1])) / 2
        blit(b, ic, x0, h / 2, ax=0, ay=0.5)
        blit(b, tx, x0 + ic.shape[1] + 22, h / 2 + 3, ax=0, ay=0.5)
        return add_shadow(b, 18, 0, 10, 0.5)

    def draw_endcard(self, dst, t):
        T0 = self.endcard
        u = e_io(lin(t, T0, T0 + 0.55))
        card = self.bg_plate().copy()
        self.grid(card, t); self.floaters(card, t)
        if self.fmt == "16:9":
            PX, PY, PW, PH = 115, 262, 790, 444
            tx, l1y, l2y, l3y, bx, by, ax = 990, 262, 405, 668, 1252, 812, 0
        else:
            PX, PY, PW, PH = 240, 190, 600, 750
            tx, l1y, l2y, l3y, bx, by, ax = self.W / 2, 1000, 1140, 1400, self.W / 2, 1555, 0.5
        # crop window: full frame -> PIP aspect around the face, while shrinking into place
        vx_, vy_, vw_, vh_ = self.view
        tcw = min(vw_, vh_ * PW / PH); tch = tcw * PH / PW
        tcx, tcy = vx_ + vw_ / 2, clamp(vy_ + vh_ / 2 if self.vlayout == "stack" else self.H * 0.42, tch / 2, self.H - tch / 2)
        cw = self.W + (tcw - self.W) * u; ch = self.H + (tch - self.H) * u
        ccx = self.W / 2 + (tcx - self.W / 2) * u; ccy = self.H / 2 + (tcy - self.H / 2) * u
        x_a, y_a = int(max(0, ccx - cw / 2)), int(max(0, ccy - ch / 2))
        crop = dst[y_a:y_a + int(ch), x_a:x_a + int(cw)]
        w = self.W + (PW - self.W) * u; h = self.H + (PH - self.H) * u
        x0 = PX * u; y0 = PY * u
        fr = cv2.resize(crop, (max(2, int(round(w))), max(2, int(round(h)))), interpolation=cv2.INTER_AREA)
        r = 34 * u
        m = rounded_rect(fr.shape[1], fr.shape[0], max(1, int(r)), "#FFFFFF")[..., 3:4] if u > 0.01 else np.ones(fr.shape[:2] + (1,), np.float32)
        spr = np.concatenate([fr * m, m], 2).astype(np.float32)
        if u > 0.05:
            bd = rounded_rect(fr.shape[1] + 12, fr.shape[0] + 12, max(1, int(r + 6)), hexc(self.C["accent"], 0.9))
            sh = add_shadow(bd, 24, 0, 16, 0.6 * u)
            mm = (sh.shape[1] - bd.shape[1]) // 2
            blit(card, sh, x0 - 6 - mm, y0 - 6 - mm, ax=0, ay=0, op=u)
        blit(card, spr, x0, y0, ax=0, ay=0)
        S = self.end_sprites()
        a1 = e_out(lin(t, T0 + 0.30, T0 + 0.70)); a2 = e_out(lin(t, T0 + 0.40, T0 + 0.82)); a3 = e_out(lin(t, T0 + 0.7, T0 + 1.1))
        if S["l1"] is not None:
            blit(card, S["l1"], tx, l1y + (1 - a1) * 200, ax=ax, ay=0, clip=(0, l1y, self.W, l1y + 180))
        if S["l2"] is not None:
            blit(card, S["l2"], tx - (4 if ax == 0 else 0), l2y + (1 - a2) * 260, ax=ax, ay=0, clip=(0, l2y + 10, self.W, l2y + 245))
        if S["l3"] is not None:
            blit(card, S["l3"], tx + (6 if ax == 0 else 0), l3y + (1 - a3) * 30, ax=ax, ay=0, op=a3)
        if "btn" in S:
            bt = T0 + 1.0; bu = lin(t, bt, bt + 0.4); CLICK = T0 + 2.45
            if bu > 0:
                sc = e_back(bu, 2.0)
                if CLICK <= t < CLICK + 0.25: sc *= 1 - 0.08 * math.sin(math.pi * (t - CLICK) / 0.25)
                blit(card, S["btn"] if t < CLICK + 0.06 else S["btn2"], bx, by, scale=sc, op=clamp(bu * 3))
                if t >= CLICK:
                    ru = lin(t, CLICK, CLICK + 0.5)
                    if ru < 1: blit(card, ring_sprite(420, self.C["accent2"]), bx - 120, by, scale=0.1 + 0.5 * e_out(ru), op=(1 - ru) * 0.9)
                ct0 = T0 + 1.6
                if ct0 <= t < CLICK + 1.2:
                    cu = e_io(lin(t, ct0, CLICK - 0.05))
                    cx = self.W - 20 + (bx - 150 - self.W + 20) * cu; cy = self.H + 40 + (by + 5 - self.H - 40) * cu
                    lv = e_in(lin(t, CLICK + 0.55, CLICK + 1.2)); cx += 500 * lv; cy += 300 * lv
                    blit(card, S["cursor"], cx, cy, scale=0.88 if CLICK <= t < CLICK + 0.15 else 1.0, ax=0.25, ay=0.12, op=1 - lv)
        dst[...] = card
        fo = e_in(lin(t, self.total - 0.45, self.total))
        if fo > 0: dst *= (1 - fo)

    # ------------------------------------------------------------------ frame
    def base_footage(self, t):
        if self.endcard is not None and t >= self.endcard:
            s = self.segs[-1]
            st = s["vin"] + (t - s["ostart"])
            last = (self.pool.N) / FPS
            loop = self.ecard.get("loop")
            f, A = self.footage(s, min(t, s["oend"]), src_t=min(st, last))
            if loop:
                w0, w1 = float(loop[0]), float(loop[1]); per = max(0.2, w1 - w0)
                lt = t - (self.endcard + 0.45)
                k = lt % (2 * per)
                stl = w0 + (k if k < per else 2 * per - k)
                mix = smooth(lin(t, self.endcard + 0.45, self.endcard + 0.70))
                if mix > 0:
                    f2, A2 = self.footage(s, min(t, s["oend"]), src_t=stl if lt >= 0 else w0)
                    f = f * (1 - mix) + f2 * mix; A = A * (1 - mix) + A2 * mix
            return f, A
        s = self.seg_at(t)
        f, A = self.footage(s, t)
        i = self.order.index(s["name"])
        # morph cut (optical flow) around a boundary
        for j in (i, i + 1):
            if 0 < j < len(self.segs) and self.segs[j]["enter"] == "morph":
                T_ = self.segs[j]["ostart"]; hf = 5 / FPS
                if abs(t - T_) < hf:
                    w = smooth((t - T_ + hf) / (2 * hf))
                    fa, Aa = self.footage(self.segs[j - 1], t)
                    fb, Ab = self.footage(self.segs[j], t)
                    f = self.morph(fa, fb, w); A = Aa * (1 - w) + Ab * w
        for W_ in self.whips:
            if abs(t - W_) < 0.2:
                d = (t - W_) / 0.2
                if d < 0: k = e_in(d + 1); shift = -k * self.W * 0.55
                else: k = 1 - e_out(d); shift = k * self.W * 0.55
                blur = int(2 + k * 160 * self.W / 1920)
                M = np.float32([[1, 0, shift], [0, 1, 0]])
                f = cv2.warpAffine(f, M, (self.W, self.H), borderMode=cv2.BORDER_REFLECT)
                A = cv2.warpAffine(A[..., 0], M, (self.W, self.H), borderMode=cv2.BORDER_REFLECT)[..., None]
                if blur > 3: f = cv2.filter2D(f, -1, np.ones((1, blur), np.float32) / blur, borderType=cv2.BORDER_REFLECT)
        return f, A

    def whip_shift(self, t):
        for W_ in self.whips:
            if abs(t - W_) < 0.2:
                d = (t - W_) / 0.2
                return (-e_in(d + 1) * self.W * 0.55) if d < 0 else None
        return 0.0

    @lru_cache(None)
    def header_sprites(self):
        tc = self.ed.get("title_card") or {}
        top = tc.get("kicker") or ""
        main = tc.get("title") or " ".join(self.ed.get("hook_text", {}).get("lines", []))
        k = text_sprite(top, self.F["label"], 34, self.C["accent"], tracking=5, pad=6) if top else None
        size = 110
        while size > 40 and font(self.F["display"], size).getlength(main) > self.W - 140: size -= 4
        m = text_sprite(main, self.F["display"], size, self.C["text"], pad=8, shadow=(10, 0, 5, 0.5)) if main else None
        return k, m

    def draw_header(self, img, t):
        if self.vlayout != "stack": return
        start = self.title["te"] if self.title else (self.hook["to"] + 0.4 if self.hook else 0.5)
        if t < start or (self.endcard is not None and t >= self.endcard): return
        u = e_out(lin(t, start, start + 0.5))
        k, m = self.header_sprites()
        vy = self.view[1]
        if k is not None: blit(img, k, self.W / 2, vy - 165 + (1 - u) * 20, op=u)
        if m is not None: blit(img, m, self.W / 2, vy - 85 + (1 - u) * 30, op=u)

    def overlays(self, img, t):
        self.draw_header(img, t)
        self.draw_stickers(img, t)
        self.draw_cards(img, t)
        self.draw_burst(img, t)

    def render(self, t, finish=True):
        f, A = self.base_footage(t)
        img = f
        if self.hook and t < self.hook["to"] + 0.45:
            L = self.hook_layer(t)
            sh = self.whip_shift(t)
            if L is not None and sh is not None:
                if sh: L = cv2.warpAffine(L, np.float32([[1, 0, sh], [0, 1, 0]]), (self.W, self.H))
                m = 1 - A
                img = img * (1 - L[..., 3:4] * m) + L[..., :3] * m
        if self.endcard is None or t < self.endcard:
            self.overlays(img, t)
            self.captions(img, t)
            self.draw_title(img, t)
            if finish: img *= self.vig
        else:
            if finish: img *= self.vig
            self.draw_endcard(img, t)
        if finish: img = self.grain(img)
        return np.clip(img, 0, 1)

    def grain(self, img):
        amt = self.brand["finish"]["grain"]
        if amt <= 0: return img
        n = self.rng.standard_normal((self.H // 2, self.W // 2)).astype(np.float32)
        n = cv2.resize(n, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        lum = img.mean(2, keepdims=True)
        return np.clip(img + n[..., None] * amt * (0.6 + 2 * lum * (1 - lum)), 0, 1)


# ====================================================================== drivers
LONG_ENC = ["-preset", "veryfast", "-crf", "18"]


def encode_cmd(W, H, out, crf=15, enc=None):
    enc = enc or ["-preset", "medium", "-crf", str(crf)]
    return ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
            "-c:v", "libx264"] + enc + ["-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", out]


def render_part(job_dir, t0, t1, out, simple=False, finish=True, static=False):
    job = Job(job_dir)
    eng = Engine(job, simple_grade=simple)
    enc = None
    if static:      # long-video mode: static framing, no grain, the body's encoder settings
        for s in eng.segs:
            zm = sum(s["_z"]) / 2; s["_z"] = [zm, zm]
        eng.brand["finish"]["grain"] = 0
        enc = LONG_ENC
    p = subprocess.Popen(encode_cmd(eng.W, eng.H, out, enc=enc)[:-1] + ["-video_track_timescale", "15360", out], stdin=subprocess.PIPE)
    n0, n1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    tt = time.time()
    for n in range(n0, n1):
        im = eng.render(n / FPS, finish=finish)
        p.stdin.write((im * 255 + 0.5).astype(np.uint8).tobytes())
        if (n - n0) % 60 == 0:
            print(f"  part {t0:.1f}-{t1:.1f}: frame {n - n0}/{n1 - n0}  {time.time() - tt:.0f}s", flush=True)
    p.stdin.close(); p.wait()


def render_full(job_dir, name="render", workers=None, t0=0.0, t1=None, simple=False, extra=()):
    job = Job(job_dir)
    edl = job.load("edl.json")
    t1 = edl["total"] if t1 is None else t1
    workers = workers or os.cpu_count() or 2
    nfr = int(round((t1 - t0) * FPS))
    cuts = [t0 + round(nfr * k / workers) / FPS for k in range(workers + 1)]
    procs, parts = [], []
    for k in range(workers):
        out = job.p("out", f"_{name}_part{k}.mp4"); parts.append(out)
        procs.append(subprocess.Popen([sys.executable, "-m", "vek", "render-part", job_dir, str(cuts[k]), str(cuts[k + 1]), out]
                                      + (["--simple"] if simple else []) + list(extra), cwd=os.path.dirname(os.path.dirname(__file__))))
    for p in procs:
        p.wait()
        if p.returncode: raise SystemExit("render part failed")
    lst = job.p("out", f"_{name}_parts.txt")
    with open(lst, "w") as f:
        f.write("".join(f"file '{x}'\n" for x in parts))
    video = job.p("out", f"{name}_video.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video], check=True)
    for x in parts: os.remove(x)
    os.remove(lst)
    return video


def mux(job, video, out):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-i", job.p("master.wav"), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", "-shortest", out],
                   check=True)
    return out


def test_frames(job_dir, times):
    job = Job(job_dir)
    eng = Engine(job)
    os.makedirs(job.p("out", "frames"), exist_ok=True)
    paths = []
    for t in times:
        im = eng.render(float(t))
        fn = job.p("out", "frames", f"f_{float(t):07.2f}.jpg")
        cv2.imwrite(fn, cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(fn)
    from .review import sheet
    s = sheet(paths, job.p("out", "frames_sheet.jpg"), cols=3, w=640)
    print("[frames]", s)
    return s
