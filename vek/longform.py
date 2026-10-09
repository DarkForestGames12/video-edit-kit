"""Long videos (roughly > 4 min of output): same edit spec, hybrid render so an hour doesn't take a day.

  A  opening (hook + title card) -> full engine, frame by frame, parallel workers
  B  body -> one ffmpeg encode per piece (segment, split at punches): denoise, LUT grade exported from
     the engine's per-pixel grade (colours match), static shot crop, graphics as alpha clips rendered by
     the engine only where they appear, captions from a per-piece .ass, vignette. Encoded once.
  C  end card -> full engine
Then a stream-copy concat. No film grain (it would triple the file size of an hour-long video).
Whip/morph transitions inside the body become cuts; background matte treatment is skipped there.
"""
import os, subprocess, math, time
import numpy as np, cv2
from PIL import ImageFont
from .common import Job, FPS
from .engine import Engine, LONG_ENC, render_full
from .gfx import font_path

TS = ["-video_track_timescale", "15360"]


def ass_color(hexs):
    h = hexs.lstrip("#")
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def ass_header(eng):
    fp = font_path(eng.F["caption"])
    fam = " ".join(ImageFont.truetype(fp, 20).getname())
    size = eng.brand["captions"]["size_169" if eng.fmt == "16:9" else "size_916"]
    f = ImageFont.truetype(fp, 100); asc, desc = f.getmetrics()
    mv = int(eng.H - eng.L["cap_y"] - size * 0.45)
    size = int(round(size * (asc + desc) / 100))        # libass sizes by ascent+descent, Pillow by em
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {eng.W}
PlayResY: {eng.H}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{fam},{size},{ass_color(eng.C['text'])},{ass_color(eng.C['text'])},&H000F1106,&H8C000000,0,0,0,0,100,100,1,0,1,7,4,2,60,60,{mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""", os.path.dirname(fp)


def caption_events(eng, t_from, t_to):
    """(start, end, ass_text) for every caption state overlapping [t_from, t_to). Words appear as
    spoken (later words are laid out but transparent, so the line never shifts), active word in accent."""
    up = eng.brand["captions"]["uppercase"]
    ev = []
    for c in eng.chunks:
        if c["en"] <= t_from or c["st"] >= t_to:
            continue
        ws = c["words"]
        for k, w in enumerate(ws):
            a = c["st"] if k == 0 else w["t"] - 0.06
            b = ws[k + 1]["t"] - 0.06 if k + 1 < len(ws) else c["en"]
            a, b = max(a, t_from), min(b, t_to)
            if b - a < 0.01:
                continue
            parts = []
            for j, x in enumerate(ws):
                txt = x["w"].upper() if up else x["w"]
                key = eng.kw.get(x["i"])
                if j > k:
                    parts.append("{\\alpha&HFF&}" + txt)
                else:
                    colr = eng.col(key) if key else (eng.C["accent"] if j == k else eng.C["text"])
                    parts.append("{\\alpha&H00&\\c" + ass_color(colr) + "}" + txt)
            ev.append((a, b, " ".join(parts)))
    return ev


def ts(t):
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def overlay_windows(eng, t0, t1):
    wins = []
    for s in eng.stickers: wins.append((s["ti"], s["to"] + 0.02))
    for c in eng.cards: wins.append((c["ti"] - 0.02, c["to"] + 0.3))
    if eng.burst: wins.append((eng.burst["ti"] - 0.03, eng.burst["to"]))
    if eng.vlayout == "stack": wins.append((t0, t1))        # persistent header
    wins = sorted((max(a, t0), min(b, t1)) for a, b in wins if b > t0 and a < t1)
    merged = []
    for a, b in wins:
        if merged and a <= merged[-1][1] + 0.2: merged[-1][1] = max(merged[-1][1], b)
        else: merged.append([a, b])
    return merged


def render_overlay_clip(eng, a, b, out):
    n0, n1 = int(math.floor(a * FPS)), int(math.ceil(b * FPS))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{eng.W}x{eng.H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "qtrle", out], stdin=subprocess.PIPE)
    for n in range(n0, n1):
        lay = np.zeros((eng.H, eng.W, 4), np.float32)
        eng.overlays(lay, n / FPS)
        al = np.clip(lay[..., 3:4], 1e-6, 1)
        rgb = np.where(lay[..., 3:4] > 1e-4, lay[..., :3] / al, 0)
        p.stdin.write((np.concatenate([rgb, lay[..., 3:4]], 2).clip(0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait()
    return n0 / FPS, n1 / FPS


def body_pieces(eng, t0, t1):
    pieces = []
    for s in eng.segs:
        a, b = max(s["ostart"], t0), min(s["oend"], t1)
        if b - a < 1 / FPS: continue
        cuts = [a] + [p["t"] for p in eng.punches if a < p["t"] < b] + [b]
        for k in range(len(cuts) - 1):
            pieces.append((s, cuts[k], cuts[k + 1]))
    return pieces


def render_long(job_dir, name="final"):
    job = Job(job_dir)
    eng = Engine(job, simple_grade=True)
    for s in eng.segs:
        zm = sum(s["_z"]) / 2; s["_z"] = [zm, zm]
    W, H = eng.W, eng.H
    A_end = round((eng.title["te"] if eng.title else (eng.whips[0] if eng.whips else 0.0)) * FPS) / FPS
    C0 = round((eng.endcard if eng.endcard is not None else eng.total) * FPS) / FPS
    d = job.p("out", "_long"); os.makedirs(d, exist_ok=True)
    cube = os.path.join(d, "grade.cube"); eng.grader.export_cube(cube)
    tt = time.time()
    for tag, (a, b) in (("A", (0.0, A_end)), ("C", (C0, eng.total))):
        if b - a < 1 / FPS: continue
        v = render_full(job_dir, f"_long{tag}", t0=a, t1=b, simple=True, extra=("--static",))
        os.replace(v, os.path.join(d, f"{tag}.mp4"))
        print(f"[long] part {tag} {a:.2f}-{b:.2f} done ({time.time()-tt:.0f}s)", flush=True)
    # graphics clips for the body (only where something is on screen)
    clips = []
    for k, (a, b) in enumerate(overlay_windows(eng, A_end, C0)):
        fn = os.path.join(d, f"ov{k:03d}.mov")
        s0, s1 = render_overlay_clip(eng, a, b, fn)
        clips.append((s0, s1, fn))
    print(f"[long] {len(clips)} graphics clips ({time.time()-tt:.0f}s)", flush=True)
    vig = os.path.join(d, "vignette.png")
    v = 1 - eng.vig[..., 0]
    cv2.imwrite(vig, np.dstack([np.zeros_like(v)] * 3 + [v * 255]).astype(np.uint8))
    hdr, fontsdir = ass_header(eng)
    pr = eng.probe
    pieces = body_pieces(eng, A_end, C0)
    bfiles = []
    stack = eng.vlayout == "stack"
    for k, (s, a, b) in enumerate(pieces):
        pt = a + 1e-3
        if any(s["ostart"] <= p["t"] <= pt for p in eng.punches):
            p_ = max((p for p in eng.punches if s["ostart"] <= p["t"] <= pt), key=lambda p: p["t"])
            pt = p_["t"] + p_["dur"] + 0.01
        M, Z, sc = eng.cam(s, pt)
        vx, vy, vw, vh = eng.view
        cw, ch = vw / sc, vh / sc
        x0, y0 = -M[0, 2] / sc, -M[1, 2] / sc
        src_t = s["vin"] + (a - s["ostart"])
        nfr = int(round(b * FPS)) - int(round(a * FPS))
        dur = nfr / FPS
        ass = os.path.join(d, f"cap{k:04d}.ass")
        with open(ass, "w") as f:
            f.write(hdr + "\n".join(f"Dialogue: 0,{ts(x - a)},{ts(y - a)},Cap,,0,0,0,,{txt}"
                                    for x, y, txt in caption_events(eng, a, b)) + "\n")
        base = (f"fps={FPS},scale={pr['work_w']}:{pr['work_h']}:flags=bicubic,crop={cw:.2f}:{ch:.2f}:{x0:.2f}:{y0:.2f},"
                f"hqdn3d=2.5:2:5:4,lut3d='{cube}',scale={vw}:{vh}:flags=bicubic,unsharp=3:3:0.45")
        inputs = ["-ss", f"{src_t:.4f}", "-i", pr["source"]]
        fc = []
        if stack:
            fc.append(f"[0:v]{base},split[w][w2]")
            fc.append(f"[w2]scale=-2:{H}:flags=fast_bilinear,crop={W}:{H},boxblur=40:2,eq=brightness=-0.22:saturation=0.85[bg]")
            fc.append(f"[bg][w]overlay={vx}:{vy}[v0]")
        else:
            fc.append(f"[0:v]{base}[v0]")
        last = "v0"; n = 1
        for (s0, s1, fn) in clips:
            if s1 <= a or s0 >= b: continue
            seek = max(0.0, a - s0); start = max(0.0, s0 - a)
            inputs += ["-ss", f"{seek:.4f}", "-itsoffset", f"{start:.4f}", "-i", fn]
            fc.append(f"[{last}][{n}:v]overlay=eof_action=pass[o{n}]"); last = f"o{n}"; n += 1
        inputs += ["-loop", "1", "-framerate", str(FPS), "-i", vig]
        fc.append(f"[{last}]ass='{ass}':fontsdir='{fontsdir}'[c]")
        fc.append(f"[c][{n}:v]overlay=shortest=1,format=yuv420p[vout]")
        out = os.path.join(d, f"B{k:05d}.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y"] + inputs + ["-filter_complex", ";".join(fc), "-map", "[vout]",
                        "-frames:v", str(nfr), "-an", "-c:v", "libx264"] + LONG_ENC +
                       ["-r", str(FPS), "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"] + TS + [out],
                       check=True)
        bfiles.append(out)
        if k % 10 == 0 or k == len(pieces) - 1:
            print(f"[long] body piece {k+1}/{len(pieces)} ({time.time()-tt:.0f}s)", flush=True)
    order = ([os.path.join(d, "A.mp4")] if A_end > 0 else []) + bfiles + \
            ([os.path.join(d, "C.mp4")] if eng.total - C0 > 0.03 else [])
    lst = os.path.join(d, "list.txt")
    with open(lst, "w") as f: f.write("".join(f"file '{x}'\n" for x in order))
    video = job.p("out", f"{name}_video.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video], check=True)
    print(f"[long] video done in {time.time()-tt:.0f}s -> {video}")
    return video
