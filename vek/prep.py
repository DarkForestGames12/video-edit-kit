"""Stage 1 - ingest: probe the source, fix A/V offset, extract audio, contact sheets, head track."""
import os, json, subprocess, shutil
import numpy as np
from .common import Job, ffprobe, FPS, RVM


def probe(job, src):
    info = ffprobe(src)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = int(v.get("tags", {}).get("rotate", rot))
    if abs(rot) % 180 == 90:
        w, h = h, w
    vstart = float(v.get("start_time", 0) or 0)
    astart = float(a.get("start_time", 0) or 0) if a else 0.0
    dur = float(info["format"]["duration"])
    pr = dict(source=os.path.abspath(src), width=w, height=h, duration=dur,
              fps=v.get("avg_frame_rate"), audio_offset=round(astart - vstart, 6),
              has_audio=a is not None, audio_channels=int(a.get("channels", 1)) if a else 0)
    # working resolution: cap the long side at 1920 (4K sources are downscaled for speed)
    scale = min(1.0, 1920 / max(w, h))
    pr["work_w"] = int(round(w * scale / 2) * 2)
    pr["work_h"] = int(round(h * scale / 2) * 2)
    job.save("probe.json", pr)
    return pr


def extract_audio(job):
    import soundfile as sf
    pr = job.load("probe.json")
    tmp = job.p("_a.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", pr["source"], "-vn", "-ac", "1", "-ar", "48000",
                    "-c:a", "pcm_f32le", tmp], check=True)
    a, sr = sf.read(tmp, dtype="float32")
    os.remove(tmp)
    off = pr["audio_offset"]  # audio starts this long after the first video frame
    n = int(round(abs(off) * sr))
    if off > 0:
        a = np.concatenate([np.zeros(n, np.float32), a])
    elif off < 0:
        a = a[n:]
    total = int(pr["duration"] * sr)
    if len(a) < total:
        a = np.concatenate([a, np.zeros(total - len(a), np.float32)])
    sf.write(job.p("audio48.wav"), a, sr, subtype="FLOAT")
    import scipy.signal as ss
    sf.write(job.p("audio16.wav"), ss.resample_poly(a, 1, 3).astype(np.float32), 16000, subtype="FLOAT")
    return off


def contact_sheets(job, every=1.0, per_sheet=40, cols=8, width=270):
    import cv2
    pr = job.load("probe.json")
    out = []
    cmd = ["ffmpeg", "-v", "error", "-i", pr["source"], "-vf", f"fps={1/every},scale={width}:-2", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    hh = int(round(pr["height"] * width / pr["width"] / 2) * 2)
    tiles = []
    k = 0
    while True:
        b = p.stdout.read(width * hh * 3)
        if len(b) < width * hh * 3:
            break
        im = np.frombuffer(b, np.uint8).reshape(hh, width, 3).copy()
        cv2.putText(im, f"{k*every:.0f}s", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        tiles.append(im)
        k += 1
    for s in range(0, len(tiles), per_sheet):
        blk = tiles[s:s + per_sheet]
        while len(blk) % cols:
            blk.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(blk[i:i + cols]) for i in range(0, len(blk), cols)])
        fn = job.p("out", f"contact_{s//per_sheet:02d}.jpg")
        cv2.imwrite(fn, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append(fn)
    return out


def head_track(job, sample_fps=10, work_w=512):
    """Person matte at low res -> head position per source frame (30 fps index), in work-res pixels."""
    import onnxruntime as ort, cv2
    pr = job.load("probe.json")
    W, H = pr["work_w"], pr["work_h"]
    ww = work_w; wh = int(round(H * ww / W / 2) * 2)
    sess = ort.InferenceSession(RVM, providers=["CPUExecutionProvider"])
    rec = [np.zeros([1, 1, 1, 1], np.float32)] * 4
    dr = np.array([0.5], np.float32)
    cmd = ["ffmpeg", "-v", "error", "-i", pr["source"], "-vf", f"fps={sample_fps},scale={ww}:{wh}", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    rows = []
    while True:
        b = p.stdout.read(ww * wh * 3)
        if len(b) < ww * wh * 3:
            break
        f = np.frombuffer(b, np.uint8).reshape(wh, ww, 3)
        src = (f.astype(np.float32) / 255).transpose(2, 0, 1)[None]
        fgr, pha, *rec = sess.run(None, {"src": src, "r1i": rec[0], "r2i": rec[1], "r3i": rec[2], "r4i": rec[3],
                                          "downsample_ratio": dr})
        rows.append(head_from_matte(pha[0, 0] > 0.5))
    rows = np.array(rows, np.float64)
    sx = W / ww
    rows[:, :] *= sx
    # resample to 30 fps index
    n30 = int(pr["duration"] * FPS)
    ts = np.arange(len(rows)) / sample_fps
    t30 = np.arange(n30) / FPS
    out = np.stack([np.interp(t30, ts, rows[:, c]) for c in range(rows.shape[1])], 1)
    job.save("head.json", dict(fields=["cx", "top", "width", "eye_y"], work=[W, H], rows=np.round(out, 1).tolist()))
    return out


def head_from_matte(m):
    H, W = m.shape
    rowsum = m.sum(1)
    ys = np.where(rowsum > max(3, 0.02 * W))[0]
    if len(ys) == 0:
        return [W / 2, H * 0.15, W * 0.25, H * 0.35]
    y0 = int(ys[0])
    widths = []
    lim = min(H, y0 + int(0.6 * H))
    for y in range(y0, lim):
        xs = np.where(m[y])[0]
        widths.append((xs[-1] - xs[0]) if len(xs) else 0)
    widths = np.array(widths, np.float64)
    if len(widths) < 8:
        return [W / 2, y0, W * 0.25, y0 + 0.2 * H]
    ref = np.median(widths[len(widths) // 8: len(widths) // 3 + 2])
    over = np.where(widths > 1.6 * max(ref, 1))[0]
    end = int(over[over > len(widths) // 6][0]) if len(over[over > len(widths) // 6]) else len(widths)
    hw = float(np.percentile(widths[:max(end, 4)], 90))
    band = m[y0:y0 + max(end, 4)]
    cols = band.sum(0)
    cx = float((cols * np.arange(W)).sum() / max(cols.sum(), 1))
    return [cx, float(y0), hw, float(y0 + 0.72 * hw)]


def main(job_dir, src):
    job = Job(job_dir)
    pr = probe(job, src)
    print(f"[prep] {pr['width']}x{pr['height']} {pr['duration']:.1f}s  audio offset {pr['audio_offset']:+.3f}s"
          f"  work {pr['work_w']}x{pr['work_h']}")
    off = extract_audio(job)
    if abs(off) > 0.02:
        print(f"[prep] NOTE: audio track starts {off:+.3f}s from the picture; corrected (all times = video clock)")
    sheets = contact_sheets(job, every=1.0 if pr["duration"] < 120 else max(2.0, pr["duration"] / 400))
    print("[prep] contact sheets:", *sheets)
    head_track(job)
    print("[prep] head track ok")
