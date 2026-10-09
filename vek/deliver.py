"""Stage 7 - deliverables. A high-quality master plus a share copy that fits the chat's 30 MB limit
(two-pass, bitrate computed from the length). Long videos: deliver the master to a linked folder."""
import os, subprocess, json
from .common import Job, ffprobe

LIMIT_MB = 29.0


def encode_share(src, out, limit_mb=LIMIT_MB, max_v=12000, audio_k=160):
    dur = float(ffprobe(src)["format"]["duration"])
    total_kbit = limit_mb * 8 * 1024 * 0.96
    v = int(min(max_v, total_kbit / dur - audio_k))
    scale = []
    if v < 2500:
        scale = ["-vf", "scale='if(gt(iw,ih),-2,720)':'if(gt(iw,ih),720,-2)'"]
    if v < 600:
        print(f"[deliver] {dur/60:.1f} min will not fit {limit_mb} MB at watchable quality - deliver the master via a folder")
        return None
    plog = out + ".pass"
    base = ["ffmpeg", "-v", "error", "-y", "-i", src] + scale + ["-c:v", "libx264", "-preset", "slow", "-b:v", f"{v}k"]
    subprocess.run(base + ["-pass", "1", "-passlogfile", plog, "-an", "-f", "null", "/dev/null"], check=True)
    subprocess.run(base + ["-maxrate", f"{int(v*1.5)}k", "-bufsize", f"{int(v*2)}k", "-pass", "2", "-passlogfile", plog,
                           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", f"{audio_k}k", "-movflags", "+faststart", out], check=True)
    for f in os.listdir(os.path.dirname(out) or "."):
        if f.startswith(os.path.basename(plog)): os.remove(os.path.join(os.path.dirname(out) or ".", f))
    return out


def encode_master(src, out, crf=18, maxrate="16M"):
    dur = float(ffprobe(src)["format"]["duration"])
    if dur > 240:   # long render is already a final encode: just make it web-ready (fast start)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-c", "copy", "-movflags", "+faststart", out], check=True)
        return out
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                    "-maxrate", maxrate, "-bufsize", "32M", "-pix_fmt", "yuv420p", "-color_primaries", "bt709",
                    "-color_trc", "bt709", "-colorspace", "bt709", "-c:a", "copy", "-movflags", "+faststart", out], check=True)
    return out


def main(job_dir, video, name):
    job = Job(job_dir)
    os.makedirs("/mnt/user-data/outputs", exist_ok=True)
    outdir = "/mnt/user-data/outputs" if os.path.isdir("/mnt/user-data/outputs") else job.p("out")
    m = encode_master(video, os.path.join(outdir, f"{name}_master.mp4"))
    s = encode_share(m, os.path.join(outdir, f"{name}.mp4"))
    for p in (m, s):
        if p: print(f"[deliver] {p}  {os.path.getsize(p)/1e6:.1f} MB")
