"""video-edit-kit command line.  python -m vek <command> <job_dir> [args]

  prep <job> <video>          probe, fix A/V offset, audio, contact sheets, head track
  transcribe <job>            phrase map -> transcript.md (flags fillers / false starts / pauses)
  asr <job> s:e [s:e ...]     transcribe source ranges (add --join to hear a planned join)
  autocut <job>               draft cut list -> cuts.draft.json (review before using)
  voice <job>                 edit.json -> edl.json, voice.wav (processed + levelled), words.json
  audio <job>                 music + sfx + mix/master (needs edit.json, edl.json, words.json)
  grade-test <job> t [t ...]  before/after grade on source times -> out/grade_test.jpg
  frames <job> t [t ...]      render single output frames -> out/frames_sheet.jpg
  render <job> [name]         full render (parallel) + mux -> out/<name>.mp4   (mode "long" -> hybrid)
  review <job> <video>        review sheets, speech check, loudness, spot-check times
  deliver <job> <video> <name>  master + <=29 MB share copy in /mnt/user-data/outputs
  vertical <job> <new_job>    copy a finished 16:9 job as a 9:16 job (reframed shots, same cut/audio)
  example <job> <slug> "Title"  save the finished edit as examples/<slug>/ (edit.json, cutlog.md, NOTES.md)
  example-pack <slug>         check NOTES.md, rebuild examples/README.md, zip for upload to the repo
  pack-kit                    zip the whole kit (examples included) for upload to the repo
  icons <word>                search Font Awesome icon names
  brands                      list brand kits
"""
import sys, os, json


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__); return
    cmd = a[0]
    if cmd == "prep":
        from .prep import main as m; m(a[1], a[2])
    elif cmd == "transcribe":
        from .transcribe import main as m; m(a[1])
    elif cmd == "asr":
        from .transcribe import check_ranges
        from .common import Job
        join = "--join" in a
        check_ranges(Job(a[1]), [x for x in a[2:] if x != "--join"], join)
    elif cmd == "autocut":
        from .autocut import main as m; m(a[1])
    elif cmd == "voice":
        from .voice import main as m; m(a[1])
    elif cmd == "audio":
        from . import music, sfx, mix
        music.main(a[1]); sfx.main(a[1]); mix.main(a[1])
    elif cmd == "grade-test":
        grade_test(a[1], [float(x) for x in a[2:]])
    elif cmd == "frames":
        from .engine import test_frames; test_frames(a[1], [float(x) for x in a[2:]])
    elif cmd == "render-part":
        from .engine import render_part
        render_part(a[1], float(a[2]), float(a[3]), a[4], "--simple" in a, "--nofinish" not in a, "--static" in a)
    elif cmd == "render":
        from .common import Job
        from .engine import render_full, mux
        job = Job(a[1]); name = a[2] if len(a) > 2 else "final"
        ed = job.load("edit.json")
        if ed.get("mode") == "long":
            from .longform import render_long
            video = render_long(a[1], name)
        else:
            video = render_full(a[1], name)
        out = mux(job, video, job.p("out", f"{name}.mp4"))
        os.remove(video)
        print("[render]", out)
    elif cmd == "review":
        from .review import main as m; m(a[1], a[2])
    elif cmd == "deliver":
        from .deliver import main as m; m(a[1], a[2], a[3])
    elif cmd == "vertical":
        make_vertical(a[1], a[2])
    elif cmd == "example":
        from .example import make; make(a[1], a[2], a[3] if len(a) > 3 else None)
    elif cmd == "example-pack":
        from .example import pack; pack(a[1])
    elif cmd == "pack-kit":
        from .example import pack_kit; pack_kit()
    elif cmd == "icons":
        from .gfx import icon_codes
        q = a[1].lower()
        print(" ".join(sorted(k for k in icon_codes() if q in k)))
    elif cmd == "brands":
        from .common import BRANDS
        for f in sorted(os.listdir(BRANDS)):
            if f.endswith(".json") and f != "TEMPLATE.json":
                print(f[:-5], "-", json.load(open(os.path.join(BRANDS, f))).get("name", ""))
    else:
        print(__doc__)


def make_vertical(src, dst):
    """Same cut, captions, music and graphics; shots re-framed for 9:16 (the vertical crop is
    already tight, so zooms are compressed)."""
    import shutil
    from .common import Job
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst, f))
    j = Job(dst)
    ed = j.load("edit.json")
    ed["format"] = "9:16"
    for s in ed["segments"]:
        sh = s.get("shot", {})
        z = sh.get("z", [1.0, 1.03])
        z = [z, z] if isinstance(z, (int, float)) else z
        s["shot"] = {"z": [round(1 + (z[0] - 1) * 0.4, 3), round(1 + (z[1] - 1) * 0.4, 3)]}
    for p in ed.get("punches", []):
        p["z"] = round(1 + (p["z"] - 1) * 0.5, 3)
    for st in ed.get("stickers", []):
        st.pop("dx", None); st.pop("dy", None)
    j.save("edit.json", ed)
    from .voice import build_edl
    build_edl(j)
    print("[vertical] ready:", dst, "(audio, words and cut reused; run frames/render)")


def grade_test(job_dir, times):
    import numpy as np, cv2
    from .common import Job, load_brand, FPS
    from .reader import FramePool
    from .grade import Grader
    job = Job(job_dir)
    ed = job.load("edit.json", {}) if job.has("edit.json") else {}
    brand = load_brand(ed.get("brand", "example-science"))
    g = Grader(brand.get("grade"))
    pool = FramePool(job.load("probe.json"))
    head = np.array(job.load("head.json")["rows"])
    rows = []
    for t in times:
        i = int(t * FPS); fr, al = pool.get(i)
        cx, top, hw, eye = head[min(i, len(head) - 1)]
        out = g.full(fr, al, cx, eye - 0.1 * hw)
        w = 640; h = int(fr.shape[0] * w / fr.shape[1])
        rows.append(np.hstack([cv2.resize(fr, (w, h)), cv2.resize(out, (w, h))]))
    fn = job.p("out", "grade_test.jpg")
    cv2.imwrite(fn, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("[grade-test]", fn, "(left: source, right: graded)")


if __name__ == "__main__":
    main()
