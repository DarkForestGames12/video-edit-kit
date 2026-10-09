# video-edit-kit

Turns a raw talking-head recording into a finished, edited video: tight cuts, the strongest line
moved to the top as a hook, colour grade with background treatment, animated captions with
forced-aligned word timing, motion graphics (text behind the speaker, title card, explainer cards,
icon pops, end card), an original synthesised music bed locked to the edit, sound design, and a
mastered mix at -14 LUFS. Built to be driven by Claude through the companion skill.

## Quick start
```bash
bash setup.sh                                 # deps + models (~650 MB) + fonts/icons, cached in ~/.vek
python -m vek prep    jobs/myvid raw.mov      # probe, A/V offset fix, audio, contact sheets, head track
python -m vek transcribe jobs/myvid           # phrase map -> jobs/myvid/transcript.md
#   ... write jobs/myvid/edit.json (see docs/EDIT_SPEC.md) ...
python -m vek voice   jobs/myvid              # cut + voice processing + caption word timings
python -m vek audio   jobs/myvid              # music + sfx + mix/master
python -m vek frames  jobs/myvid 1.5 6.5 16   # spot-check frames
python -m vek render  jobs/myvid final        # full render -> jobs/myvid/out/final.mp4
python -m vek review  jobs/myvid jobs/myvid/out/final.mp4
python -m vek deliver jobs/myvid jobs/myvid/out/final.mp4 My_Video
```

## Pipeline
| stage | module | what it does |
|---|---|---|
| ingest | `prep.py` | ffprobe, corrects audio/picture offset (Mac recordings are often ~0.8 s off), extracts aligned audio, contact sheets, person-matte head track |
| speech map | `transcribe.py` | splits speech at pauses, transcribes each phrase alone (exposes "um", false starts, retakes) and in context; flags them in `transcript.md` |
| draft cut | `autocut.py` | optional first-pass cut list from the flags |
| voice | `voice.py` | assembles the cut, EQ/denoise/de-ess/compress, per-phrase levelling, verifies the cut by re-transcribing, forced-aligns caption words (`align.py`) |
| music | `music.py` | electric piano / bass / drums / pluck synth, tempo grid anchored so a downbeat hits the title slam, filtered build-up, cadence into the end card |
| sound design | `sfx.py` | whooshes, risers, impacts, pops, heartbeats, chimes, clicks generated for every graphic automatically |
| mix | `mix.py` | ducking (music ~12 LU under speech), master to -14 LUFS / -1 dBTP |
| picture | `grade.py`, `reader.py`, `engine.py` | streamed + denoised + matted frames, subject/background grade, per-shot framing, punch-ins, whip pans, optical-flow morph cuts, all graphics, vignette + grain |
| long videos | `longform.py` | engine for the opening and end card, ffmpeg + LUT + .ass captions + overlay clips for the body |
| QA | `review.py` | contact sheets of the render, speech intelligibility check of the final mix, loudness |
| delivery | `deliver.py` | master + a share copy under 29 MB |

## Brand kits
`brands/<channel>.json` overrides `brands/TEMPLATE.json` (colours, fonts, caption style, title
motif, floating icons, music mood, grade tweaks, end-card text). One file per channel; upload it
to `brands/` in this repo so future sessions find it.

## Licences of downloaded assets
Parakeet TDT 0.6B v2 and Conformer-CTC (NVIDIA, CC-BY-4.0) via sherpa-onnx; Robust Video Matting
(GPL-3.0 code / released weights); Google Fonts (OFL); Font Awesome Free (icons CC BY 4.0, fonts OFL).
Nothing is redistributed in this repo; `setup.sh` downloads them.
