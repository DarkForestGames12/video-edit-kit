---
name: pro-video-edit
description: Edit a raw talking-head video into a polished, engaging finished video (hook-first cut, filler removal, grade, animated captions, motion graphics, original music, sound design, mastered audio) using the video-edit-kit toolkit. Any channel, length or format.
---

# Pro video edit

You are the editor the user paid a lot for. Deliver a finished, publish-ready video that feels
designed for *this* video, never a template pass. Work through every stage below in order; each
one exists because skipping it caused a visible or audible flaw before.

The toolkit lives at **https://github.com/DarkForestGames12/video-edit-kit**. Reference spec:
`docs/EDIT_SPEC.md`. Worked example (the first video edited this way, 38 s webcam take → 27 s):
`examples/biology-daily-life/edit.json`, which shows every feature with a note on each cut.

## 0. Setup (every session — the workspace is wiped between sessions)
1. Get the toolkit anonymously (the repo is public):
   `git clone https://github.com/DarkForestGames12/video-edit-kit ~/video-edit-kit`.
   **Never use `add_repo`, never push, never use any GitHub account or token for this skill.** The
   GitHub account connected to this Claude account belongs to someone else in the family.
   If the clone comes back empty, the toolkit hasn't been uploaded yet: ask the user to upload the
   contents of video-edit-kit.zip to that repo (github.com → Add file → Upload files), or to attach
   the zip here, and unzip it to `~/video-edit-kit`.
2. `cd ~/video-edit-kit && bash setup.sh` (~2 min first run; downloads models from GitHub releases —
   HuggingFace is blocked, don't try it).
3. Find the video: attachments are in `/mnt/user-data/uploads/`. If it's on the user's linked
   computer, stage it with the device tools (per-file cap ~400 MB — ask the user to split longer
   recordings or attach parts). Copy uploads before modifying; never edit originals.
4. Tell the user in one line what you're about to do, then create a task list (stages 1–7).

## 1. Intake (one AskUserQuestion call, before heavy work)
- **Channel** → `python -m vek brands` lists saved kits. Options: each saved kit + "New channel".
- **Format** → YouTube 16:9 / Shorts-Reels-TikTok 9:16 / Both.
- Optional free text: anything to keep, cut or emphasise.
**New channel** → ask (second call): channel name; niche + vibe; colours (theirs or "pick for me" —
then propose a palette with ink/background, accent, accent2, alert); font feel (bold condensed =
anton + montserrat black | clean = bebas-neue + inter | friendly = poppins); captions UPPERCASE or
Sentence case; end-card lines + button text; music mood (uplifting/chill/energetic/serious/warm);
title motif (helix = science, orbit = tech/space, bars = music/gaming/business, none).
Write `brands/<slug>.json` (only the overrides of `brands/TEMPLATE.json`). The workspace is wiped
after the session, so **send the user that JSON file** (SendUserFile) and ask them to upload it into
the `brands/` folder of their repo on github.com (Add file → Upload files) — then it's there next
time. Don't push it yourself. If the user is away: use the closest saved kit, say so at the top.

## 2. Ingest and understand
```
python -m vek prep JOB VIDEO        # JOB = ~/jobs/<short-name>
python -m vek transcribe JOB
```
- `prep` fixes the audio/picture offset (Mac recordings are often ~0.8 s off) — every time in the
  kit is the video clock afterwards. Never extract audio yourself with plain ffmpeg.
- **Look at every contact sheet** (`JOB/out/contact_*.jpg`). Note visual problems with times:
  leaning into the lens, looking away/thinking, hands over face, leaving frame, focus/exposure jumps.
- **Read `JOB/transcript.md` fully.** It's the phrase map: each phrase transcribed alone (`asr`,
  exposes "um", "mm", muttered "yeah") and in context (`ctx`, better wording), plus flags
  (FILLER, false start, retake, long pause). The recogniser alone silently drops fillers — the
  phrase map is how you find them.
- Check doubtful spots: `python -m vek asr JOB 27.3:28.9 28.2:29.7` (transcribe ranges).

## 3. Edit decisions — the craft
Optionally start from `python -m vek autocut JOB` (draft in `cuts.draft.json`), but decide yourself.
1. **Hook first.** Find the single strongest self-contained line (a bold claim, payoff, surprise, or
   "X can help you so much…") and move it to 0.0 s. Greeting ("hey guys, welcome back") comes
   after the hook. Don't repeat that line later.
2. **Structure:** HOOK → whip → greeting/intro → TITLE CARD → body in the speaker's order →
   payoff → goodbye → END CARD.
3. **Remove:** fillers, stutters, false starts and retakes (keep the *clean* take — usually the
   last), muttered asides, thinking pauses, doubled words across a pause ("can… mm… can help" →
   "Biology can help"), dead air. Keep ~0.1–0.2 s breaths between sentences; never clip a word tail.
4. **Prove every join** before building on it: `python -m vek asr JOB a:b c:d --join` must read
   as the intended sentence. If a join reads wrong, move the cut point to the energy dip.
5. **Visual flaws over good audio** (lean-in, looking off) → cover with the title card (perfect
   for "today I want to tell you about…"), a card, or cut. Picture may lead sound: start a goodbye
   segment early so the wave/smile plays before the word.
6. **Framing:** change framing at every jump cut: wide z≈1.0 ↔ medium 1.12–1.18 ↔ close
   1.26–1.32, slow push (+3 %) within each segment. Max ~1.35 on webcam/720p (softer beyond).
   `enter: "morph"` when two takes of similar framing join mid-thought; `enter: "whip"` once,
   hook → intro. Hook segments: subject right (`px` .70) so the hook text can sit behind on the left.
7. **Punch-in** (`punches`) on the one key word of a key sentence; 1–3 per minute.
8. **Graphics — something new every 3–6 s in short videos, never two big things at once:**
   - `hook_text`: the topic in 1–2 words, giant, behind the speaker during the hook.
   - `title_card`: kicker (why/how framing), TITLE (topic, one word ideal), sub (the benefit).
   - `stickers`: concrete nouns as they're spoken (study → book-open, science → dna, hi/bye → hand).
   - `cards`: the 1–3 core concepts; put `beats`/`pop` on the spoken keyword
     (`pulse`+ECG for heart/health/time; `rings` for protection/achievement; `plain` otherwise).
   - `burst`: lists, "and so on", "etc.".
   - `end_card`: `loop` a wave or smile from the source.
   - caption `keywords`: topic word = accent, warning/emotion = alert, positive = accent2.
     `skip` caption words the hook text already shows.
9. **Captions stay faithful** to what was said (fillers may be dropped; don't put words in their mouth).
10. **Music:** mood from the brand; the kit anchors a downbeat to `title_card.in`.
Write `JOB/edit.json` (spec: `docs/EDIT_SPEC.md`), with a `note` on every segment saying why.

## 4. Build the sound
```
python -m vek voice JOB     # cut, voice chain, per-phrase levelling, re-transcribe, word timings
python -m vek audio JOB     # music + sound design + mix/master
```
- "edited voice reads:" must be fluent with no fillers or doubled words — if not, fix edit.json.
- `words.json` now gives word indices; switch keyword-synced times to refs like `"w45"` and rerun
  `voice` only if the cut changed, `audio` if graphics times changed.
- Mix check line: music 11–14 LU under the voice during speech; master −14 LUFS, true peak ≤ −1.

## 5. Picture checks before the long render
```
python -m vek grade-test JOB 5 15 25     # source times; left original, right graded
python -m vek frames JOB 0.5 1.5 <whip> <title.in+0.8> <each card> <burst> <end+0.3> <end+2>
```
View both images. Grade: face natural and bright, background a step darker and cooler, no blotchy
skin (if blotchy, lower `subj_smooth`/`subj_clarity` in the brand's `grade`). Frames: hook text
tucked partly behind the head but readable; nothing clipped at frame edges; cards clear of the face;
captions never over the mouth.

## 6. Render and review
- `mode`: `"full"` up to ~3–4 min of output (≈1.5 s per frame per core: a 1-min video ≈ 25 min);
  `"long"` beyond (premium opening + end card through the engine, fast ffmpeg body, same
  colours/captions/graphics; ≈0.4× real time, so an hour-long video renders in ~2.5 h). Tell the
  user the expected wait up front and keep working/polling while it runs.
- Run in the background and poll (tool calls time out at 10 min):
  `nohup python -m vek render JOB final > JOB/render.log 2>&1 &` then `tail JOB/render.log`.
  Never `pkill -f`/`pgrep -f` a pattern that also appears in your own command line.
- `python -m vek review JOB JOB/out/final.mp4` → **look at every review sheet** (4 frames/s):
  caption overlaps or flicker, ghosting at morphs, clipped graphics, pops not on their word, end
  card. The speech check must show no missing words (spelling variants like want/wanted are fine).
  Fix edit.json and re-render until clean. For 9:16 as well: `python -m vek vertical JOB JOB_v`
  (same cut and audio; reframed; landscape sources get the stacked layout automatically), then
  frames → render → review that job too.

## 7. Deliver
- `python -m vek deliver JOB JOB/out/final.mp4 <Title_Name>` → `<name>.mp4` (≤29 MB share copy)
  and `<name>_master.mp4` in `/mnt/user-data/outputs`. Send the share copy with SendUserFile.
  If the computer is linked, also write the master into a folder they choose.
- Final message: what changed (hook moved, what was cut, with times), the look, graphics, sound,
  any caption wording choices, and one next step. No step-by-step recap.

## Gotchas
- Word timings come from forced alignment (CTC); recogniser timestamps alone drift 0.1–0.3 s.
- Skin: keep the grade light; heavy smoothing + sharpening + zoom = painted look.
- Plain crossfades between takes ghost; use `morph`.
- Hook-text side layout needs the subject to one side; 9:16 from landscape uses a top band.
- Long videos: picking the hook and cuts from an hour of transcript is the real work — read the
  phrase map in chunks, use `autocut` for fillers, spend judgment on hook, structure and the
  1–3 cards per section; add `title_card` and an `end_card` as usual.
- Brand kits only persist if the user uploads them to `brands/` in their repo (you never push).
