# Why Biology Matters in Daily Life

Source 38.1 s (1080x720) -> 27.5 s, 16:9 full render. Brand: channel kit `example-science`. Edited 2026-10-09.
Tags: webcam 720p, short (<1 min), heavy fillers and thinking pauses, hook stitched across a pause, channel kit, pulse + rings cards, icon burst

## The video
A short to-camera talk on why studying biology helps in daily life (your heart, your health). One
speaker on a laptop webcam, 720p, plain room. The delivery is hesitant: "um", "mm", "uh", muttered
"okay… right", a stutter ("it… it is"), a false start ("It is… it can help") and a 4.5 s thinking
pause in the middle of the best sentence. On the contact sheets the speaker leans toward the lens
at 3.0-4.9 s during the greeting. Mac recording: the audio sat 0.79 s off the picture.

## Key decisions
- **Hook:** "Biology… can help you so much in your daily life." It is the only line that makes a
  promise to the viewer, but it was spoken in two pieces with "can mm", a mumble and a 4.5 s
  pause between them. Stitching "Biology" (16.6 s) to "can help you so much…" (22.1 s) made one
  clean sentence at 0.0 s. The hook text BIO / LOGY shows behind the speaker, so caption word 0 is
  skipped (`skip [0]`). The line is not repeated later.
- **Structure:** hook -> whip -> greeting -> title card -> "it is very important to study in our
  daily life. Biology. It can help you know about your heart, it can teach you about your health,
  and etc." -> "Goodbye" -> end card. The speaker's order kept; only the hook moved.
- **Title card** "WHY YOU SHOULD STUDY | BIOLOGY | IN YOUR DAILY LIFE" lands on "channel. And I
  want to tell you about how", which covers the lean-in.
- **Cuts:** every filler and the stutter, the false start, the muttered asides and all pauses
  over ~0.3 s. "to" and "study" were said with gaps, so they became their own short segments
  joined with morph cuts.
- **Graphics:** a hand sticker on "How are you" and on "Goodbye", book-open on "study", dna on "Biology".
  A pulse card YOUR HEART with heartbeats and an ECG on "heart". A rings card YOUR HEALTH with
  the chime on "health". An 8-icon burst (brain, lungs, microscope…) on "and etc.", which turns a
  weak list ending into a visual. One punch-in on "important". Keywords: BIOLOGY, STUDY and
  IMPORTANT in accent, HEART in alert, HEALTH in accent2.
- **End:** the goodbye segment starts early so the smile and wave play before the word. The end
  card loops that wave (36.17-36.90 s), with a subscribe button since it's a channel video.

## Problems and fixes
This video was edited before the kit existed; each problem became a kit feature.
- Audio 0.79 s late against the picture -> `prep` corrects the offset from the container start time.
- The recogniser's word times were 0.1-0.3 s late, so captions and pops trailed the voice ->
  CTC forced alignment of the final voice (`align.py`).
- Plain crossfades between two takes ghosted (double face) -> optical-flow `morph` cuts.
- Caption lines overlapped and broke awkwardly -> lines end before the next starts; rewritten line grouper.
- The 9:16 crop of a landscape webcam put captions over the mouth -> the stacked 9:16 layout
  (square window over a blurred background, captions below).
- The recogniser misheard "important" as "Wotent" and "my channel" as "main channels" -> read
  the `ctx` column of the phrase map, which hears words in context.

## Lessons for next time
- The hook may be assembled from pieces said seconds apart; prove the join with `asr --join`.
- Hesitant speakers produce single-word segments; morph-join them so the picture doesn't jump.
- A weak ending like "and etc." is a perfect place for an icon burst.
- Cover a lean-in or look-away with the title card, not a cut.
