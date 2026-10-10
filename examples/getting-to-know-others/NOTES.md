# Getting to Know Others: 2 Ways

Source 77.4 s (1620x1080) -> 68.0 s, 16:9 full render. Brand: one-off (brand.json in this folder). Edited 2026-10-10.
Tags: Mac webcam 3:2, careful reader with long pauses, pros/cons comparison, cold open + repeat, no channel, one-off brand, hand-set caption lines, caption fix

## The video
A short speaking-exercise style talk with no channel behind it. Topic: two ways to get to know
people before starting a new school or university, an online chat group or an informal meeting.
For each it gives pros and cons, then picks the informal meeting. One speaker reads at a careful
pace on a Mac webcam (1620x1080, a bright window behind). Pauses run up to 1.8 s, plus a few stumbles:
- "this… at once"
- a doubled "be"
- "pe– per–… yep" before "people"
- "okay… uh" before "choices"
"Advantages" is said fast and the recogniser hears it as "dishes" or "digits". The voice is
quieter than in the first video. The audio offset was small (0.07 s).

## Key decisions
- **Hook:** "Some students may feel nervous when speaking to others." It is the only emotional,
  relatable line. But it is the "con" of the informal meeting, and cutting it from the body breaks
  the comparison. So it is a cold open at 0 s and plays again in context, framed differently, with a
  punch-in on "nervous". (This video added the cold-open exception to the skill's hook rule.)
- **Hook text** FEEL / NERVOUS? turns the statement into a question to the viewer. `tuck` 0.36,
  because the close-up head hid too much of it at the default.
- **Title card** "GETTING TO KNOW OTHERS | 2 WAYS | BEFORE A NEW SCHOOL OR UNIVERSITY" lands on
  "two ways of getting to know others" and covers it.
- **Body in the speaker's order.** All pauses over ~0.3 s cut. Morph cuts where a doubled word or
  stumble sat inside a sentence of the same framing (B8b, B9b). Kept on purpose: the pause before
  the verdict "…but I would prefer — an informal meeting", for drama.
- **Cards:**
  - ONLINE CHAT and INFORMAL MEETING introduce the two options, each popping on its spoken keyword.
  - MY PICK (rings and check) lands on the verdict, so the ending pays off visually.
- **Stickers** on concrete nouns: sun on "morning", balance scale on "advantages", house on
  "home", school, question mark on "misunderstood", handshake on "meet", grimace on "nervous" and
  a wave on "bye". Food icons burst on "enjoy snacks". Punches on "convenient" and "nervous".
- **No channel:** a one-off navy / yellow / teal brand and warm music. The end card asks WHICH
  WOULD YOU PICK? and has no subscribe button.

## Problems and fixes
- **"advantages" misheard** ("dishes"/"digits"). The cut was moved to start cleanly on the word
  (B3b at 21.33), and `captions.fix {"55": "advantages"}` was added (new kit feature).
- **Choppy automatic caption lines** for a pausing speaker (one-word lines, "online | chat group").
  Fixed with 52 hand-set `captions.groups` (new kit feature).
- **Card label too wide:** "INFORMAL MEETING" overflowed the card, so card labels now auto-fit.
- **Hook text hidden** behind the close-up head; `hook_text.tuck` was added.
- **Music too loud for a quieter voice, and true peak -0.5.** Music and effects are now balanced
  against the measured voice (`music_rel` -12.5, `sfx_rel` -5.5), with a tighter final limiter.
- **Forced 105 bpm:** TEMPLATE.json forced it for every brand. Removed, so the mood sets the tempo
  (warm = 96 bpm).
- **Silent render failure:** one render part died without a traceback, out of memory at 1620x1080.
  The reader's frame cache was cut to 24 frames. The failed range was re-rendered with
  `render-part` in two halves and concatenated.
- **Effects masking speech:** burst pops and the card chime covered "snacks" and "informal
  meeting" in the speech check. Fixed with softer pops and chime, and effects now duck 4 dB under
  speech. Re-mixed, then an audio-only remux.
- **Deliver timeout:** `deliver` ran past the 10-minute tool limit, and restarting it overwrote
  the master. It now runs in the background with nohup and is never restarted mid-way.
- **Known leftover:** captions touch for a few frames during the pop-in overshoot
  ("CANCOMMUNICATE"). Barely visible; worth fixing in the engine some day.

## Lessons for next time
- In comparison videos the best hook is often one side of the argument: cold-open it and let it
  replay in context with a new framing and a punch-in.
- Turn a hook statement into a question in the hook text ("FEEL NERVOUS?").
- Slow or pausing speaker: check caption lines on the first `frames` sheet and hand-set groups early.
- Fast single words get misheard; fix the caption with `fix` instead of re-cutting around them.
- A source wider than 1080p is heavy on memory, so watch the render log for parts that stop.
- Run the speech check after adding pops on keywords; an effect on the word can mask it.
- With no channel, make the end card a question that ties back to the video.
