# edit.json — the per-video edit spec

One JSON file per video, in the job folder. Claude writes it after reading the transcript and
contact sheets; every render stage reads it. Full worked example: `examples/biology-daily-life/edit.json`.

## Time values
- `segments[].in` / `.out` are **source video seconds** (the video clock; the audio offset is
  already corrected by `prep`).
- Every other time is **output seconds** or a reference string:
  `"w12"` (caption word 12 from words.json), `"w12+0.2"`, `"B1.start"`, `"B1.end-0.1"`,
  `"title.in"`, `"title.out"`, `"title.end"`, `"end"`.
  Word indices only exist after `vek voice`; use numbers or segment refs before that.

## Top level
| key | value |
|---|---|
| `brand` | brand kit name in `brands/` (e.g. `"example-science"`) |
| `format` | `"16:9"` (1920x1080) or `"9:16"` (1080x1920). Make two job folders for both. |
| `mode` | `"full"` (every frame designed; up to ~5 min) or `"long"` (hybrid render for long videos) |
| `segments` | the cut, **in playback order** (reordering = moving the hook to the top) |
| `punches` | `[{"at": time, "z": 1.30, "dur": 0.16}]` quick zoom-in for emphasis inside a segment |
| `hook_text` | `{"lines": ["BIO","LOGY"], "in": 0.06, "out": 2.95, "tuck": 0.22}` giant text *behind* the speaker; raise `tuck` (0.3-0.4) when the speaker sits close and hides too much of it |
| `title_card` | `{"in","out","end","kicker","title","sub"}` full-screen animated title; `in`->`out` hold, `out`->`end` zoom-through exit. Voice keeps playing under it. |
| `captions` | `{"skip":[0], "keywords":{"HEART":"alert","12":"accent"}, "fix":{"55":"advantages"}, "groups":[[1,2,3],[4,5]], "hide":[[a,b]]}` all optional; `fix` corrects a misheard word, `groups` overrides automatic line breaking |
| `stickers` | `[{"icon","color","in","out","side":"left|right","mode":"pop|wave|bob|spin","dx","dy"}]` icon pops beside the head |
| `cards` | explainer cards (see below) |
| `burst` | `{"in","out","icons":[["brain","#FF8FB1"],...]}` arc of icons around the head (lists, "and more") |
| `end_card` | `{"in": "B10.end", "length": 5.4, "loop": [s,e], "lines": [...], "button": "SUBSCRIBE"}`; `loop` = source seconds to ping-pong in the picture-in-picture (pick a wave/smile); `button: null` hides it |
| `music` | `{"mood": "uplifting|chill|energetic|serious|warm", "bpm": 105, "key": 0, "body": true, "file": null}` (`file` = use a provided track instead of synthesising) |
| `mix` | optional ducking overrides: `hook`, `title`, `speech`, `gap`, `endcard` (dB), `music_gain`, `sfx_gain` |
| `sfx_extra` | `[{"t": time, "kind": "pop|whoosh|impact|thump|ding|click|heartbeat|riser|chime", "gain": 0.4, "pan": 0}]` |

## Segment
```json
{"name": "B1", "in": 7.71, "out": 9.41, "enter": "cut|whip|morph",
 "shot": {"z": [1.12, 1.15], "px": 0.5, "py": 0.41}, "note": "why this cut exists"}
```
- `z` = zoom at segment start/end (slow push). 1.0 = widest crop of the source for the format.
- `px`/`py` = where the speaker's eyes land in the frame (0-1). Defaults: px .5, py .40 (16:9) / .44 (9:16).
  Segments that overlap a card are auto-framed to make room (subject right in 16:9, lower in 9:16).
- `enter`: `cut` (default), `whip` (fast blurred pan; use once, hook -> intro), `morph` (optical-flow
  morph cut; use when two takes of similar framing join mid-thought, instead of a visible jump).

## Cards
```json
{"kind": "pulse", "icon": "heart", "label": "YOUR HEART", "color": "alert",
 "in": 15.40, "out": 17.02, "beats": [16.46, 17.05], "ecg": true}
{"kind": "rings", "icon": "shield-heart", "label": "YOUR HEALTH", "color": "accent2",
 "in": 17.08, "out": 19.33, "pop": 18.98, "check": true}
{"kind": "plain", "icon": "lightbulb", "label": "KEY IDEA", "color": "accent", "in": .., "out": .., "pop": ..}
```
`pulse` = icon beats on `beats` (+ heartbeat sound, optional ECG line). `rings` = rings + check
badge + chime at `pop`. `plain` = soft pop at `pop`. Put `pop`/`beats` on the spoken keyword.
Card `out` should be <= the next segment's start if the next segment is framed differently.

## Colours
Anywhere a colour is accepted: a brand role (`accent`, `accent2`, `alert`, `text`, `soft`, `ink`)
or a hex string. Icons: any Font Awesome Free *solid* name — search with `python -m vek icons heart`.
