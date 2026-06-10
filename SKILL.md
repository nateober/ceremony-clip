---
name: ceremony-clip
description: Use when making a short cinematic highlight clip from a long ceremony or event recording (graduation, recital, awards, wedding) — locating the moment a name is called, adding a title card and locally-generated music, all from the command line with no video editor.
---

# Ceremony Highlight Clip

Turn a multi-hour ceremony recording into a ~35-second shareable cinematic clip: the
name call, the walk, a title card, an original orchestral cue ducked under the
announcer. Every step is a command; every step is machine-verifiable. Helper scripts
live in `scripts/` next to this file.

## Pipeline

1. **Find the moment** (transcript search) → 2. **Cut** → 3. **Title card** →
4. **Generate music** → 5. **Mix with ducking** → 6. **Verify**

## 1. Find the moment

**If a transcript (`.srt`) already exists next to the recording, search it — don't re-transcribe.**

```bash
grep -n -i 'montgomery' recording.srt   # surname first; first names get mangled
```

Otherwise transcribe. **Critical:** Whisper hallucinates filler text ("Thank you.")
over music/applause, and its own `no_speech_threshold`/`logprob` settings CANNOT
filter this (music ≠ silence to Whisper; the filler is high-confidence). Pre-filter
with VAD — use the bundled script (Silero VAD → mlx whisper-turbo Python API; the
`mlx_whisper` CLI degenerates into repetition loops, never use it):

```bash
scripts/transcribe_vad.py recording.mp4 --output-dir work/   # → .txt + .srt
```

A ~2h ceremony transcribes in ~10–15 min on Apple Silicon. (On Nate's fleet,
`~/bin/transcribe-clean` is the same pipeline with more output formats.)

**Search tolerantly.** ASR mangles first names (Delilah → "Dalila"). Search the
surname; if no hit, fuzzy-match (`rapidfuzz`, threshold ~70) over the transcript.
Names are read in sequence — the right hit sits between other names.

## 2. Cut the window — and verify the subject is actually in it

**The name call is NOT when the subject is on stage.** At a big ceremony the
walkers lag the name reader — measured **~20 seconds** at a 601-graduate
commencement. The reader is working through a queue: whoever is shaking hands
when you hear the name was called ~3 names earlier. A clip framed only around
the name shows the *wrong kid* walking (this exact mistake shipped once — the
clip starred the subject's friend).

Window: start `T-5`, duration **~42s** — name lands ~5s in, the subject typically
enters ~T+12..T+20, diploma/photographer pose ~T+20, exit ~T+25..T+30. Then
**verify before polishing**: extract frames at `T+10`, `T+18`, `T+25`, look at
them (Read tool), and confirm the subject — glasses, stole color, cords. If two
graduates match the description (regalia is a uniform; they will), show the user
one frame and ask. One human confirmation beats any amount of name-math.

```bash
for off in 10 18 25; do ffmpeg -y -ss $((T+off)) -i recording.mp4 -frames:v 1 /tmp/who_$off.jpg; done
# Read the jpgs; confirm the subject is the one crossing
```

Re-encode the cut;
`-c copy` glitches on non-keyframe cuts. Use `-t <dur>`, NOT `-to`: with `-ss` before
`-i` (fast input-seek), `-to` is interpreted on the original timeline and miscuts.
This re-encode also normalizes any source codec (AV1/Opus etc.) to H.264/AAC, which
is what makes the later `-c:a copy` safe:

```bash
ffmpeg -y -ss 1:59:25 -i recording.mp4 -t 42 \
  -c:v libx264 -crf 20 -preset fast -c:a aac -movflags +faststart clip.mp4
```

With the longer window, let the music bow out (step 5) *before* the subject's
walk — their crossing plays over the real applause, which is the moment anyway.

## 3. Title card

**Do NOT use drawtext** — homebrew ffmpeg is often built without it. Render a
transparent PNG instead, **sized to the video** (probe first; a mismatched canvas
misaligns the overlay):

```bash
ffprobe -v error -select_streams v -show_entries stream=width,height -of csv=p=0 clip.mp4
uv run --with pillow python3 scripts/make_title.py \
  --name "Brinley Montgomery" --subtitle "Rosemount High School · Class of 2026" \
  --width 1920 --height 1080 --out /tmp/title.png
```

Overlay with fades. **Trap:** a bare image input is a single frame at t=0 — alpha
fades evaluate once and the title is invisible. You MUST `-loop 1 -t <dur>`:

```bash
ffmpeg -y -i clip.mp4 -loop 1 -t 37 -i /tmp/title.png -filter_complex "
[1:v]format=rgba,fps=30,fade=t=in:st=0.3:d=0.8:alpha=1,fade=t=out:st=4.4:d=1.0:alpha=1[ttl];
[0:v][ttl]overlay=0:0:shortest=1,fade=t=in:st=0:d=1.2,fade=t=out:st=35.5:d=1.5[v]
" -map "[v]" -map 0:a -c:v libx264 -crf 20 -c:a copy clip-titled.mp4
```

## 4. Generate music (MusicGen, local)

`facebook/musicgen-small` runs fine on Apple Silicon MPS — **if you manage memory**.
On a 16 GB box, evict any resident Ollama model first or Metal OOMs *silently*
(hallucinated/empty output, wedged runner):

```bash
ollama ps                      # anything resident?
ollama stop <model>            # evict; reload after with an empty /api/generate call
uv run --with torch,transformers,scipy python3 scripts/gen_music.py \
  --prompt "cinematic orchestral, warm strings and french horns, proud and uplifting, gentle build to a triumphant swell, no vocals" \
  --out /tmp/music.wav
```

Hard limits: fp32 on MPS (fp16 → NaNs); ≤30 s per generation (1500 tokens @ 50 Hz —
the model's training cap). 30 s is enough: the music should bow out to natural
applause anyway. Do not substitute a numpy sine-wave synth — it sounds like a
doorbell, not a ceremony.

## 5. Mix — duck under the announcer

A deterministic volume envelope keyed to the name offset beats sidechaincompress
(reproducible, no tuning-by-ear). Name at `N` seconds into the clip (here 5.8):
fade in → duck to 18% just before `N` → swell after → fade out at 25 s so real
applause carries the ending. MusicGen output is mono — pan to stereo:

```bash
ffmpeg -y -i clip-titled.mp4 -i /tmp/music.wav -filter_complex "
[1:a]volume='if(lt(t,2),0.55*t/2,if(lt(t,4.5),0.55,if(lt(t,5.2),0.55-0.37*(t-4.5)/0.7,if(lt(t,9.5),0.18,if(lt(t,11),0.18+0.27*(t-9.5)/1.5,0.45)))))':eval=frame,
afade=t=out:st=25:d=4.9,aresample=48000,pan=stereo|c0=c0|c1=c0[m];
[0:a][m]amix=inputs=2:duration=first:normalize=0[a]
" -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k final.mp4
```

Shift the duck window (the 4.5/5.2/9.5/11 breakpoints) to bracket your `N` — and
check the title card's fade-out (step 3, `st=4.4`) also clears before `N`, so the
name is heard over video, not over text.

## 6. Verify (you cannot listen — check mechanically)

```bash
# Name audible through the mix? Re-transcribe the name window of the FINAL file:
ffmpeg -y -ss <N-1.3> -i final.mp4 -t 4 -vn /tmp/check.wav
~/bin/transcribe-clean /tmp/check.wav --output-dir /tmp/check-out && cat /tmp/check-out/check.txt
# → must contain the name (ASR spelling variants OK)

# Title card rendered? Extract a frame mid-title and view it:
ffmpeg -y -ss 2.5 -i final.mp4 -frames:v 1 /tmp/frame.jpg   # then Read the jpg
```

If you evicted an Ollama model, restore it:
`curl -s localhost:11434/api/generate -d '{"model":"<model>","prompt":"","keep_alive":-1}'`

## Common mistakes

| Mistake | Reality |
|---|---|
| Filtering Whisper filler with no_speech/logprob thresholds | Can't work — music isn't silence and filler is high-confidence. VAD pre-filter. |
| `drawtext` for titles | Often missing from homebrew ffmpeg builds. PNG + overlay. |
| PNG overlay without `-loop 1` | Single frame at t=0 + alpha fade-in = invisible title. |
| Skipping MusicGen for a numpy synth | "Generated" ≠ "musical." MusicGen-small works on MPS with memory evicted. |
| MusicGen fp16 on MPS / >30 s | NaNs / degraded tail. fp32, ≤1500 tokens. |
| "I'll listen to check the mix" | You can't. Re-transcribe the mixed name window. |
| sidechaincompress + tune by ear | Same problem. Deterministic envelope keyed to the name offset. |
| Exact-matching the first name in the transcript | ASR mangles proper nouns. Surname first, then fuzzy. |
| Assuming the walker during the name call is the subject | Walkers lag the reader 10–25s (queue). Verify with frames at T+10/18/25; ask the user if two kids match. |
| Trusting face-quality scores to pick "the best shot of X" | Vision scores find *a* good face, not *the right* face — two graduates in identical regalia fooled it. Identity first (anchor frame confirmed by a human), quality second. |

## Beyond the clip (optional extras, all local)

- **Stills + contact sheet:** extract the subject's window at 6–10 fps, score
  sharpness (Laplacian variance) or face quality (Apple Vision
  `VNDetectFaceCaptureQualityRequest` + yaw via a small Swift CLI), dedupe per
  0.5s, `magick montage` with timestamp labels. The diploma-photographer pose
  (~T+20) is reliably the best face-to-camera moment.
- **Enhance for print:** Real-ESRGAN 2x/4x then GFPGAN (weight ~0.5 preserves
  identity) rescues 1080p video faces to small-print quality. Both run on MPS —
  packaged as a sibling skill: [image-kit](https://github.com/nateober/image-kit).
- **Slow-mo finale:** 59.94fps sources give clean 2x slow motion at 30fps
  (`setpts=2*PTS`) — made for cap tosses. Find the real toss by frame
  inspection; transcript cues for it can be ~10s early.
