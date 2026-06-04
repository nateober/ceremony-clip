# ceremony-clip

Turn a multi-hour ceremony recording into a ~35-second cinematic highlight clip of
**your kid's moment** — their name being called, the walk across the stage — with a
title card and an original orchestral score, generated entirely on your Mac. No
video editor, no stock-music licensing, nothing uploaded anywhere.

Built as an [Agent Skill](https://agentskills.io) for
[Claude Code](https://claude.com/claude-code): you hand Claude the recording and a
name; it finds the moment, cuts it, scores it, mixes it, and *verifies its own work*
(it re-transcribes the final mix to prove the name is still audible through the music).

## What you get

- The right 35-second window found automatically — no scrubbing through 2+ hours
- Fade-in title card (name + school + honors)
- A unique orchestral cue composed locally by [MusicGen](https://huggingface.co/facebook/musicgen-small),
  ducked under the announcer so the name rings clear, bowing out to the real applause
- A small shareable MP4 your family will actually watch

## Requirements

- Apple Silicon Mac (M1 or later; 16 GB RAM is fine)
- [Claude Code](https://claude.com/claude-code)
- `ffmpeg` (`brew install ffmpeg`) and [`uv`](https://docs.astral.sh/uv/) (`brew install uv`)

Python dependencies (Whisper, Silero VAD, MusicGen) install themselves on first run
via `uv`. First run downloads ~3 GB of models; after that it's all offline.

## Use it

Install as a skill:

```bash
mkdir -p ~/.claude/skills && git clone https://github.com/nateober/ceremony-clip ~/.claude/skills/ceremony-clip
```

Then, in Claude Code:

> Use the ceremony-clip skill to make a highlight clip of **Jordan Alvarez** from
> `~/Movies/graduation-2026.mp4`. Subtitle "Lincoln High School · Class of 2026".

Claude reads `SKILL.md` and drives the pipeline end to end. A 2-hour recording takes
roughly 20–30 minutes total on an M1 (transcription ~10–15 min, music ~3 min, the
rest seconds).

Prefer to drive it by hand? `SKILL.md` is also a human-readable runbook — every step
is a plain command, and `scripts/` contains the three helpers
(`transcribe_vad.py`, `make_title.py`, `gen_music.py`), each with `--help`.

## How it works (the short version)

1. **Find the moment** — VAD-gated Whisper transcription (plain Whisper hallucinates
   "Thank you." over ceremony music; speech-only gating fixes it), then a
   surname-first search of the transcript.
2. **Cut** a 37-second window starting 5 seconds before the name.
3. **Title card** rendered as a transparent PNG, composited with alpha fades.
4. **Score** composed locally by MusicGen-small on the GPU.
5. **Mix** with a deterministic volume envelope: music ducks to 18% under the name,
   swells for the walk, yields to natural applause.
6. **Verify** — the name window of the *final mix* is re-transcribed; if Whisper
   can hear the name through the music, so can grandma.

## Notes

- Works for any ceremony with an announcer: graduations, recitals, award nights,
  citizenship ceremonies, weddings.
- MusicGen's weights are CC-BY-NC — fine for personal/family clips, not for
  commercial use.
- Whisper may spell the name creatively in transcripts ("Delilah" → "Dalila");
  the skill searches tolerantly, and it doesn't affect the video.

## License

MIT (the skill and scripts). Model weights retain their own licenses.
