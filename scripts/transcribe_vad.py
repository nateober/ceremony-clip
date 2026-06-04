#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-whisper", "silero-vad", "torch", "numpy"]
# ///
"""VAD-gated Whisper transcription (Apple Silicon).

Silero VAD finds the speech regions first; only those get transcribed
(mlx whisper-turbo), so music/applause/silence never reaches Whisper and
can't trigger its filler-text hallucinations ("Thank you."). Timestamps
stay in original-media time. Accepts anything ffmpeg can read (mp4, mov,
m4a, wav, ...).

Usage: transcribe_vad.py INPUT [--output-dir DIR]
Outputs: <stem>.txt and <stem>.srt
"""

import argparse
import sys
from pathlib import Path

SR = 16000


def fmt_ts(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--model", default="mlx-community/whisper-turbo")
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1
    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / args.input.stem

    import numpy as np
    import torch
    import mlx_whisper
    from mlx_whisper.audio import load_audio
    from silero_vad import load_silero_vad, get_speech_timestamps

    print(f"[load] {args.input.name}", flush=True)
    audio = np.asarray(load_audio(str(args.input)), dtype=np.float32)  # mlx → numpy
    total_s = len(audio) / SR

    print(f"[vad] scanning {total_s/60:.1f} min", flush=True)
    raw = get_speech_timestamps(
        torch.from_numpy(audio), load_silero_vad(), sampling_rate=SR,
        threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=500,
    )

    # Pad 0.5s, merge gaps < 1.5s
    pad, merge_gap = SR // 2, int(1.5 * SR)
    regions: list[list[int]] = []
    for ts in raw:
        s, e = max(0, ts["start"] - pad), min(len(audio), ts["end"] + pad)
        if regions and s <= regions[-1][1] + merge_gap:
            regions[-1][1] = max(regions[-1][1], e)
        else:
            regions.append([s, e])
    speech_s = sum(e - s for s, e in regions) / SR
    print(f"[vad] {len(regions)} regions, {speech_s/60:.1f} min speech", flush=True)

    segments = []
    for i, (s, e) in enumerate(regions, 1):
        res = mlx_whisper.transcribe(
            audio[s:e], path_or_hf_repo=args.model, language=args.language,
            condition_on_previous_text=False, verbose=None,
        )
        off = s / SR
        for seg in res["segments"]:
            if seg["text"].strip():
                segments.append((seg["start"] + off, min(seg["end"] + off, e / SR), seg["text"].strip()))
        if i % 10 == 0 or i == len(regions):
            print(f"[whisper] {i}/{len(regions)}", flush=True)

    out_base.with_suffix(".txt").write_text(
        "".join(t + "\n" for _, _, t in segments), encoding="utf-8")
    out_base.with_suffix(".srt").write_text(
        "\n".join(f"{i}\n{fmt_ts(a)} --> {fmt_ts(b)}\n{t}\n"
                  for i, (a, b, t) in enumerate(segments, 1)), encoding="utf-8")
    print(f"[done] {len(segments)} segments → {out_base}.txt / .srt", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
