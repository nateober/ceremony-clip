#!/usr/bin/env python3
"""Generate a short music cue locally with MusicGen-small.

Run: uv run --with torch,transformers,scipy python3 gen_music.py --prompt "..." --out music.wav

Apple Silicon notes (hard-won):
- fp32 only on MPS — fp16 produces NaNs.
- <=30 s per generation (1500 tokens @ 50 Hz) — the model's training cap.
- On 16 GB boxes, evict resident Ollama models FIRST (`ollama stop <model>`);
  Metal OOM corrupts output silently (no error, just garbage or empty audio).
"""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seconds", type=float, default=30.0, help="max 30 (model cap)")
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import scipy.io.wavfile
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-small", torch_dtype=torch.float32  # fp16 NaNs on MPS
    ).to(device)

    tokens = min(int(args.seconds * 50), 1500)  # 50 Hz frame rate, 30 s cap
    inputs = proc(text=[args.prompt], padding=True, return_tensors="pt").to(device)
    audio = model.generate(
        **inputs, do_sample=True, guidance_scale=args.guidance, max_new_tokens=tokens
    )
    sr = model.config.audio_encoder.sampling_rate
    wav = audio[0, 0].cpu().float().numpy()
    scipy.io.wavfile.write(args.out, sr, wav)
    print(f"wrote {args.out}  {len(wav)/sr:.1f}s @ {sr}Hz  device={device}")


if __name__ == "__main__":
    main()
