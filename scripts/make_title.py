#!/usr/bin/env python3
"""Render a transparent title-card PNG for ffmpeg overlay.

Run: uv run --with pillow python3 make_title.py --name "..." [--subtitle "..."] --out title.png
"""
import argparse
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def pick_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise SystemExit("no usable system font found — add one to FONT_CANDIDATES")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--name-size", type=int, default=96)
    ap.add_argument("--subtitle-size", type=int, default=40)
    ap.add_argument("--y", type=float, default=0.35, help="name baseline as fraction of height")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img = Image.new("RGBA", (args.width, args.height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def center(text: str, font: ImageFont.FreeTypeFont, y: int) -> None:
        x = (args.width - d.textlength(text, font=font)) / 2
        for dx, dy in ((2, 2), (3, 3)):  # soft shadow for legibility over video
            d.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 160))
        d.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    y0 = int(args.height * args.y)
    center(args.name, pick_font(args.name_size), y0)
    if args.subtitle:
        center(args.subtitle, pick_font(args.subtitle_size), y0 + args.name_size + 34)
    img.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
