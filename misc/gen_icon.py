"""
gen_icon.py — regenerate the SO-WAT application icon.

Produces a 128x128 PNG of a sine wave dissolving into binary digits,
then writes the base64-encoded constant into ../gui.py, replacing the
existing _ICON_B64 block in-place.

Usage:
    python misc/gen_icon.py           # update gui.py
    python misc/gen_icon.py --preview # save icon_preview.png and exit
"""

import argparse
import base64
import io
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def build_icon(size: int = 128) -> bytes:
    img  = Image.new("RGBA", (size, size), (10, 14, 28, 255))
    draw = ImageDraw.Draw(img)
    cy   = size // 2
    cyan = (0, 220, 195)
    freq = 2.2
    amp  = 34
    fade = 0.68          # x-fraction where wave alpha reaches 0

    def sine_y(t: float) -> float:
        return cy - amp * math.sin(2 * math.pi * freq * t)

    # ── sine wave, fading right ───────────────────────────────────────────────
    prev = None
    for px in range(size):
        t = px / (size - 1)
        a = int(255 * max(0.0, 1.0 - t / fade) ** 1.3)
        if a == 0:
            prev = None
            continue
        cur = (px, int(sine_y(t)))
        if prev:
            draw.line([prev, cur], fill=(*cyan, a), width=3)
        prev = cur

    # ── binary digit field ────────────────────────────────────────────────────
    rng = np.random.default_rng(17)
    try:
        fnt   = ImageFont.truetype("consola.ttf", 12)
        fnt_s = ImageFont.truetype("consola.ttf", 10)
    except OSError:
        fnt = fnt_s = ImageFont.load_default()

    for ct in np.linspace(0.40, 0.97, 8):
        px      = int(ct * (size - 1))
        sy      = sine_y(ct)
        fade_in = min(1.0, max(0.0, (ct - 0.37) / 0.30))
        for row_off in [-34, -18, 0, 18, 34]:
            py = int(sy) + row_off
            if py < 4 or py > size - 4:
                continue
            proximity = 1.0 - abs(row_off) / 40.0
            a = int(255 * fade_in * (0.45 + 0.55 * proximity))
            if a < 15:
                continue
            bit = str(rng.integers(0, 2))
            f   = fnt if abs(row_off) <= 18 else fnt_s
            draw.text((px, py), bit, font=f, fill=(*cyan, a), anchor="mm")

    # ── soft glow ─────────────────────────────────────────────────────────────
    glow = img.filter(ImageFilter.GaussianBlur(radius=1.8))
    img  = Image.blend(img, glow, alpha=0.30)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def embed_in_gui(png_bytes: bytes, gui_path: Path) -> None:
    raw_b64 = base64.b64encode(png_bytes).decode()
    lines   = textwrap.wrap(raw_b64, 76)
    new_const = "_ICON_B64 = (\n"
    for line in lines:
        new_const += f'    "{line}"\n'
    new_const += ")"

    text = gui_path.read_text(encoding="utf-8")
    src_lines = text.splitlines(keepends=True)

    start = next(
        i for i, l in enumerate(src_lines) if l.startswith("_ICON_B64 = (")
    )
    end = next(
        i for i in range(start, len(src_lines)) if src_lines[i].strip() == ")"
    )

    new_lines = src_lines[:start] + [new_const + "\n"] + src_lines[end + 1:]
    gui_path.write_text("".join(new_lines), encoding="utf-8")
    print(f"Updated {gui_path}  ({len(lines)} base64 lines)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview", action="store_true",
        help="Save icon_preview.png next to this script and exit without touching gui.py",
    )
    args = parser.parse_args()

    png_bytes = build_icon()

    if args.preview:
        out = Path(__file__).parent / "icon_preview.png"
        out.write_bytes(png_bytes)
        print(f"Saved preview -> {out}")
        return

    gui_path = Path(__file__).parent.parent / "gui.py"
    if not gui_path.exists():
        print(f"ERROR: {gui_path} not found", file=sys.stderr)
        sys.exit(1)

    embed_in_gui(png_bytes, gui_path)


if __name__ == "__main__":
    main()
