#!/usr/bin/env python3
"""Halftone an image into black dots on a transparent background (for the coloured mosaic tiles of the hub).

The dots sit on a grid rotated by --angle; each dot's area follows the darkness of the image under it (amplitude
modulation), so a page can lay the PNG over any coloured ground.

    python3 tools/halftone.py in.jpg out.png [--crop x0 y0 x1 y1] [--width 960] [--cell 9] [--angle 45]
                                             [--gamma 1.0] [--invert]

--crop takes fractions of the image (0-1). --invert makes light parts dark (for dark source frames).
"""
import argparse
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


def halftone(img, width, cell, angle, gamma, invert):
    g = ImageOps.grayscale(img)
    g = g.resize((width, round(g.height * width / g.width)), Image.LANCZOS)
    if invert:
        g = ImageOps.invert(g)
    g = ImageOps.autocontrast(g, cutoff=1)
    a = np.asarray(g.filter(ImageFilter.GaussianBlur(cell / 3)), dtype=np.float32) / 255.0
    h, w = a.shape
    scale = 4  # draw large, then downsample for smooth dots
    out = Image.new("L", (w * scale, h * scale), 0)
    d = ImageDraw.Draw(out)
    th = math.radians(angle)
    c, s = math.cos(th), math.sin(th)
    r = int(math.hypot(w, h) / cell) + 2
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            # lattice point (i, j) rotated about the image centre
            x = w / 2 + (i * c - j * s) * cell
            y = h / 2 + (i * s + j * c) * cell
            if not (0 <= x < w and 0 <= y < h):
                continue
            dark = (1.0 - a[int(y), int(x)]) ** gamma
            rad = cell * 0.5 * math.sqrt(dark) * 1.12  # dots touch at full darkness
            if rad < 0.35:
                continue
            d.ellipse([(x - rad) * scale, (y - rad) * scale, (x + rad) * scale, (y + rad) * scale], fill=255)
    alpha = out.resize((w, h), Image.LANCZOS)
    rgba = Image.new("RGBA", (w, h), (17, 17, 16, 0))
    rgba.putalpha(alpha)
    return rgba


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--crop", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--cell", type=float, default=9)
    ap.add_argument("--angle", type=float, default=45)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--invert", action="store_true")
    a = ap.parse_args()
    img = Image.open(a.src).convert("RGB")
    if a.crop:
        x0, y0, x1, y1 = a.crop
        img = img.crop((round(x0 * img.width), round(y0 * img.height), round(x1 * img.width), round(y1 * img.height)))
    save(halftone(img, a.width, a.cell, a.angle, a.gamma, a.invert), a.out)


def save(rgba, out):
    """A PNG with one ink colour and 16 alpha levels in a palette: about a quarter of the size of RGBA, and the same on
    screen (at most 8/255 off at the dots' anti-aliased edges)."""
    level = rgba.getchannel("A").point(lambda v: round(v / 255 * 15))
    pal = Image.new("P", rgba.size)
    pal.putdata(list(level.getdata()))
    pal.putpalette(list(rgba.getpixel((0, 0))[:3]) * 16)
    pal.save(out, optimize=True, transparency=bytes(round(i * 255 / 15) for i in range(16)))


if __name__ == "__main__":
    main()
