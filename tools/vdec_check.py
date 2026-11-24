#!/usr/bin/env python3
"""Compare the board's hardware JPEG decode (board/vdec_probe --dump) with libjpeg.

    python3 tools/vdec_check.py ~/src/ai3probe/lpr/img/00038.jpg --loff 960

vdec_probe writes <jpg>.y (Y plane, loff bytes per row) and <jpg>.uv (UV
interleaved, half height). The JPEG's own YCbCr is decoded on the host with
PIL (draft mode off, no colour conversion: Image.open(...).convert("YCbCr")
would round-trip through RGB, so libjpeg's raw YCbCr via mode "YCbCr" is used).
Prints the absolute difference of Y and of the 4:2:0 chroma.
"""
import argparse
import pathlib

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jpg")
    ap.add_argument("--loff", type=int, required=True, help="Y line offset printed by vdec_probe")
    ap.add_argument("--uv-loff", type=int, default=0, help="UV line offset (default: same as Y)")
    a = ap.parse_args()
    im = Image.open(a.jpg)
    im.draft("YCbCr", im.size)               # keep libjpeg's YCbCr, no RGB round trip
    ycc = np.asarray(im.convert("YCbCr") if im.mode != "YCbCr" else im)
    h, w = ycc.shape[:2]
    p = pathlib.Path(a.jpg)
    y = np.frombuffer((p.parent / (p.name + ".y")).read_bytes(), np.uint8)
    rows = len(y) // a.loff
    y = y[:rows * a.loff].reshape(rows, a.loff)[:h, :w].astype(int)
    dy = np.abs(y - ycc[:, :, 0].astype(int))
    print(f"{p.name} {w}x{h}: Y |diff| mean {dy.mean():.3f}  p99 {np.percentile(dy, 99):.0f}  max {dy.max()}")
    ul = a.uv_loff or a.loff
    uv = np.frombuffer((p.parent / (p.name + ".uv")).read_bytes(), np.uint8)
    r2 = len(uv) // ul
    uv = uv[:r2 * ul].reshape(r2, ul)[:h // 2, :w].astype(int)
    u, v = uv[:, 0::2], uv[:, 1::2]
    cb = ycc[0::2, 0::2, 1].astype(int)[:u.shape[0], :u.shape[1]]
    cr = ycc[0::2, 0::2, 2].astype(int)[:v.shape[0], :v.shape[1]]
    for name, a_, b_ in (("U/Cb", u, cb), ("V/Cr", v, cr)):
        d = np.abs(a_ - b_)
        print(f"   {name} |diff| mean {d.mean():.3f}  p99 {np.percentile(d, 99):.0f}  max {d.max()}")


if __name__ == "__main__":
    main()
