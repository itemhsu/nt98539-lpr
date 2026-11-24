#!/usr/bin/env python3
"""Rebuild the per-canvas LPD calibration zips from the 200 calibration images.

    python3 tools/make_calib.py dist/lpr-board/calib/lpd dist/lpr-board/models/models.json dist/calib

Each image is scaled to fit the canvas and centred in it (pad 114), saved as
PNG at the canvas size -- the size the NPU preprocessor receives on the
board. One zip per canvas: calib_<W>x<H>.zip, the `calib` input of the
converter for that LPD (see models/BUILD.md).
"""
import json
import pathlib
import sys
import zipfile

import cv2
import numpy as np


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, models, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.is_file())     # every image, in name order
    for c in json.loads(models.read_text())["lpd"]:
        in_w, in_h = c["w"], c["h"]
        z = out / f"calib_{in_w}x{in_h}.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for i, f in enumerate(files):
                img = cv2.imread(str(f))
                if img is None:
                    sys.exit(f"cannot read {f}")
                h, w = img.shape[:2]
                s = min(in_w / w, in_h / h)
                im = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_LINEAR)
                can = np.full((in_h, in_w, 3), 114, np.uint8)
                t, l = (in_h - im.shape[0]) // 2, (in_w - im.shape[1]) // 2
                can[t:t + im.shape[0], l:l + im.shape[1]] = im
                zf.writestr(f"c{i:04d}.png", cv2.imencode(".png", can)[1].tobytes())
        print(f"{z}: {len(files)} images")


if __name__ == "__main__":
    main()
