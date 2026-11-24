> 開發過程的技術筆記。文中提到的 `tools/lpr_gs.py`、`tools/lpd_rect.py`、`tools/richard_board_eval.py` 是訓練專案裡匯出模型、產生校正資料用的腳本,不在這個 repo;它們的產出(ONNX、校正資料)都已經放在 Release。路徑 `board/` 對應這個 repo 的根目錄。

# lpr_pipe — one command on the NT98539A: JPEG in, plate string out

Both models are loaded once. Every image then goes through the whole pipeline
on the board:

1. JPEG decode (stb_image)
2. letterbox to 640x640
3. LPD on the NPU
4. decode the top anchor, widen from the detector box, add the margin
5. perspective warp to 400x120, resize to 128x32
6. OCR on the NPU
7. take the argmax at each position

With a two-input OCR model (`ocr_gs.bin`, see below) step 5 moves to the NPU.

The steps follow `tools/richard_board_eval.py`, which does steps 4–5 on the host
between two `ai3_bench` runs.

## Build (host)

```bash
make -C board
```

This needs `gcc-aarch64-linux-gnu` and the onnx2novaic-98539 checkout for
`ai3_abi.h` and the link-time stubs (`AI3_BOARD=…/onnx2novaic-98539/board`).

## Run (board, in the NFS share)

```bash
./lpr_pipe lpd.bin ocr.bin charset.txt --verbose car.jpg
./lpr_pipe lpd.bin ocr.bin charset.txt --list files.txt > results.tsv
```

- `lpd.bin` is the YOLO26n-pose LPD, compiled int16.
- `ocr.bin` and `charset.txt` come from a package's `int16/` folder and its
  root.
- `--lpd-scale` / `--lpd-zp` default to the v4 LPD's values from its
  `quant_input_output.json`.
- `--margin-x 0.12 --margin-y 0.20 --box-fix 0.6 --conf 0.25` are the defaults.

Each output line is:

    file  plate  score  lpd_ms  ocr_ms  total_ms  corners

Measured on 1508 test images (2026-11-24):

| | value |
|---|---|
| exact plate | 99.67% (host-side pipeline: 99.73%) |
| median time per image, decode to plate | 108 ms |
| NPU time per image | 30.3 ms (LPD 22.2 + OCR 8.1) |

The small gap comes from interpolation: the board uses bilinear for the warp,
where the host uses OpenCV's Lanczos.

## Warp on the NPU (GridSample)

`tools/lpr_gs.py export` builds an OCR model with GridSample in front. It has
two inputs: a 128x64 patch cut from the original image, and a 128x32 sampling
grid (int16, value = int / 2^14). The NPU's CAL unit accepts at most about 8192
GridSample input pixels, which is why the patch is 128x64. Large plates are
cut from a 2x, 4x, … block-averaged copy of the image.

`lpr_pipe` switches to this mode by itself when the OCR model has two inputs.
The CPU then only cuts the patch and computes the grid.

```bash
python3 tools/lpr_gs.py export          # jobs/lpr-gs/ocr_gs.onnx
python3 tools/lpr_gs.py calib           # 200 patch/grid pairs from LPD training images
# compile: onnx2novaic-98539 branch feat-input, net_size 128x64,
#   args "--bit-mode 2 --gen-config precision/quan_asymme=2 --feat-input grid:128x32x2:int16:14"
./lpr_pipe lpd.bin ocr_gs.bin charset.txt --list files.txt > results_gs.tsv
```

On the same 1508 images (2026-11-24):

| | CPU warp (`ocr.bin`) | GridSample (`ocr_gs.bin`) |
|---|---|---|
| exact plate | 99.67% (1503) | 99.47% (1500) |
| median time per image | 108 ms | 86 ms |
| warp step on the CPU (mean) | 19.8 ms | 1.6 ms (patch + grid) |
| OCR NPU | 8.1 ms | 9.4 ms (GridSample + OCR) |

GridSample misses 3 plates that the CPU warp reads (MBY6900, 0887E9, 6056UX).
The float model on the PC misses the same 3, so quantisation is not the cause.
The cause is the sampling: GridSample samples the plate directly at 32 rows,
while the CPU path warps to 120 rows first and then averages down to 32.

## Letterbox on the NPU (NUE2 canvas)

`tools/lpd_rect.py` builds one LPD per camera resolution. Each one is compiled
with an input size (the "canvas") bigger than its net input (640 x H), so the
NPU's preprocessor (NUE2) does the resize in hardware. NUE2 stretches and does
not pad, so the canvas sizes are chosen to keep the aspect ratio. The CPU only
copies the decoded image into the middle of the canvas, with a 114 border:

    in_w = w rounded up to a multiple of 20
    H    = h * 640 / in_w rounded up to a multiple of 32
    in_h = H * in_w / 640

A rectangular net input is also less LPD work: 640x384 has 5040 anchors instead
of 8400.

```bash
python3 tools/lpd_rect.py export      # jobs/lpd-rect/lpd_<canvas>.onnx
python3 tools/lpd_rect.py eval        # PC float: letterbox 640 vs canvas (1502 vs 1503 / 1508)
python3 tools/lpd_rect.py calib       # 200 training images per canvas
# compile: net_size 640xH, preproc mean=0,scale=0.00392157, quant "weight_quan vlc",
#   args "--bit-mode 2 --in-size <canvas>"
./lpr_pipe lpd_960x576.bin ocr_gs.bin charset.txt --canvas 960x576 \
    --lpd-scale <output0_scale> --lpd-zp <output0_zeropoint> --list files.txt
```

`net_get` reports the net size, not the canvas, so `--canvas` is required.
`--lpd-scale` and `--lpd-zp` come from each model's `quant_input_output.json`.
An image larger than the canvas prints `no-fit-WxH`.

On the 1508 test images, with 6 canvases and GridSample OCR (2026-11-24):

| | CPU letterbox | NUE2 canvas |
|---|---|---|
| exact plate | 99.47% (1500) | 99.54% (1501) |
| median time per image | 86 ms | 50 ms |
| mean time per image | 102 ms | 70 ms |
| letterbox / canvas copy on the CPU (mean) | 30.5 ms | 6.2 ms |
| LPD NPU (mean) | 22.2 ms | 13.1 ms (resize included) |

The misses changed. 3 came back, and 5 new ones appeared. All 5 new ones are on
the edge: one LPD score is 0.225 against a 0.25 threshold, and on three plates
the corners are within 1–7 px of the CPU-letterbox run, yet the OCR reads one
character differently. JPEG decode is now the largest step: 18–22 ms at
960 wide, 75–107 ms at 1920 wide.

## JPEG decode on the hardware decoder (`--hw-jpeg`)

With `--hw-jpeg`, the NT98539A's hardware JPEG decoder (`hd_videodec`) replaces
stb_image. The flow per image:

1. The decoder writes YUV420.
2. `lpr_pipe` copies the frame into the middle of a YUV canvas. The border is
   Y 114 / UV 128, which is RGB 114.
3. The LPD is compiled for YUV420 input, so NUE2 converts the canvas to RGB
   (BT.601 full range, the same as JPEG) and resizes it.
4. The OCR patch is built from the same canvas.

A JPEG the hardware refuses falls back to stb_image. On the 1508 test images
none did.

Load the decoder's kernel module once after each boot. The vendor init script
has this line commented out:

```bash
insmod /lib/modules/$(uname -r)/hdal/kflow_videodec/unit/kflow_videodec.ko
```

Compile the LPDs like the canvas ones, with YUV420 input added:

```bash
#   args "--bit-mode 2 --in-size <canvas> --in-fmt yuv420 --gen-config preproc/yuv2rgb/csc=3"
#   (onnx2novaic-98539 branch feat-input: a YUV420 input is [preproc/in/channel] = 2)
./lpr_pipe lpd_yuv_960x576.bin ocr_gs.bin charset.txt --canvas 960x576 --hw-jpeg \
    --lpd-scale <output0_scale> --lpd-zp <output0_zeropoint> --list files.txt
```

`vdec_abi.h` declares only what the decoder needs. The board's libhdal (v3.500.1)
is newer than the 64-bit HDAL 3.0 headers in the Novaic toolchain, and three
fields moved. Their offsets were read off the library's own struct-copy code;
the header comment lists them. The header also fixes the block-handle type:
`ai3_abi.h` declares it `INT32`, which truncates the 64-bit handle.

`vdec_probe` decodes JPEGs and dumps the planes. `tools/vdec_check.py` compares
them with libjpeg: Y is within ±1 at 480, 960 and 1920 wide, and UV is within 2
at the 99th percentile.

Two mistakes stall the process in the kernel, and only a reboot clears them:

- passing the decoder a NULL output frame
- an output frame without `total_blk_size`

In both cases the decoder accepts the JPEG but never outputs a frame, and
`hd_videodec_stop` then waits forever. Both are fixed here; do not remove either.

On the 1508 test images (2026-11-24), GridSample OCR in every column:

| | stb_image + NUE2 canvas | hardware decode + NUE2 canvas |
|---|---|---|
| exact plate | 99.54% (1501) | 99.67% (1503) |
| median time per image | 50 ms | 29 ms |
| mean time per image | 70 ms | 31 ms |
| decode, 960 wide (mean) | 21.8 ms | 2.0 ms |
| decode, 1920x1200 (mean) | 107 ms | 8.0 ms |

The file read is timed separately (`read`, mean 1.6 ms here, served from the
board's page cache). Without it, the median is 27.6 ms. Most of what is left is
the NPU: LPD 13.5 ms plus GridSample+OCR 9.4 ms (mean).
