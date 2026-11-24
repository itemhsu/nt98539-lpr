# 板上模型(int16 `.bin`)怎麼編譯

Release 已經附上編譯好的板上模型(`lpr-board-models.zip`),只想在板子上執行的話,不需要看這份。

這份說明如何從 Release 裡的 ONNX(`lpr-weights.zip` 的 `onnx/`)和校正資料(`lpr-calib.zip`)重新編譯出同樣的 `.bin`。

**需要 Novatek 的 Novaic 轉換工具鏈。** 它是原廠授權的軟體,不包含在這個 repo 裡。這些模型是用私有 repo `itemhsu/onnx2novaic-98539` 的 `convert.yml` workflow(`feat-input` 分支)在 CI 上編出來的。那個 workflow 在 Docker 裡呼叫工具鏈的 `compiler.V30`,這份文件以它的輸入參數來描述每個模型。

## LPD(每種解析度一個)

- 來源:`onnx/lpd_<畫布>.onnx`。
  - 由 YOLO26n-pose v4(`weights/lpd_yolo26n_pose_v4.pt`)以 640×H 輸入匯出。
  - 輸出在 `/model.23/Concat_5` 截斷,內容是 17 × anchor 數(box 4 + score 1 + 4 個角點 × (x, y, vis))。
- 校正資料:200 張 LPD 訓練影像,在 `lpr-calib.zip` 的 `calib/lpd/`。用下面的指令依畫布尺寸做成 6 個 zip:
  ```bash
  python3 tools/make_calib.py dist/lpr-board/calib/lpd dist/lpr-board/models/models.json dist/calib
  ```

| 畫布 | onnx | net_size | calib | args |
|---|---|---|---|---|
| 480x312 | `lpd_480x312.onnx` | 640x416 | `calib_480x312.zip` | `--bit-mode 2 --in-size 480x312 --in-fmt yuv420 --gen-config preproc/yuv2rgb/csc=3` |
| 960x384 | `lpd_960x384.onnx` | 640x256 | `calib_960x384.zip` | `--bit-mode 2 --in-size 960x384 --in-fmt yuv420 --gen-config preproc/yuv2rgb/csc=3` |
| 960x576 | `lpd_960x576.onnx` | 640x384 | `calib_960x576.zip` | `--bit-mode 2 --in-size 960x576 --in-fmt yuv420 --gen-config preproc/yuv2rgb/csc=3` |
| 1920x960 | `lpd_1920x960.onnx` | 640x320 | `calib_1920x960.zip` | `--bit-mode 2 --in-size 1920x960 --in-fmt yuv420 --gen-config preproc/yuv2rgb/csc=3` |
| 1920x1152 | `lpd_1920x1152.onnx` | 640x384 | `calib_1920x1152.zip` | `--bit-mode 2 --in-size 1920x1152 --in-fmt yuv420 --gen-config preproc/yuv2rgb/csc=3` |
| 1920x1248 | `lpd_1920x1248.onnx` | 640x416 | `calib_1920x1248.zip` | `--bit-mode 2 --in-size 1920x1248 --in-fmt yuv420 --gen-config preproc/yuv2rgb/csc=3` |

6 個 LPD 共用的參數:`chip 539A`、`op_mode 2`、`preproc mean=0,scale=0.00392157`、`quant "weight_quan vlc"`。

各參數的作用:

- `--in-size`:讓 NPU 的前處理單元 NUE2 把畫布縮放到網路輸入尺寸。
- `--in-fmt yuv420`:輸入是 Y 加交錯 UV 兩個平面。這時 `[preproc/in/channel]` 必須是 2,`feat-input` 分支的 `convert.py` 會自動設定。
- `csc=3`:BT.601 全範圍的 YUV→RGB 轉換,和 JPEG 相同。

編譯完成後,從各模型的 `quant_input_output.json` 取出輸出的 `output0_scale` 和 `output0_zeropoint`,寫進 `models.json` 的 `scale` / `zp`。`lpr_pipe` 用 `--lpd-scale` / `--lpd-zp` 接收這兩個值。

## OCR(前面接 GridSample)

- 來源:`onnx/ocr_gs.onnx`,由 PARSeq d192 單次(non-AR)解碼的 checkpoint `weights/parseq_d192_nonar_ep115.ckpt` 匯出。
- 兩個輸入:
  - `patch`:1×3×64×128 的車牌區塊。
  - `grid`:1×2×32×128 的取樣網格,實際值 = int16 / 2^14。
- 計算流程:GridSample(bilinear、zeros、align_corners=False)把區塊拉正成 128×32,再接 OCR。
- 輸出:8 × 40 的 logits(EOS 加上 39 個字元)。
- 校正資料:`lpr-calib.zip` 的 `calib/ocr/`(`patch/*.png` 和 `grid/*.bin` 各 200 組,依檔名配對),把整個 `calib/ocr/` 打包成一個 zip 當作 `calib`。

| onnx | net_size | preproc | args |
|---|---|---|---|
| `ocr_gs.onnx` | 128x64 | `mean=127.5,scale=0.00784314` | `--bit-mode 2 --gen-config precision/quan_asymme=2 --feat-input grid:128x32x2:int16:14` |

其他參數:`chip 539A`、`op_mode 2`、`quant "weight_quan vlc"`。

各參數的作用:

- `--feat-input`:宣告第二個輸入 `grid` 是 int16、Q14 定點的特徵輸入。轉換器會為它寫出多輸入的 `[blob]` 設定。
- `quan_asymme=2`:conv 權重用對稱量化,可以避開第一層 patch embedding 的延遲。

## 硬體限制(實測)

- **GridSample 的輸入上限:** NT98539A 的 CAL 單元最多約 8192 像素(128×64 可以,96×96 就超過),所以 CPU 要先剪出 128×64 的區塊。大車牌改從 2 倍、4 倍等區塊平均縮小的影像上剪。
- **NUE2 縮放不補邊:** 它只會拉伸,所以畫布尺寸要選成剛好等比縮放到 640×H:
  - 畫布寬 = 影像寬補到 20 的倍數
  - H = 32 的倍數
  - 畫布高 = H × 畫布寬 / 640
