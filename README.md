# nt98539-lpr:NT98539A 板上車牌辨識,一個指令從 JPEG 到車牌字串

```
JPEG ─► 硬體 JPEG 解碼(YUV420)─► 複製到 YUV 畫布 ─► NUE2:YUV→RGB + 縮放 ─► LPD(YOLO26n-pose,NPU)
     ─► 四角點 + 外擴(CPU)─► 128×64 車牌區塊 + 取樣網格(CPU)─► GridSample 拉正 + PARSeq OCR(NPU)─► 車牌字串
```

- 兩個模型只載入一次,之後每張影像在同一個程序裡跑完。
- 主要運算都在硬體上:JPEG 解碼由硬體解碼器做,色彩轉換和縮放由 NUE2 做,LPD 和 OCR 在 NPU 上跑。
- CPU 只負責複製像素和少量幾何計算。

## 下載

模型和測試集放在 GitHub Release [`v1`](https://github.com/itemhsu/nt98539-lpr/releases/tag/v1),不用登入就能下載:

| 檔案 | 內容 |
|---|---|
| `lpr-board-models.zip` | 板上執行要的全部:`models/` 有 6 個 LPD(每種解析度一個,int16,YUV420 輸入)、`ocr_gs.bin`(OCR int16,前面接 GridSample)、`charset.txt`、`models.json`(每個 LPD 的畫布尺寸和量化參數);`bin/lpr_pipe` 是預先編好的執行檔 |
| `lpr-testset.zip` | 1508 張測試影像。檔名最後一段 `_` 後面就是正確答案(多個車牌用 `-` 分隔,讀對其中之一就算對) |
| `lpr-weights.zip` | 原始權重和 ONNX:LPD `lpd_yolo26n_pose_v4.pt`、OCR `parseq_d192_nonar_ep115.ckpt`、6 個長方形輸入 LPD ONNX、`ocr_gs.onnx`、`ocr_float_128x32.onnx` |
| `lpr-calib.zip` | 重新編譯模型用的校正資料:`calib/lpd/`(200 張 LPD 訓練影像)、`calib/ocr/`(200 組車牌區塊和取樣網格),用法見 [`models/BUILD.md`](models/BUILD.md) |

## 需要的環境

- **主機(Linux):**
  - `gcc-aarch64-linux-gnu`
  - Python 3.10 以上,加上 `pillow`(重新做校正資料時另外需要 `numpy`、`opencv-python`)
  - `curl`(下載 Release 檔案)
  - NFS 匯出一個目錄給板子,預設是 `~/src/ai3probe`
- **板子:**
  - NT98539A,能用 telnet 登入,預設 IP 是 `168.168.11.183`
  - 主機的共享目錄掛在 `/tmp/nfs/ai3probe`
  - 板子沒有自動掛載時,用 `python3 tools/lpr_board_run.py mount --nfs-server <主機 IP>` 掛載
- **硬體解碼器的核心模組:** 每次開機後要載入一次。`tools/lpr_board_run.py run` 會自動檢查並載入;手動執行的指令是:
  ```bash
  insmod /lib/modules/$(uname -r)/hdal/kflow_videodec/unit/kflow_videodec.ko
  ```

## 建置

```bash
git clone https://github.com/itemhsu/nt98539-lpr.git && cd nt98539-lpr
make                   # → ./lpr_pipe(aarch64,約 100 KB)
```

- 不需要任何從板子複製出來的檔案。
- `ai3/` 有 AI3 / HDAL 的宣告,以及連結用的 stub 函式庫符號清單。
- 板上的 Novatek 函式庫(`/usr/lib`)是執行時才載入的。

## 執行

一條指令從下載跑到評分:

```bash
python3 tools/lpr_board_run.py all --binary lpr_pipe
```

也可以分四步:

```bash
python3 tools/lpr_board_run.py fetch     # 從 Release 下載並解壓到 dist/lpr-board/
python3 tools/lpr_board_run.py deploy --binary lpr_pipe   # 複製到 NFS 共享目錄,依解析度分組
python3 tools/lpr_board_run.py run       # telnet:載入解碼模組,每種解析度跑一次 lpr_pipe(約 75 秒)
python3 tools/lpr_board_run.py score     # → dist/lpr-board/report.md、report.json、report.html
```

不加 `--binary` 時,用 Release 裡預先編好的 `bin/lpr_pipe`。用 `--board-ip`、`--share-host`、`--share-board` 換成你的板子和共享目錄。

單張影像直接在板上執行:

```bash
./lpr_pipe lpd_yuv_960x576.bin ocr_gs.bin charset.txt --canvas 960x576 --hw-jpeg \
    --lpd-scale 0.012458804994821548 --lpd-zp -20587 --verbose car.jpg
```

- `--canvas`、`--lpd-scale`、`--lpd-zp` 依解析度從 `models.json` 取值。
- 輸出每張一行,欄位以 tab 分隔:`檔名 車牌 分數 LPD_ms OCR_ms 總計_ms 四角點`。
- 加 `--verbose` 會在 stderr 印出各步驟耗時。

## 預期結果

板子:NT98539A。測試集:1508 張。日期:2026-11-24。表格由 `tools/lpr_board_run.py score` 產生。

- 每一欄是一種解析度;每種解析度有自己的畫布和 LPD 模型,尺寸相近的影像歸到能裝下它的最小畫布。
- 時間是每張影像的平均值,單位 ms。

| | 480×300 | 960×380 | 960×540 | 1920×920 | 1920×1080 | 1920×1200 | 全部 |
|---|---|---|---|---|---|---|---|
| 張數 | 27 | 353 | 752 | 112 | 44 | 220 | 1508 |
| **準確率**(整串車牌全對) | **100.00%** (27/27) | **100.00%** (353/353) | **99.60%** (749/752) | **100.00%** (112/112) | **100.00%** (44/44) | **99.09%** (218/220) | **99.67%** (1503/1508) |
| 讀檔(NFS,未快取) | 5.8 | 8.4 | 6.6 | 24.7 | 19.6 | 41.8 | 13.9 |
| JPEG 硬體解碼 | 0.8 | 1.5 | 2.0 | 6.2 | 7.2 | 8.0 | 3.2 |
| 複製到 YUV 畫布 | 0.2 | 0.4 | 0.6 | 2.1 | 2.5 | 2.7 | 1.0 |
| LPD(NPU,含 NUE2 色彩轉換 + 縮放) | 14.6 | 9.3 | 13.9 | 13.8 | 16.6 | 18.0 | 13.5 |
| LPD 後處理(CPU) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 車牌區塊 + 取樣網格(CPU) | 1.3 | 1.8 | 1.5 | 6.3 | 9.2 | 2.3 | 2.3 |
| GridSample + OCR(NPU) | 9.4 | 9.4 | 9.4 | 9.4 | 9.4 | 9.4 | 9.4 |
| **總計**(從讀檔到車牌字串) | **32.4** | **31.0** | **34.4** | **62.9** | **65.0** | **82.8** | **43.6** |
| 總計(中位數) | 32.5 | 30.7 | 34.0 | 61.3 | 63.3 | 83.9 | 34.0 |
| **總計扣除讀檔** | **26.6** | **22.7** | **27.7** | **38.2** | **45.4** | **41.0** | **29.8** |
| LPD 模型 KB(int16) | 4,859 | 4,752 | 4,827 | 4,790 | 4,827 | 4,859 | 28,914(6 個合計) |
| OCR 模型 KB(int16,含 GridSample) | 12,220 | 12,220 | 12,220 | 12,220 | 12,220 | 12,220 | 12,220 |
| 畫布 → LPD 網路輸入 | 480×312 → 640×416 | 960×384 → 640×256 | 960×576 → 640×384 | 1920×960 → 640×320 | 1920×1152 → 640×384 | 1920×1248 → 640×416 | — |

- `lpr_pipe` 執行檔約 100 KB。1508 張全部由硬體解碼,沒有一張退回 CPU 解碼。
- **讀檔時間:** 這次是第一次從 NFS 讀取,檔案還沒被快取;實際部署時影像來自攝影機或記憶體,看「總計扣除讀檔」那一列。同一批影像第二次跑時讀檔約 1.5 ms,總計中位數約 29 ms。
- **讀錯的 5 張:**
  - 7175XJ:偵測分數 0.225,低於門檻 0.25,沒有輸出。
  - 875FP 讀成 875FR、8R775 讀成 8R173、0728PH 讀成 0728PR、ATH6811 讀成 ATH6611。
- 同一份 HTML 報告在 [`report.html`](report.html)。

## 版本演進

全部在同一個測試集(1508 張)上量測:

| 版本 | 準確率 | 每張中位數 |
|---|---|---|
| CPU 解碼 + CPU letterbox + CPU 拉正 | 99.67% | 108 ms |
| + 拉正改在 NPU 上用 GridSample | 99.47% | 86 ms |
| + letterbox 縮放改由 NUE2 做 | 99.54% | 50 ms |
| **+ JPEG 改用硬體解碼器(本版)** | **99.67%** | **34 ms**(檔案已快取時 29 ms) |

## 檔案

| 路徑 | 內容 |
|---|---|
| `src/lpr_pipe.c` | 板上程式:硬體 JPEG 解碼 → LPD → 外擴 → 車牌區塊 + 取樣網格 → GridSample + OCR |
| `src/vdec_abi.h` | 硬體 JPEG 解碼器(`hd_videodec`)的宣告,結構布局以板上函式庫為準 |
| `src/vdec_probe.c` | 只做硬體解碼並輸出 Y / UV 平面的測試程式(編譯需要 Novaic 工具鏈的 HDAL 標頭檔) |
| `ai3/` | AI3 / HDAL 宣告和連結用的 stub 符號清單 |
| `tools/lpr_board_run.py` | 從主機端一條龍:下載、掛載、部署、板上執行、評分 |
| `tools/vdec_check.py` | 比對硬體解碼和 libjpeg 的結果 |
| `tools/make_calib.py` | 依畫布尺寸產生 LPD 校正資料 |
| `models/BUILD.md` | 板上模型的編譯參數 |
| `NOTES.md` | 技術細節:各版本的設計、硬體限制、函式庫和標頭檔的差異、會讓板子卡住的錯誤 |

## 模型

- **LPD:** YOLO26n-pose v4,偵測車牌框和四個角點。每種解析度依自己的畫布匯出一個長方形輸入的模型,編成 int16,輸入是 YUV420。
- **OCR:** PARSeq d192 單次解碼,前面接 GridSample,編成 int16。

編譯參數和校正資料的做法見 [`models/BUILD.md`](models/BUILD.md)。
