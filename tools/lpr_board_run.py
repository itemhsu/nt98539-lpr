#!/usr/bin/env python3
"""Run the license-plate pipeline (lpr_pipe) on the NT98539A and score it.

    python3 tools/lpr_board_run.py fetch                 # release assets -> dist/lpr-board
    python3 tools/lpr_board_run.py mount --nfs-server IP # (after a board reboot) mount the share
    python3 tools/lpr_board_run.py deploy                # -> NFS share: binary, models, test images
    python3 tools/lpr_board_run.py run                   # telnet: one lpr_pipe per canvas
    python3 tools/lpr_board_run.py score                 # -> dist/lpr-board/report.{md,json,html}
    python3 tools/lpr_board_run.py all                   # the four in order

The host and the board share a directory over NFS: --share-host on this
machine is --share-board on the board (defaults ~/src/ai3probe and
/tmp/nfs/ai3probe). Everything goes into <share>/lpr-run/.

Each test image's file name carries its plate (the last `_` token, several
plates joined by `-`; reading any one counts). Images are copied as
img/NNNNN.jpg (names.tsv keeps the original names) and grouped by the
smallest canvas that holds them; each canvas has its own LPD model
(models.json) and gets one lpr_pipe run.
"""
import argparse
import html
import re
import json
import pathlib
import shutil
import statistics as st
import subprocess
import sys
import time
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent

TAG = "v1"
GH_REPO = "itemhsu/nt98539-lpr"
ASSETS = ("lpr-board-models.zip", "lpr-testset.zip")
STEPS = ("read", "decode", "canvas", "LPD npu", "post", "patch+grid", "GS+OCR npu", "total")
STEP_ZH = {"read": "讀檔", "decode": "JPEG 硬體解碼", "canvas": "複製到 YUV 畫布",
           "LPD npu": "LPD(NPU,含 NUE2 色彩轉換 + 縮放)", "post": "LPD 後處理(CPU)",
           "patch+grid": "車牌區塊 + 取樣網格(CPU)", "GS+OCR npu": "GridSample + OCR(NPU)",
           "total": "總計(解碼到車牌字串)"}


# Plate text in one canonical form, applied to both the reading and the file
# name: upper-case, keep A-Z 0-9 and the special prefixes, I -> 1, O -> 0
# (Taiwan plates use neither I nor O).
SPECIAL_PREFIXES = "軍使領外試臨機"
_DROP = re.compile(f"[^0-9A-Z{re.escape(SPECIAL_PREFIXES)}]")
_FOLD = str.maketrans("IO", "10")


def normalize(s):
    return _DROP.sub("", str(s).upper()).translate(_FOLD) if s else ""


def gts_of(name):
    """All plates in a file name's last `_` token; several are joined by `-` and
    reading any one of them counts."""
    tok = re.sub(r"\.jpe?g$", "", name, flags=re.I).split("_")[-1]
    return [g for g in (normalize(t) for t in tok.split("-")) if g]


# ---- telnet (busybox telnetd, root shell, no login) ------------------------
class Telnet:
    """Just enough of telnet for a busybox shell: refuse every option the
    server offers, send a command line, read what comes back. telnetlib is
    gone from Python 3.13."""
    IAC, DONT, DO, WONT, WILL, SB, SE = 255, 254, 253, 252, 251, 250, 240

    def __init__(self, host, port=23, timeout=10):
        import socket
        self.s = socket.create_connection((host, port), timeout=timeout)
        self.s.settimeout(0.3)
        self.read(1.0)

    def read(self, wait):
        out, t0 = bytearray(), time.time()
        while time.time() - t0 < wait:
            try:
                data = self.s.recv(4096)
            except OSError:
                continue
            if not data:
                break
            i = 0
            while i < len(data):
                b = data[i]
                if b == self.IAC and i + 1 < len(data):
                    cmd = data[i + 1]
                    if cmd in (self.DO, self.DONT, self.WILL, self.WONT) and i + 2 < len(data):
                        reply = self.WONT if cmd in (self.DO, self.DONT) else self.DONT
                        self.s.sendall(bytes([self.IAC, reply, data[i + 2]]))
                        i += 3
                        continue
                    if cmd == self.SB:
                        j = data.find(bytes([self.IAC, self.SE]), i)
                        i = j + 2 if j >= 0 else len(data)
                        continue
                    i += 2
                    continue
                out.append(b)
                i += 1
        return out.decode(errors="replace")

    def run(self, cmd, wait=5):
        self.s.sendall((cmd + "\n").encode())
        return self.read(wait)

    def close(self):
        self.s.close()


def pick(w, h, canvases):
    for c in sorted(canvases, key=lambda c: c["w"] * c["h"]):
        if c["w"] >= w and c["h"] >= h:
            return c
    return None


# ---- fetch ----------------------------------------------------------------
def cmd_fetch(a):
    out = pathlib.Path(a.dist)
    out.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        if not (out / name).exists():
            # the repo is public: a plain HTTPS download needs no gh login
            url = f"https://github.com/{GH_REPO}/releases/download/{TAG}/{name}"
            print("downloading", url)
            subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(out / name) + ".part", url], check=True)
            (out / f"{name}.part").rename(out / name)
        with zipfile.ZipFile(out / name) as z:
            z.extractall(out)
        print("fetched", out / name)


# ---- deploy ---------------------------------------------------------------
def cmd_deploy(a):
    from PIL import Image
    dist = pathlib.Path(a.dist)
    run = pathlib.Path(a.share_host).expanduser() / "lpr-run"
    (run / "img").mkdir(parents=True, exist_ok=True)
    models = json.loads((dist / "models" / "models.json").read_text())
    binary = pathlib.Path(a.binary) if a.binary else dist / "bin" / "lpr_pipe"
    shutil.copy2(binary, run / "lpr_pipe")
    (run / "lpr_pipe").chmod(0o755)                  # zipfile does not keep the mode
    for f in [m["file"] for m in models["lpd"]] + [models["ocr"], models["charset"]]:
        shutil.copy2(dist / "models" / f, run / f)
    images = sorted(p for p in (pathlib.Path(a.images) if a.images else dist / "testset").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg"))
    names, lists = [], {}
    for i, p in enumerate(images):
        dst = f"img/{i:05d}.jpg"
        if not (run / dst).exists() or (run / dst).stat().st_size != p.stat().st_size:
            shutil.copy2(p, run / dst)
        w, h = Image.open(p).size
        c = pick(w, h, models["lpd"])
        if c is None:
            print(f"skip {p.name}: {w}x{h} fits no canvas")
            continue
        names.append(f"{dst}\t{p.name}")
        lists.setdefault(c["canvas"], []).append(dst)
    (run / "names.tsv").write_text("\n".join(names) + "\n")
    sh = ["grep -q kflow_videodec /proc/modules || "
          "insmod /lib/modules/$(uname -r)/hdal/kflow_videodec/unit/kflow_videodec.ko",
          f"cd {a.share_board}/lpr-run", "rm -f results.tsv run.log done.flag"]
    for c in models["lpd"]:
        k = c["canvas"]
        if k not in lists:
            continue
        (run / f"list_{k}.txt").write_text("\n".join(lists[k]) + "\n")
        sh.append(f"./lpr_pipe {c['file']} {models['ocr']} {models['charset']} --canvas {k} --hw-jpeg "
                  f"--lpd-scale {c['scale']} --lpd-zp {c['zp']} --verbose --list list_{k}.txt "
                  f">> results.tsv 2>> run.log")
    sh.append("echo $? > done.flag")
    (run / "run.sh").write_text("\n".join(sh) + "\n")
    print(f"deployed {len(names)} images to {run}: " +
          ", ".join(f"{k} {len(v)}" for k, v in lists.items()))


# ---- run ------------------------------------------------------------------
def cmd_mount(a):
    """Mount the host's share on the board (the board forgets it on reboot)."""
    if not a.nfs_server:
        sys.exit("--nfs-server <this host's IP> is needed to mount")
    t = Telnet(a.board_ip)
    host_dir = str(pathlib.Path(a.share_host).expanduser())
    print(t.run(f"mkdir -p {a.share_board} && mount -t nfs -o nolock,vers=3 {a.nfs_server}:{host_dir} "
                f"{a.share_board} 2>&1; mount | grep {a.share_board}", wait=6))
    t.close()


def cmd_run(a):
    run = pathlib.Path(a.share_host).expanduser() / "lpr-run"
    flag = run / "done.flag"
    flag.unlink(missing_ok=True)
    t = Telnet(a.board_ip)
    t.run(f"sh {a.share_board}/lpr-run/run.sh > /dev/null 2>&1 &", wait=2)
    t.close()
    t0 = time.time()
    while not flag.exists() and time.time() - t0 < a.timeout:
        time.sleep(5)
    if not flag.exists():
        sys.exit(f"no done.flag after {a.timeout} s; see {run}/run.log")
    rc = flag.read_text().strip()
    res = run / "results.tsv"
    if rc != "0" or not res.exists() or not res.read_text().strip():
        log = (run / "run.log").read_text()[-1500:] if (run / "run.log").exists() else ""
        sys.exit(f"board run failed (exit {rc}); tail of run.log:\n{log}")
    print(f"board run finished in {time.time() - t0:.0f} s")


# ---- score ----------------------------------------------------------------
def parse_log(path):
    """Per image, in run order: the verbose timing line after each result."""
    import re
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r"\s+(\d+)x(\d+)\s", line)
        if not m:
            continue
        d = {k: float(v) for k, v in re.findall(
            r"(read|decode|canvas|LPD npu|post|patch\+grid|GS\+OCR npu|total) ([\d.]+)", line)}
        d["size"] = f"{m.group(1)}x{m.group(2)}"
        d["sw"] = "(sw)" in line
        rows.append(d)
    return rows


def cmd_score(a):
    dist = pathlib.Path(a.dist)
    run = pathlib.Path(a.share_host).expanduser() / "lpr-run"
    models = json.loads((dist / "models" / "models.json").read_text())
    names = dict(l.split("\t")[:2] for l in (run / "names.tsv").read_text().splitlines() if "\t" in l)
    canvas_of = {}
    for c in models["lpd"]:
        f = run / f"list_{c['canvas']}.txt"
        if f.exists():
            for l in f.read_text().split():
                canvas_of[l] = c["canvas"]
    results = [l.split("\t") for l in (run / "results.tsv").read_text().splitlines() if l.count("\t") >= 5]
    timings = parse_log(run / "run.log")
    # results.tsv and the verbose lines come out in the same order, one per decoded image
    timed = [r for r in results if r[6] not in ("decode-failed",) and not r[6].startswith("no-fit")]
    if len(timed) != len(timings):
        print(f"warning: {len(timed)} results vs {len(timings)} timing lines")
    tmap = {r[0]: t for r, t in zip(timed, timings)}
    groups = {}
    wrong = []
    for r in results:
        k = canvas_of.get(r[0], "?")
        g = groups.setdefault(k, {"n": 0, "ok": 0, "t": [], "sizes": {}})
        orig = names.get(r[0], r[0])
        gts = gts_of(orig)
        ok = normalize(r[1]) in gts
        g["n"] += 1
        g["ok"] += ok
        if not ok:
            wrong.append({"file": orig, "gt": "-".join(gts), "read": r[1] or r[6], "canvas": k})
        if r[0] in tmap:
            g["t"].append(tmap[r[0]])
            sz = tmap[r[0]]["size"]
            g["sizes"][sz] = g["sizes"].get(sz, 0) + 1
    order = [c["canvas"] for c in models["lpd"] if c["canvas"] in groups]
    order.sort(key=lambda k: (int(k.split("x")[0]), int(k.split("x")[1])))
    if not order:
        sys.exit("no results to score -- did the board run succeed? see run.log")
    allg = {"n": sum(groups[k]["n"] for k in order), "ok": sum(groups[k]["ok"] for k in order),
            "t": [t for k in order for t in groups[k]["t"]]}
    lpd_size = {c["canvas"]: (run / c["file"]).stat().st_size for c in models["lpd"]}
    ocr_size = (run / models["ocr"]).stat().st_size
    bin_size = (run / "lpr_pipe").stat().st_size

    def mean(g, k):
        v = [t[k] for t in g["t"] if k in t]
        return st.mean(v) if v else 0.0

    def label(k):
        sz = max(groups[k]["sizes"], key=groups[k]["sizes"].get) if groups[k]["sizes"] else k
        return sz.replace("x", "×")

    cols = [(label(k), groups[k], k) for k in order] + [("全部", allg, None)]
    rows = []
    rows.append(("張數", [str(g["n"]) for _, g, _ in cols]))
    rows.append(("**準確率**(整串車牌全對)", [f"**{g['ok'] / g['n'] * 100:.2f}%** ({g['ok']}/{g['n']})"
                                       for _, g, _ in cols]))
    for s_ in STEPS:
        vals = [f"{mean(g, s_):.1f}" for _, g, _ in cols]
        if s_ == "total":
            vals = [f"**{v}**" for v in vals]
            rows.append(("**" + STEP_ZH[s_] + " ms(平均)**", vals))
        else:
            rows.append((STEP_ZH[s_] + " ms(平均)", vals))
    rows.append(("總計 ms(中位數)", [f"{st.median([t['total'] for t in g['t']]):.1f}" for _, g, _ in cols]))
    rows.append(("總計扣除讀檔 ms(平均)", [f"{st.mean([t['total'] - t.get('read', 0) for t in g['t']]):.1f}"
                                    for _, g, _ in cols]))
    rows.append(("LPD 模型 KB(int16)", [f"{lpd_size[k] / 1024:,.0f}" for _, _, k in cols[:-1]] +
                 [f"{sum(lpd_size[k] for k in order) / 1024:,.0f}(6 個合計)"]))
    rows.append(("OCR 模型 KB(int16,含 GridSample)", [f"{ocr_size / 1024:,.0f}"] * len(cols)))
    rows.append(("畫布 → LPD 網路輸入", [f"{k.replace('x', '×')} → 640×{int(k.split('x')[1]) * 640 // int(k.split('x')[0])}"
                                   for _, _, k in cols[:-1]] + ["—"]))
    md = ["| | " + " | ".join(c[0] for c in cols) + " |",
          "|---|" + "---|" * len(cols)]
    md += [f"| {name} | " + " | ".join(v) + " |" for name, v in rows]
    sw = sum(t.get("sw", False) for t in allg["t"])
    notes = [f"`lpr_pipe` 執行檔 {bin_size / 1024:,.0f} KB。{len(allg['t'])} 張中 {sw} 張改用 CPU 解碼(硬體解碼器不接受時的備援)。",
             "讀檔時間取決於檔案從哪裡來(這裡是 NFS,板子已快取時約 1.5 ms)。"]
    report = {"columns": [c[0] for c in cols], "rows": rows, "wrong": wrong, "notes": notes,
              "accuracy": allg["ok"] / allg["n"], "n": allg["n"]}
    (dist / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    wr = "\n".join(f"| {w['file']} | {w['gt']} | {w['read']} |" for w in wrong)
    (dist / "report.md").write_text("\n".join(md) + "\n\n" + "\n".join(notes) +
                                    "\n\n讀錯的影像:\n\n| 檔名 | 正確 | 讀到 |\n|---|---|---|\n" + wr + "\n")
    write_html(dist / "report.html", cols, rows, wrong, notes)
    print("\n".join(md))
    print(f"\n{allg['ok']}/{allg['n']} = {allg['ok'] / allg['n'] * 100:.2f}%   wrong: " +
          ", ".join(f"{w['gt']}→{w['read']}" for w in wrong))
    print(f"wrote {dist}/report.md, report.json, report.html")


def write_html(path, cols, rows, wrong, notes):
    def cell(v):
        v = html.escape(v)
        while "**" in v:
            v = v.replace("**", "<b>", 1).replace("**", "</b>", 1)
        return v
    head = "".join(f"<th>{html.escape(c[0])}</th>" for c in cols)
    body = "".join(f"<tr><th>{cell(n)}</th>" + "".join(f"<td>{cell(v)}</td>" for v in vals) + "</tr>"
                   for n, vals in rows)
    wr = "".join(f"<tr><td>{html.escape(w['file'])}</td><td>{html.escape(w['gt'])}</td>"
                 f"<td>{html.escape(w['read'])}</td></tr>" for w in wrong)
    path.write_text(f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NT98539A 車牌辨識板上結果</title>
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd;--head:#f4f5f7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#e8e8e8;--mut:#9aa;--line:#333;--head:#22252b}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:16px}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;font-variant-numeric:tabular-nums;margin:12px 0}}
th,td{{border:1px solid var(--line);padding:6px 10px;text-align:right;white-space:nowrap}}
thead th,tbody th{{background:var(--head)}} tbody th{{text-align:left}} p{{color:var(--mut)}}
</style></head><body>
<h1>NT98539A 車牌辨識:板上 1508 張</h1>
<p>JPEG 硬體解碼 → NUE2(YUV→RGB + 縮放)→ LPD(YOLO26n-pose int16)→ GridSample 拉正 + PARSeq d192 OCR(int16)。每欄是一種解析度(各自一個畫布與 LPD 模型)。</p>
<div class="wrap"><table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>
{"".join(f"<p>{html.escape(n)}</p>" for n in notes)}
<h2>讀錯的影像({len(wrong)})</h2>
<div class="wrap"><table><thead><tr><th>檔名</th><th>正確</th><th>讀到</th></tr></thead><tbody>{wr}</tbody></table></div>
</body></html>
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "mount", "deploy", "run", "score", "all"])
    ap.add_argument("--dist", default=str(REPO / "dist" / "lpr-board"))
    ap.add_argument("--share-host", default="~/src/ai3probe")
    ap.add_argument("--share-board", default="/tmp/nfs/ai3probe")
    ap.add_argument("--board-ip", default="168.168.11.183")
    ap.add_argument("--nfs-server", help="this host's IP, for `mount`")
    ap.add_argument("--images", help="test images (default: <dist>/testset)")
    ap.add_argument("--binary", help="lpr_pipe to deploy (default: the prebuilt <dist>/bin/lpr_pipe)")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    for c in (["fetch", "deploy", "run", "score"] if a.cmd == "all" else [a.cmd]):
        {"fetch": cmd_fetch, "mount": cmd_mount, "deploy": cmd_deploy, "run": cmd_run, "score": cmd_score}[c](a)


if __name__ == "__main__":
    main()
