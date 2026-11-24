/* lpr_pipe: one command on the NT98539A, from a JPEG to a plate string.
 *
 *   lpr_pipe lpd.bin ocr.bin charset.txt --lpd-scale S --lpd-zp Z  IMG.jpg
 *   lpr_pipe lpd.bin ocr.bin charset.txt --lpd-scale S --lpd-zp Z  --list files.txt
 *
 * Both models are opened once, side by side (two proc_ids), and every image goes
 * through the whole pipeline in this process:
 *
 *   JPEG decode (stb_image) -> letterbox 640x640, pad 114 -> LPD on the NPU
 *   -> top-scoring anchor: box + 4 corner keypoints, mapped back to the image
 *   -> detector-box widening (quad narrower than --box-fix x box: stretch to box)
 *   -> margin (--margin-x / --margin-y of the quad on each side)
 *   -> perspective warp 400x120 (bilinear) -> resize 128x32 (PIL-style bilinear)
 *   -> OCR on the NPU -> per-position argmax, stop at EOS -> plate
 *
 * An OCR model with two inputs (tools/lpr_gs.py: patch 128x64x3 + grid
 * 128x32x2 int16 Q14) moves the warp onto the NPU: the CPU only cuts a 128x64
 * patch around the quad from the original (2^L x 2^L block means, L the
 * smallest level the quad fits) and writes the 128x32 sampling grid; GridSample
 * (the NPU's CAL unit) and the OCR then run in one net_proc. The patch and the
 * grid match tools/lpr_gs.py bit for bit (same rounding).
 *
 * --canvas WxH runs a "canvas" LPD model (tools/lpd_rect.py): compiled with an
 * input size (the canvas, e.g. 1920x1152) bigger than its net input (640 x H),
 * so the NPU's preprocessor (NUE2) does the resize. net_get reports the net
 * size, so the canvas size comes from the command line and goes into the
 * input buffer's width / height / line_ofs. The CPU only copies the decoded image into the middle
 * of the canvas (planar, pad 114) -- no letterbox resize. An image larger than
 * the canvas is reported as "no-fit".
 *
 * --hw-jpeg (with --canvas and an LPD compiled for YUV420 input) decodes on the
 * NT98539A's hardware JPEG decoder (hd_videodec, see vdec_abi.h) instead of
 * stb_image: the YUV420 frame is copied into the middle of a YUV canvas (pad
 * Y 114 / UV 128, i.e. RGB 114), NUE2 converts it to RGB (BT.601 full range,
 * as JPEG) and resizes, and the OCR patch is built from the same canvas.
 * Needs the kflow_videodec module (insmod .../hdal/kflow_videodec/unit/
 * kflow_videodec.ko -- the vendor init script has the line commented out).
 * The anchor count follows from the net
 * size: 80*H/8 + 40*H/16 + 20*H/32 = 13.125 * H (8400 for 640x640).
 *
 * This mirrors tools/richard_board_eval.py, which does the same steps on the
 * host between two ai3_bench runs. The LPD output dequantises as
 * (int16 - zp) * scale with the compiler's quant_input_output.json values
 * (--lpd-scale / --lpd-zp); net_get's own scale_ratio/zero_point for this
 * output are in another convention and are not used.
 *
 * Output: one line per image,
 *   <file>\t<plate>\t<score>\t<lpd ms>\t<ocr ms>\t<total ms>\t<x0,y0;x1,y1;x2,y2;x3,y3>
 * with --verbose adding a per-step timing breakdown.
 */
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_STDIO_WRITE
#include "third_party/stb_image.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ai3_abi.h"
#include "vdec_abi.h"

#define DIE(...) do { fprintf(stderr, "FAIL: " __VA_ARGS__); cleanup(); exit(1); } while (0)

#define MAX_BLK 16
#define IO_PAD  (64 * 1024)
static struct { uintptr_t blk; void *va; uintptr_t pa; UINT32 size; } g_blk[MAX_BLK];
static int g_nblk = 0, g_hd_common = 0, g_hd_mem = 0, g_dev = 0;
static int g_cw = 0, g_ch = 0;          /* --canvas: NUE2 input size, 0 = CPU letterbox */
/* hardware JPEG decoder (--hw-jpeg) */
static int g_hwdec = 0, g_vinit = 0, g_vopen = 0, g_vstart = 0;
static uint32_t g_vpath = 0;
static uintptr_t g_bs_pa = 0, g_yuv_pa = 0, g_bs_blk = 0, g_yuv_blk = 0;
static unsigned char *g_bs_va = NULL, *g_yuv_va = NULL;
static UINT32 g_bs_max = 0, g_yuv_max = 0;
static int g_dec_w = 0, g_dec_h = 0;     /* decoder output buffer: canvas rounded up to 64 x 16 (480 wide
                                          * decodes with a 512 line offset) */
static int g_sw_fallback = 0;            /* images the hardware refused, decoded by stb_image instead */

typedef struct {
    const char *name;
    UINT32 proc_id;
    int opened, started;
    UINT32 in_path, out_path, in2_path;
    VENDOR_AI3_BUF in, out, in2;     /* in2: the GridSample grid, when there is one */
    int nin;
} net_t;
static net_t g_lpd = {.name = "lpd"}, g_ocr = {.name = "ocr"};

static void net_close(net_t *n)
{
    if (n->started) { vendor_ai3_net_stop(n->proc_id);  n->started = 0; }
    if (n->opened)  { vendor_ai3_net_close(n->proc_id); n->opened = 0; }
}

static void cleanup(void)
{
    if (g_vstart) { hd_videodec_stop(g_vpath); g_vstart = 0; }
    if (g_vopen)  { hd_videodec_close(g_vpath); g_vopen = 0; }
    if (g_vinit)  { hd_videodec_uninit(); g_vinit = 0; }
    net_close(&g_ocr);
    net_close(&g_lpd);
    for (int i = g_nblk - 1; i >= 0; i--) {
        if (g_blk[i].va)  hd_common_mem_munmap(g_blk[i].va, g_blk[i].size);
        if (g_blk[i].blk != HD_MEM_INVALID_BLK_64)
            hd_mem_release_block_64(g_blk[i].blk);
    }
    g_nblk = 0;
    if (g_dev)       { vendor_ai3_dev_uninit(); g_dev = 0; }
    if (g_hd_mem)    { hd_common_mem_uninit();  g_hd_mem = 0; }
    if (g_hd_common) { hd_common_uninit();      g_hd_common = 0; }
}

static int alloc_blk(UINT32 size, uintptr_t *pa_out, uintptr_t *va_out)
{
    if (g_nblk >= MAX_BLK) return -1;
    uintptr_t blk = hd_mem_get_block_64(HD_COMMON_MEM_COMMON_POOL, size, DDR_ID0);
    if (blk == HD_MEM_INVALID_BLK_64) return -1;
    uintptr_t pa = hd_mem_blk2pa_64(blk);
    void *va = pa ? hd_common_mem_mmap(HD_COMMON_MEM_MEM_TYPE_CACHE, (UINT32)pa, size) : NULL;
    if (!pa || !va) { hd_mem_release_block_64(blk); return -1; }
    g_blk[g_nblk].blk = blk; g_blk[g_nblk].va = va; g_blk[g_nblk].pa = pa; g_blk[g_nblk].size = size;
    g_nblk++;
    *pa_out = pa; *va_out = (uintptr_t)va;
    return 0;
}

static uintptr_t blk_of(uintptr_t pa)
{
    for (int i = 0; i < g_nblk; i++) if (g_blk[i].pa == pa) return g_blk[i].blk;
    return HD_MEM_INVALID_BLK_64;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

static long file_size(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long s = ftell(f);
    fclose(f);
    return s;
}

/* Load one nvt_model.bin, open it and bind fixed I/O buffers (as ai3_bench). */
static void net_open(net_t *n, const char *path, int plugin_cpu)
{
    long size = file_size(path);
    if (size <= 0) DIE("cannot read %s\n", path);
    uintptr_t mpa, mva;
    if (alloc_blk((UINT32)size, &mpa, &mva)) DIE("%s: no block for the model (%ld bytes)\n", n->name, size);
    FILE *f = fopen(path, "rb");
    if (fread((void *)mva, 1, (size_t)size, f) != (size_t)size) { fclose(f); DIE("short read %s\n", path); }
    fclose(f);
    if (memcmp((void *)mva, "NVT3", 4)) DIE("%s is not an NVT3 model\n", path);
    hd_common_mem_flush_cache((void *)mva, (UINT32)size);

    VENDOR_AI3_MODEL_INFO mi;
    memset(&mi, 0, sizeof(mi));
    mi.model_buf.pa = mpa; mi.model_buf.va = mva; mi.model_buf.size = (UINT32)size;
    if (vendor_ai3_dev_get(1, &mi) != HD_OK) DIE("%s: dev_get(model info)\n", n->name);

    VENDOR_AI3_PROC_CFG pc;
    memset(&pc, 0, sizeof(pc));
    pc.model_buf = mi.model_buf;
    for (int i = 0; i < AI3_PROC_BUF_MAX; i++) {
        UINT32 need = mi.proc_mem.buf[i].size;
        if (!need) continue;
        uintptr_t pa, va;
        if (alloc_blk(need, &pa, &va)) DIE("%s: no block for work buffer %d (%u bytes)\n", n->name, i, need);
        pc.proc_mem.buf[i].pa = pa; pc.proc_mem.buf[i].va = va; pc.proc_mem.buf[i].size = need;
    }
    if (plugin_cpu >= 0 && plugin_cpu < AI3_DEV_PLUGIN_MAX)
        pc.plugin[plugin_cpu] = vendor_ai_cpu1_get_engine();

    VENDOR_AI3_NET_INFO ni;
    memset(&ni, 0, sizeof(ni));
    if (vendor_ai3_net_open(&n->proc_id, &pc, &ni) != HD_OK) DIE("%s: net_open\n", n->name);
    n->opened = 1;
    if (!ni.in_buf_cnt || !ni.out_buf_cnt || !ni.in_path_list || !ni.out_path_list)
        DIE("%s: net_open reported no I/O\n", n->name);
    if (vendor_ai3_net_start(n->proc_id) != HD_OK) DIE("%s: net_start\n", n->name);
    n->started = 1;
    n->in_path = ni.in_path_list[0];
    n->out_path = ni.out_path_list[0];
    if (vendor_ai3_net_get(n->proc_id, n->in_path, &n->in) != HD_OK ||
        vendor_ai3_net_get(n->proc_id, n->out_path, &n->out) != HD_OK)
        DIE("%s: net_get\n", n->name);
    n->nin = ni.in_buf_cnt > 1 ? 2 : 1;
    if (n->nin == 2) {
        n->in2_path = ni.in_path_list[1];
        if (vendor_ai3_net_get(n->proc_id, n->in2_path, &n->in2) != HD_OK) DIE("%s: net_get(in 2)\n", n->name);
        if (n->in.channel == 2) {                      /* keep the image in `in`, the grid in `in2` */
            VENDOR_AI3_BUF t = n->in; n->in = n->in2; n->in2 = t;
            UINT32 tp = n->in_path; n->in_path = n->in2_path; n->in2_path = tp;
        }
        UINT32 need2 = n->in2.size ? n->in2.size : n->in2.width * n->in2.height * n->in2.channel * 2;
        uintptr_t pa2, va2;
        if (alloc_blk(need2 + IO_PAD, &pa2, &va2)) DIE("%s: no block for input 2\n", n->name);
        n->in2.pa = pa2; n->in2.va = va2; if (!n->in2.size) n->in2.size = need2;
        fprintf(stderr, "%s: input 2 %ux%ux%u fmt 0x%08x size %u line_ofs %llu\n", n->name, n->in2.width,
                n->in2.height, n->in2.channel, n->in2.fmt, n->in2.size, (unsigned long long)n->in2.line_ofs);
    }

    UINT32 in_need = n->in.size ? n->in.size : n->in.width * n->in.height * n->in.channel;
    UINT32 out_need = n->out.size ? n->out.size : n->out.width * n->out.height * n->out.channel * 4;
    uintptr_t pa, va;
    if (alloc_blk(in_need + IO_PAD, &pa, &va)) DIE("%s: no block for the input\n", n->name);
    n->in.pa = pa; n->in.va = va; if (!n->in.size) n->in.size = in_need;
    if (alloc_blk(out_need + IO_PAD, &pa, &va)) DIE("%s: no block for the output\n", n->name);
    n->out.pa = pa; n->out.va = va; if (!n->out.size) n->out.size = out_need;
    fprintf(stderr, "%s: %s %ld bytes, in %ux%ux%u, out %u bytes\n", n->name, path, size,
            n->in.width, n->in.height, n->in.channel, n->out.size);
}

/* The input must already be in n->in.va. Returns the net_proc time in ms. */
static double net_run(net_t *n)
{
    hd_common_mem_flush_cache((void *)n->in.va, n->in.size);
    if (n->in.p_next) hd_common_mem_flush_cache((void *)n->in.p_next->va, n->in.p_next->size);
    if (n->nin == 2) {
        hd_common_mem_flush_cache((void *)n->in2.va, n->in2.size);
        if (vendor_ai3_net_set(n->proc_id, n->in2_path, &n->in2) != HD_OK) DIE("%s: net_set(in 2)\n", n->name);
    }
    if (vendor_ai3_net_set(n->proc_id, n->in_path, &n->in) != HD_OK ||
        vendor_ai3_net_set(n->proc_id, n->out_path, &n->out) != HD_OK)
        DIE("%s: net_set\n", n->name);
    double t0 = now_ms();
    if (vendor_ai3_net_proc(n->proc_id) != HD_OK) DIE("%s: net_proc\n", n->name);
    double dt = now_ms() - t0;
    hd_common_mem_flush_cache((void *)n->out.va, n->out.size);
    return dt;
}

/* ---- image ops ------------------------------------------------------------ */

static inline unsigned char clamp8(float v) { return v < 0 ? 0 : v > 255 ? 255 : (unsigned char)(v + 0.5f); }

/* Copy an interleaved RGB image into the middle of a planar canvas cw x ch;
 * the border is (re)filled with 114 only when the image size changes. */
static void to_canvas(const unsigned char *rgb, int w, int h, unsigned char *chw, int cw, int ch,
                      int *left, int *top)
{
    static int last_w = -1, last_h = -1;
    *left = (cw - w) / 2; *top = (ch - h) / 2;
    if (w != last_w || h != last_h) { memset(chw, 114, (size_t)3 * cw * ch); last_w = w; last_h = h; }
    size_t plane = (size_t)cw * ch;
    for (int y = 0; y < h; y++) {
        const unsigned char *src = rgb + (size_t)y * w * 3;
        unsigned char *r = chw + (size_t)(y + *top) * cw + *left, *g = r + plane, *b = g + plane;
        for (int x = 0; x < w; x++) { r[x] = src[3 * x]; g[x] = src[3 * x + 1]; b[x] = src[3 * x + 2]; }
    }
}

/* cv2.resize INTER_LINEAR into a planar-RGB 640x640 canvas filled with 114. */
static void letterbox(const unsigned char *rgb, int w, int h, unsigned char *chw, int S,
                      float *gain, int *padx, int *pady)
{
    float g = fminf((float)S / h, (float)S / w);
    int nw = (int)lroundf(w * g), nh = (int)lroundf(h * g);
    int left = (S - nw) / 2, top = (S - nh) / 2;
    memset(chw, 114, (size_t)3 * S * S);
    float sx = (float)w / nw, sy = (float)h / nh;
    for (int y = 0; y < nh; y++) {
        float fy = (y + 0.5f) * sy - 0.5f; if (fy < 0) fy = 0;
        int y0 = (int)fy; float wy = fy - y0; int y1 = y0 + 1 < h ? y0 + 1 : h - 1;
        for (int x = 0; x < nw; x++) {
            float fx = (x + 0.5f) * sx - 0.5f; if (fx < 0) fx = 0;
            int x0 = (int)fx; float wx = fx - x0; int x1 = x0 + 1 < w ? x0 + 1 : w - 1;
            for (int c = 0; c < 3; c++) {
                float v = (1 - wy) * ((1 - wx) * rgb[(y0 * w + x0) * 3 + c] + wx * rgb[(y0 * w + x1) * 3 + c])
                        + wy * ((1 - wx) * rgb[(y1 * w + x0) * 3 + c] + wx * rgb[(y1 * w + x1) * 3 + c]);
                chw[(size_t)c * S * S + (size_t)(y + top) * S + (x + left)] = clamp8(v);
            }
        }
    }
    *gain = g; *padx = left; *pady = top;
}

/* TL, TR, BR, BL by x+y and y-x, as sort_corners() on the host. */
static void sort_corners(float p[4][2])
{
    float q[4][2]; int i, k;
    float best;
    for (k = 0, best = 1e30f, i = 0; i < 4; i++) if (p[i][0] + p[i][1] < best) { best = p[i][0] + p[i][1]; k = i; }
    q[0][0] = p[k][0]; q[0][1] = p[k][1];
    for (k = 0, best = -1e30f, i = 0; i < 4; i++) if (p[i][0] + p[i][1] > best) { best = p[i][0] + p[i][1]; k = i; }
    q[2][0] = p[k][0]; q[2][1] = p[k][1];
    for (k = 0, best = 1e30f, i = 0; i < 4; i++) if (p[i][1] - p[i][0] < best) { best = p[i][1] - p[i][0]; k = i; }
    q[1][0] = p[k][0]; q[1][1] = p[k][1];
    for (k = 0, best = -1e30f, i = 0; i < 4; i++) if (p[i][1] - p[i][0] > best) { best = p[i][1] - p[i][0]; k = i; }
    q[3][0] = p[k][0]; q[3][1] = p[k][1];
    memcpy(p, q, sizeof(q));
}

static float dist(const float *a, const float *b) { return hypotf(a[0] - b[0], a[1] - b[1]); }

/* Detector-box widening: see box_fix() in tools/richard_board_eval.py. */
static void box_fix(float c[4][2], const float box[4], float thresh)
{
    if (thresh <= 0) return;
    float qw = (dist(c[1], c[0]) + dist(c[2], c[3])) / 2, bw = box[2] - box[0];
    if (bw <= 0 || qw / bw >= thresh) return;
    float u[2] = {(c[1][0] - c[0][0]) + (c[2][0] - c[3][0]), (c[1][1] - c[0][1]) + (c[2][1] - c[3][1])};
    float un = hypotf(u[0], u[1]) + 1e-9f; u[0] /= un; u[1] /= un;
    float ctr[2] = {0, 0};
    for (int i = 0; i < 4; i++) { ctr[0] += c[i][0] / 4; ctr[1] += c[i][1] / 4; }
    float bx[4][2] = {{box[0], box[1]}, {box[2], box[1]}, {box[2], box[3]}, {box[0], box[3]}};
    float lo = 1e30f, hi = -1e30f, pq[4];
    for (int i = 0; i < 4; i++) {
        float pb = (bx[i][0] - ctr[0]) * u[0] + (bx[i][1] - ctr[1]) * u[1];
        pq[i] = (c[i][0] - ctr[0]) * u[0] + (c[i][1] - ctr[1]) * u[1];
        lo = fminf(lo, fminf(pb, pq[i]));
        hi = fmaxf(hi, fmaxf(pb, pq[i]));
    }
    float lmin = fminf(pq[0], pq[3]), rmax = fmaxf(pq[1], pq[2]);
    for (int i = 0; i < 4; i++) {
        float d = (i == 0 || i == 3) ? lo - lmin : hi - rmax;
        c[i][0] += d * u[0]; c[i][1] += d * u[1];
    }
}

/* Grow along the quad's own axes, then clip to the image: expand() on the host. */
static void expand(float c[4][2], float mx, float my, int w, int h)
{
    float tl[2] = {c[0][0], c[0][1]}, tr[2] = {c[1][0], c[1][1]}, br[2] = {c[2][0], c[2][1]}, bl[2] = {c[3][0], c[3][1]};
    for (int k = 0; k < 2; k++) {
        float top = tr[k] - tl[k], bot = br[k] - bl[k], lft = bl[k] - tl[k], rgt = br[k] - tr[k];
        c[0][k] = tl[k] - mx * top - my * lft;
        c[1][k] = tr[k] + mx * top - my * rgt;
        c[2][k] = br[k] + mx * bot + my * rgt;
        c[3][k] = bl[k] - mx * bot + my * lft;
    }
    for (int i = 0; i < 4; i++) {
        c[i][0] = fminf(fmaxf(c[i][0], 0), w - 1);
        c[i][1] = fminf(fmaxf(c[i][1], 0), h - 1);
    }
}

/* Homography taking the rectangle (0,0)-(W-1,H-1) onto the quad: dst pixel -> src point. */
static void rect_to_quad(float c[4][2], int W, int H, double m[9])
{
    double dst[4][2] = {{0, 0}, {W - 1, 0}, {W - 1, H - 1}, {0, H - 1}};
    double A[8][9];
    for (int i = 0; i < 4; i++) {
        double x = dst[i][0], y = dst[i][1], u = c[i][0], v = c[i][1];
        double r0[9] = {x, y, 1, 0, 0, 0, -u * x, -u * y, u};
        double r1[9] = {0, 0, 0, x, y, 1, -v * x, -v * y, v};
        memcpy(A[2 * i], r0, sizeof(r0)); memcpy(A[2 * i + 1], r1, sizeof(r1));
    }
    for (int col = 0; col < 8; col++) {                     /* Gauss-Jordan, partial pivot */
        int piv = col;
        for (int r = col + 1; r < 8; r++) if (fabs(A[r][col]) > fabs(A[piv][col])) piv = r;
        for (int k = 0; k < 9; k++) { double t = A[col][k]; A[col][k] = A[piv][k]; A[piv][k] = t; }
        for (int r = 0; r < 8; r++) {
            if (r == col || A[col][col] == 0) continue;
            double f = A[r][col] / A[col][col];
            for (int k = col; k < 9; k++) A[r][k] -= f * A[col][k];
        }
    }
    for (int i = 0; i < 8; i++) m[i] = A[i][8] / A[i][i];
    m[8] = 1;
}

/* Lanczos-4 weights for a fractional offset x in [0,1): taps at -3..+4, as
 * OpenCV's interpolateLanczos4 (sinc(t) * sinc(t/4), normalised). */
static void lanczos4(float x, float wt[8])
{
    if (x < 1e-6f) { memset(wt, 0, 8 * sizeof(float)); wt[3] = 1; return; }
    float sum = 0;
    for (int i = 0; i < 8; i++) {
        double t = x + 3 - i, pt = M_PI * t;
        wt[i] = (float)(sin(pt) / pt * sin(pt / 4) / (pt / 4));
        sum += wt[i];
    }
    for (int i = 0; i < 8; i++) wt[i] /= sum;
}

static int g_warp_lanczos = 0;

/* Perspective warp to W x H RGB, zero outside the image. Bilinear by default;
 * --warp lanczos matches cv2.warpPerspective(INTER_LANCZOS4) (sub-pixel position
 * quantised to 1/32 as OpenCV does) -- 38 ms more per plate, no accuracy gain
 * on the 1508-image test set. */
static void warp(const unsigned char *rgb, int w, int h, float c[4][2], unsigned char *out, int W, int H)
{
    if (g_warp_lanczos) {
        double m[9];
        rect_to_quad(c, W, H, m);
        static float tab[32][8];
        static int tab_ready = 0;
        if (!tab_ready) { for (int k = 0; k < 32; k++) lanczos4(k / 32.0f, tab[k]); tab_ready = 1; }
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                double d = m[6] * x + m[7] * y + m[8];
                double sx = (m[0] * x + m[1] * y + m[2]) / d, sy = (m[3] * x + m[4] * y + m[5]) / d;
                long X = lround(sx * 32), Y = lround(sy * 32);
                int ix = (int)(X >> 5), iy = (int)(Y >> 5);
                const float *wx = tab[X & 31], *wy = tab[Y & 31];
                float acc[3] = {0, 0, 0};
                for (int j = 0; j < 8; j++) {
                    int yy = iy - 3 + j;
                    if (yy < 0 || yy >= h) continue;
                    const unsigned char *row = rgb + (size_t)yy * w * 3;
                    float rs[3] = {0, 0, 0};
                    for (int i = 0; i < 8; i++) {
                        int xx = ix - 3 + i;
                        if (xx < 0 || xx >= w) continue;
                        const unsigned char *px = row + xx * 3;
                        rs[0] += wx[i] * px[0]; rs[1] += wx[i] * px[1]; rs[2] += wx[i] * px[2];
                    }
                    acc[0] += wy[j] * rs[0]; acc[1] += wy[j] * rs[1]; acc[2] += wy[j] * rs[2];
                }
                out[(y * W + x) * 3 + 0] = clamp8(acc[0]);
                out[(y * W + x) * 3 + 1] = clamp8(acc[1]);
                out[(y * W + x) * 3 + 2] = clamp8(acc[2]);
            }
        return;
    }
    double m[9];
    rect_to_quad(c, W, H, m);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            double d = m[6] * x + m[7] * y + m[8];
            double sx = (m[0] * x + m[1] * y + m[2]) / d, sy = (m[3] * x + m[4] * y + m[5]) / d;
            int x0 = (int)floor(sx), y0 = (int)floor(sy);
            float wx = (float)(sx - x0), wy = (float)(sy - y0);
            for (int ch = 0; ch < 3; ch++) {
                float acc = 0;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        int xx = x0 + dx, yy = y0 + dy;
                        float v = (xx >= 0 && yy >= 0 && xx < w && yy < h) ? rgb[(yy * w + xx) * 3 + ch] : 0;
                        acc += v * (dx ? wx : 1 - wx) * (dy ? wy : 1 - wy);
                    }
                out[(y * W + x) * 3 + ch] = clamp8(acc);
            }
        }
}

/* PIL Image.resize(BILINEAR): separable triangle filter widened by the scale
 * (so a downscale averages), horizontal pass then vertical, uint8 in between. */
static void pil_resize(const unsigned char *src, int sw, int sh, unsigned char *dst, int dw, int dh)
{
    unsigned char *tmp = malloc((size_t)dw * sh * 3);
    for (int pass = 0; pass < 2; pass++) {
        int in = pass ? sh : sw, out = pass ? dh : dw;
        float scale = (float)in / out, fs = scale > 1 ? scale : 1, support = fs;
        for (int o = 0; o < out; o++) {
            float center = (o + 0.5f) * scale;
            int lo = (int)(center - support + 0.5f); if (lo < 0) lo = 0;
            int hi = (int)(center + support + 0.5f); if (hi > in) hi = in;
            float wsum = 0, wt[64]; int n = hi - lo; if (n > 64) n = 64;
            for (int k = 0; k < n; k++) {
                float t = fabsf((k + lo - center + 0.5f) / fs);
                wt[k] = t < 1 ? 1 - t : 0; wsum += wt[k];
            }
            for (int k = 0; k < n; k++) wt[k] /= wsum;
            int lines = pass ? dw : sh;
            for (int l = 0; l < lines; l++)
                for (int ch = 0; ch < 3; ch++) {
                    float acc = 0;
                    for (int k = 0; k < n; k++)
                        acc += wt[k] * (pass ? tmp[((lo + k) * dw + l) * 3 + ch] : src[(l * sw + lo + k) * 3 + ch]);
                    if (pass) dst[(o * dw + l) * 3 + ch] = clamp8(acc);
                    else      tmp[(l * dw + o) * 3 + ch] = clamp8(acc);
                }
        }
    }
    free(tmp);
}

/* GridSample inputs (see tools/lpr_gs.py: patch_and_grid, grid_q).
 * patch: PW x PH planar RGB, each pixel the rounded mean of a 2^L block of the
 * original, 114 outside it; grid: 2 x OH x OW int16, round(g * 2^14), g the
 * align_corners=False normalised patch coordinate of each output pixel. */
#define PW 128
#define PH 64
#define GFRAC 14
static int gs_inputs(const unsigned char *rgb, int w, int h, float c[4][2], unsigned char *patch, short *grid,
                     int OW, int OH)
{
    float xmin = c[0][0], xmax = c[0][0], ymin = c[0][1], ymax = c[0][1];
    for (int i = 1; i < 4; i++) {
        xmin = fminf(xmin, c[i][0]); xmax = fmaxf(xmax, c[i][0]);
        ymin = fminf(ymin, c[i][1]); ymax = fmaxf(ymax, c[i][1]);
    }
    int L = 0;
    while ((xmax - xmin) / (float)(1 << L) > PW - 2 || (ymax - ymin) / (float)(1 << L) > PH - 2) L++;
    int s = 1 << L, ss = s * s;
    double cx = ((double)xmax + xmin) / 2, cy = ((double)ymax + ymin) / 2;
    int ox = (int)rint(cx - PW * s / 2.0), oy = (int)rint(cy - PH * s / 2.0);
    for (int py = 0; py < PH; py++)
        for (int px = 0; px < PW; px++) {
            unsigned sum[3] = {0, 0, 0};
            for (int j = 0; j < s; j++) {
                int y = oy + py * s + j;
                for (int i = 0; i < s; i++) {
                    int x = ox + px * s + i;
                    if (x < 0 || y < 0 || x >= w || y >= h) { sum[0] += 114; sum[1] += 114; sum[2] += 114; continue; }
                    const unsigned char *p = rgb + ((size_t)y * w + x) * 3;
                    sum[0] += p[0]; sum[1] += p[1]; sum[2] += p[2];
                }
            }
            for (int ch = 0; ch < 3; ch++)
                patch[ch * PW * PH + py * PW + px] = (unsigned char)((sum[ch] + ss / 2) / ss);
        }
    double m[9];
    rect_to_quad(c, OW, OH, m);
    for (int v = 0; v < OH; v++)
        for (int u = 0; u < OW; u++) {
            double d = m[6] * u + m[7] * v + m[8];
            double sx = (m[0] * u + m[1] * v + m[2]) / d, sy = (m[3] * u + m[4] * v + m[5]) / d;
            double gx = (2 * ((sx - ox - (s - 1) / 2.0) / s) + 1) / PW - 1;
            double gy = (2 * ((sy - oy - (s - 1) / 2.0) / s) + 1) / PH - 1;
            /* lpr_gs.py rounds the grid to float32 before quantising */
            double qx = rint((double)(float)gx * (1 << GFRAC)), qy = rint((double)(float)gy * (1 << GFRAC));
            grid[v * OW + u] = (short)fmax(-32768, fmin(32767, qx));
            grid[OW * OH + v * OW + u] = (short)fmax(-32768, fmin(32767, qy));
        }
    return L;
}

/* ---- hardware JPEG decode into a YUV canvas ---------------------------------- */
static VENDOR_AI3_BUF g_lpd_uv;          /* UV half of the LPD's YUV420 canvas input */

/* Decode `path` on the hardware decoder and copy the frame into the middle of
 * the YUV canvas (g_lpd.in: Y cw x ch, then UV cw x ch/2). Returns 0 and the
 * image size and placement, or -1. */
static int hw_decode_to_canvas(const char *path, int *w_out, int *h_out, int *left, int *top, double *t_read,
                               double *t_hw)
{
    double tr = now_ms();
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t n = fread(g_bs_va, 1, g_bs_max, f);
    fclose(f);
    if (n == 0 || n >= g_bs_max) return -1;
    hd_common_mem_flush_cache(g_bs_va, (UINT32)n);
    double t0 = now_ms();
    *t_read = t0 - tr;
    vdec_bs_t bs;
    memset(&bs, 0, sizeof(bs));
    bs.sign = VDEC_FOURCC('V', 'S', 'T', 'M');
    bs.vcodec_format = VDEC_CODEC_JPEG;
    bs.phy_addr = g_bs_pa;
    bs.size = (uint32_t)n;
    bs.blk = g_bs_blk;
    vdec_frame_t uo;
    memset(&uo, 0, sizeof(uo));
    uo.sign = VDEC_FOURCC('V', 'F', 'R', 'M');
    uo.pxlfmt = PXLFMT_YUV420;
    uo.w = (uint32_t)g_dec_w;
    uo.h = (uint32_t)g_dec_h;
    uo.phy_addr[0] = g_yuv_pa;
    uo.phy_addr[1] = g_yuv_pa + (uintptr_t)uo.w * uo.h;
    uo.total_blk_size = g_yuv_max;
    uo.blk = g_yuv_blk;
    if (hd_videodec_push_in_buf(g_vpath, &bs, &uo, 1000) != 0) return -1;
    vdec_frame_t fr;
    memset(&fr, 0, sizeof(fr));
    if (hd_videodec_pull_out_buf(g_vpath, &fr, 1000) != 0) return -1;
    *t_hw = now_ms() - t0;
    int w = (int)fr.w, h = (int)fr.h, loff = (int)fr.loff[0];
    uintptr_t yoff = fr.phy_addr[0] - g_yuv_pa, uvoff = fr.phy_addr[1] - g_yuv_pa;
    int ok = w > 0 && h > 0 && w <= g_cw && h <= g_ch && loff >= w &&
             yoff + (uintptr_t)loff * h <= g_yuv_max && uvoff + (uintptr_t)loff * (h / 2) <= g_yuv_max;
    if (ok) {
        hd_common_mem_flush_cache(g_yuv_va, g_yuv_max);       /* drop stale lines: the decoder wrote by DMA */
        static int last_w = -1, last_h = -1;
        unsigned char *cy = (unsigned char *)g_lpd.in.va, *cuv = (unsigned char *)g_lpd_uv.va;
        *left = ((g_cw - w) / 2) & ~1; *top = ((g_ch - h) / 2) & ~1;     /* even, for 4:2:0 */
        if (w != last_w || h != last_h) {
            memset(cy, 114, (size_t)g_cw * g_ch); memset(cuv, 128, (size_t)g_cw * g_ch / 2);
            last_w = w; last_h = h;
        }
        int wc = (w + 1) & ~1;
        for (int y = 0; y < h; y++)
            memcpy(cy + (size_t)(y + *top) * g_cw + *left, g_yuv_va + yoff + (size_t)y * loff, w);
        for (int y = 0; y < (h + 1) / 2; y++)
            memcpy(cuv + (size_t)(y + *top / 2) * g_cw + *left, g_yuv_va + uvoff + (size_t)y * loff, wc);
        *w_out = w; *h_out = h;
    }
    hd_videodec_release_out_buf(g_vpath, &fr);
    return ok ? 0 : -1;
}

/* stb_image fallback for a JPEG the hardware refuses: RGB -> JFIF YCbCr 4:2:0
 * into the same canvas. */
static int sw_decode_to_canvas(const char *path, int *w_out, int *h_out, int *left, int *top)
{
    int w, h, n;
    unsigned char *rgb = stbi_load(path, &w, &h, &n, 3);
    if (!rgb) return -1;
    if (w > g_cw || h > g_ch) { stbi_image_free(rgb); return -1; }
    unsigned char *cy = (unsigned char *)g_lpd.in.va, *cuv = (unsigned char *)g_lpd_uv.va;
    memset(cy, 114, (size_t)g_cw * g_ch); memset(cuv, 128, (size_t)g_cw * g_ch / 2);
    *left = ((g_cw - w) / 2) & ~1; *top = ((g_ch - h) / 2) & ~1;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const unsigned char *p = rgb + ((size_t)y * w + x) * 3;
            cy[(size_t)(y + *top) * g_cw + x + *left] = clamp8(0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]);
        }
    for (int y = 0; y < h; y += 2)
        for (int x = 0; x < w; x += 2) {
            float u = 0, v = 0; int k = 0;
            for (int j = 0; j < 2 && y + j < h; j++)
                for (int i = 0; i < 2 && x + i < w; i++, k++) {
                    const unsigned char *p = rgb + ((size_t)(y + j) * w + x + i) * 3;
                    u += -0.168736f * p[0] - 0.331264f * p[1] + 0.5f * p[2];
                    v += 0.5f * p[0] - 0.418688f * p[1] - 0.081312f * p[2];
                }
            unsigned char *uv = cuv + (size_t)((y + *top) / 2) * g_cw + x + *left;
            uv[0] = clamp8(u / k + 128); uv[1] = clamp8(v / k + 128);
        }
    stbi_image_free(rgb);
    *w_out = w; *h_out = h;
    return 0;
}

/* gs_inputs() from the YUV canvas: each patch pixel is the 2^L block mean of
 * Y, U, V, converted with JFIF (BT.601 full range) -- what a JPEG decoder's
 * RGB is. Corners are in image coordinates; (left, top) places the image in
 * the canvas, whose border is already RGB 114. */
static int gs_inputs_yuv(int left, int top, float cimg[4][2], unsigned char *patch, short *grid, int OW, int OH)
{
    const unsigned char *cy = (const unsigned char *)g_lpd.in.va, *cuv = (const unsigned char *)g_lpd_uv.va;
    float c[4][2];
    for (int i = 0; i < 4; i++) { c[i][0] = cimg[i][0] + left; c[i][1] = cimg[i][1] + top; }
    float xmin = c[0][0], xmax = c[0][0], ymin = c[0][1], ymax = c[0][1];
    for (int i = 1; i < 4; i++) {
        xmin = fminf(xmin, c[i][0]); xmax = fmaxf(xmax, c[i][0]);
        ymin = fminf(ymin, c[i][1]); ymax = fmaxf(ymax, c[i][1]);
    }
    int L = 0;
    while ((xmax - xmin) / (float)(1 << L) > PW - 2 || (ymax - ymin) / (float)(1 << L) > PH - 2) L++;
    int s = 1 << L;
    double cx = ((double)xmax + xmin) / 2, cy_ = ((double)ymax + ymin) / 2;
    int ox = (int)rint(cx - PW * s / 2.0), oy = (int)rint(cy_ - PH * s / 2.0);
    for (int py = 0; py < PH; py++)
        for (int px = 0; px < PW; px++) {
            float sy = 0, su = 0, sv = 0;
            for (int j = 0; j < s; j++) {
                int y = oy + py * s + j;
                for (int i = 0; i < s; i++) {
                    int x = ox + px * s + i;
                    if (x < 0 || y < 0 || x >= g_cw || y >= g_ch) { sy += 114; su += 128; sv += 128; continue; }
                    sy += cy[(size_t)y * g_cw + x];
                    const unsigned char *uv = cuv + (size_t)(y / 2) * g_cw + (x & ~1);
                    su += uv[0]; sv += uv[1];
                }
            }
            float ss = (float)(s * s), Y = sy / ss, U = su / ss - 128, V = sv / ss - 128;
            patch[0 * PW * PH + py * PW + px] = clamp8(Y + 1.402f * V);
            patch[1 * PW * PH + py * PW + px] = clamp8(Y - 0.344136f * U - 0.714136f * V);
            patch[2 * PW * PH + py * PW + px] = clamp8(Y + 1.772f * U);
        }
    double m[9];
    rect_to_quad(c, OW, OH, m);
    for (int v = 0; v < OH; v++)
        for (int u = 0; u < OW; u++) {
            double d = m[6] * u + m[7] * v + m[8];
            double sx = (m[0] * u + m[1] * v + m[2]) / d, sy2 = (m[3] * u + m[4] * v + m[5]) / d;
            double gx = (2 * ((sx - ox - (s - 1) / 2.0) / s) + 1) / PW - 1;
            double gy = (2 * ((sy2 - oy - (s - 1) / 2.0) / s) + 1) / PH - 1;
            double qx = rint((double)(float)gx * (1 << GFRAC)), qy = rint((double)(float)gy * (1 << GFRAC));
            grid[v * OW + u] = (short)fmax(-32768, fmin(32767, qx));
            grid[OW * OH + v * OW + u] = (short)fmax(-32768, fmin(32767, qy));
        }
    return L;
}

/* ---- charset (UTF-8, one code point per class) ---------------------------- */
static int g_na = 8400;                 /* LPD anchors, set from the model's net size */
static char g_cs[64][5];
static int g_ncs = 0;
static void load_charset(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) DIE("cannot read %s\n", path);
    char buf[512]; size_t n = fread(buf, 1, sizeof(buf) - 1, f); fclose(f); buf[n] = 0;
    for (size_t i = 0; i < n && g_ncs < 64;) {
        unsigned char b = (unsigned char)buf[i];
        if (b == '\n' || b == '\r' || b == ' ') { i++; continue; }
        int len = b < 0x80 ? 1 : b < 0xE0 ? 2 : b < 0xF0 ? 3 : 4;
        memcpy(g_cs[g_ncs], buf + i, len); g_cs[g_ncs][len] = 0; g_ncs++; i += len;
    }
}

/* ---- one image ----------------------------------------------------------- */
typedef struct { float lpd_s, zp, margin_x, margin_y, box_fix, conf; int verbose; } opts_t;

static int run_image(const char *path, const opts_t *o, double *t_tot)
{
    double t0 = now_ms();
    int w = 0, h = 0, n;
    unsigned char *rgb = NULL;
    int left = 0, top = 0;
    double t_hw = 0, t_read = 0, t_dec, t_lb;
    int fallback = 0;
    float gain, px, py;
    int cw = g_cw, chh = g_ch;
    if (g_hwdec) {
        if (hw_decode_to_canvas(path, &w, &h, &left, &top, &t_read, &t_hw)) {
            double tf = now_ms();
            if (sw_decode_to_canvas(path, &w, &h, &left, &top)) {
                printf("%s\t\t0\t0\t0\t0\tdecode-failed\n", path);
                return 0;
            }
            fallback = 1; g_sw_fallback++;
            t_read = 0; t_hw = now_ms() - tf;
        }
        t_dec = t0 + t_read + t_hw;                 /* split: file read, decode, canvas copy */
        gain = 640.0f / cw; px = left * gain; py = top * gain;
        goto decoded;
    }
    rgb = stbi_load(path, &w, &h, &n, 3);
    if (!rgb) { printf("%s\t\t0\t0\t0\t0\tdecode-failed\n", path); return 0; }
    t_dec = now_ms();
    if (cw) {
        if (w > cw || h > chh) {
            printf("%s\t\t0\t0\t0\t0\tno-fit-%dx%d\n", path, w, h);
            stbi_image_free(rgb);
            return 0;
        }
        to_canvas(rgb, w, h, (unsigned char *)g_lpd.in.va, cw, chh, &left, &top);
        gain = 640.0f / cw; px = left * gain; py = top * gain;
    } else {
        int ipx, ipy;
        letterbox(rgb, w, h, (unsigned char *)g_lpd.in.va, 640, &gain, &ipx, &ipy);
        px = ipx; py = ipy;
    }
decoded:;
    const int NA = g_na;
    t_lb = now_ms();
    double npu_lpd = net_run(&g_lpd);
    double t_lpd = now_ms();

    /* 17 x NA int16, row-major (NA = 8400 for 640x640) */
    const short *q = (const short *)g_lpd.out.va;
    int best = 0;
    for (int k = 1; k < NA; k++) if (q[4 * NA + k] > q[4 * NA + best]) best = k;
#define DQ(r) (((float)q[(r) * NA + best] - o->zp) * o->lpd_s)
    float score = DQ(4);
    char plate[64] = "";
    float c[4][2] = {{0}};
    double npu_ocr = 0, t_post = 0, t_warp = 0, t_ocr = 0;
    if (score >= o->conf) {
        float box[4] = {(DQ(0) - px) / gain, (DQ(1) - py) / gain, (DQ(2) - px) / gain, (DQ(3) - py) / gain};
        box[0] = fminf(fmaxf(box[0], 0), w - 1); box[2] = fminf(fmaxf(box[2], 0), w - 1);
        box[1] = fminf(fmaxf(box[1], 0), h - 1); box[3] = fminf(fmaxf(box[3], 0), h - 1);
        for (int i = 0; i < 4; i++) {
            c[i][0] = fminf(fmaxf((DQ(5 + 3 * i) - px) / gain, 0), w - 1);
            c[i][1] = fminf(fmaxf((DQ(6 + 3 * i) - py) / gain, 0), h - 1);
        }
        sort_corners(c);
        box_fix(c, box, o->box_fix);
        expand(c, o->margin_x, o->margin_y, w, h);
        t_post = now_ms();

        if (g_hwdec) {
            gs_inputs_yuv(left, top, c, (unsigned char *)g_ocr.in.va, (short *)g_ocr.in2.va, 128, 32);
        } else if (g_ocr.nin == 2) {
            gs_inputs(rgb, w, h, c, (unsigned char *)g_ocr.in.va, (short *)g_ocr.in2.va, 128, 32);
        } else {
            static unsigned char wbuf[400 * 120 * 3], small[128 * 32 * 3];
            warp(rgb, w, h, c, wbuf, 400, 120);
            pil_resize(wbuf, 400, 120, small, 128, 32);
            unsigned char *in = (unsigned char *)g_ocr.in.va;      /* planar RGB 3x32x128 */
            for (int p = 0; p < 128 * 32; p++)
                for (int ch = 0; ch < 3; ch++) in[ch * 128 * 32 + p] = small[p * 3 + ch];
        }
        t_warp = now_ms();
        npu_ocr = net_run(&g_ocr);
        t_ocr = now_ms();

        const short *lg = (const short *)g_ocr.out.va;              /* 8 x (EOS + charset) */
        int V = g_ncs + 1;
        for (int pos = 0; pos < 8; pos++) {
            int arg = 0;
            for (int v = 1; v < V; v++) if (lg[pos * V + v] > lg[pos * V + arg]) arg = v;
            if (arg == 0) break;
            strcat(plate, g_cs[arg - 1]);
        }
    }
#undef DQ
    double t_end = now_ms();
    *t_tot = t_end - t0;
    printf("%s\t%s\t%.3f\t%.2f\t%.2f\t%.2f\t%.0f,%.0f;%.0f,%.0f;%.0f,%.0f;%.0f,%.0f\n", path, plate, score,
           npu_lpd, npu_ocr, *t_tot, c[0][0], c[0][1], c[1][0], c[1][1], c[2][0], c[2][1], c[3][0], c[3][1]);
    if (o->verbose && g_hwdec)
        fprintf(stderr, "  %dx%d  read %.1f  decode %.1f%s  canvas %.1f  LPD npu %.2f (+%.1f)  post %.2f  patch+grid %.1f"
                "  GS+OCR npu %.2f (+%.1f)  total %.1f ms\n", w, h, t_read, t_hw, fallback ? " (sw)" : " (hw)",
                t_lb - t_dec, npu_lpd, t_lpd - t_lb - npu_lpd, score >= o->conf ? t_post - t_lpd : 0,
                score >= o->conf ? t_warp - t_post : 0, npu_ocr, score >= o->conf ? t_ocr - t_warp - npu_ocr : 0, *t_tot);
    else if (o->verbose)
        fprintf(stderr, g_ocr.nin == 2 ?
                "  %dx%d  decode %.1f  letterbox/canvas %.1f  LPD npu %.2f (+%.1f)  post %.2f  patch+grid %.1f"
                "  GS+OCR npu %.2f (+%.1f)  total %.1f ms\n" :
                "  %dx%d  decode %.1f  letterbox %.1f  LPD npu %.2f (+%.1f)  post %.2f  warp+resize %.1f"
                "  OCR npu %.2f (+%.1f)  total %.1f ms\n", w, h, t_dec - t0, t_lb - t_dec, npu_lpd,
                t_lpd - t_lb - npu_lpd, score >= o->conf ? t_post - t_lpd : 0, score >= o->conf ? t_warp - t_post : 0,
                npu_ocr, score >= o->conf ? t_ocr - t_warp - npu_ocr : 0, *t_tot);
    if (rgb) stbi_image_free(rgb);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s lpd.bin ocr.bin charset.txt [--lpd-scale S --lpd-zp Z] "
                        "[--margin-x 0.12 --margin-y 0.20 --box-fix 0.6 --conf 0.25] "
                        "[--canvas WxH [--hw-jpeg]] [--warp lanczos|bilinear] [--plugin-cpu -1] [--pool-mb N --pool-blocks N] [--verbose] "
                        "(IMG ... | --list files.txt)\n", argv[0]);
        return 2;
    }
    opts_t o = {.lpd_s = 0.014473156f, .zp = -19780, .margin_x = 0.12f, .margin_y = 0.20f,
                .box_fix = 0.6f, .conf = 0.25f, .verbose = 0};
    int plugin_cpu = -1; UINT32 pool_mb = 0, blk_cnt = 12;
    const char *list = NULL;
    const char *imgs[256]; int nimg = 0;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--lpd-scale") && i + 1 < argc) o.lpd_s = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--lpd-zp") && i + 1 < argc) o.zp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--margin-x") && i + 1 < argc) o.margin_x = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--margin-y") && i + 1 < argc) o.margin_y = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--box-fix") && i + 1 < argc) o.box_fix = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--conf") && i + 1 < argc) o.conf = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--plugin-cpu") && i + 1 < argc) plugin_cpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pool-mb") && i + 1 < argc) pool_mb = (UINT32)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pool-blocks") && i + 1 < argc) blk_cnt = (UINT32)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--list") && i + 1 < argc) list = argv[++i];
        else if (!strcmp(argv[i], "--verbose")) o.verbose = 1;
        else if (!strcmp(argv[i], "--canvas") && i + 1 < argc) {
            if (sscanf(argv[++i], "%dx%d", &g_cw, &g_ch) != 2) { fprintf(stderr, "--canvas WxH\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--hw-jpeg")) g_hwdec = 1;
        else if (!strcmp(argv[i], "--warp") && i + 1 < argc) g_warp_lanczos = strcmp(argv[++i], "bilinear") != 0;
        else if (nimg < 256) imgs[nimg++] = argv[i];
    }
    load_charset(argv[3]);
    if (g_hwdec && !g_cw) { fprintf(stderr, "--hw-jpeg needs --canvas\n"); return 2; }
    if (g_hwdec && blk_cnt < 14) blk_cnt = 14;      /* + bitstream and decoded-frame blocks */

    /* One pool for both models: every block must fit the largest allocation
     * (the bigger model), and there is one block per allocation -- 2 x (model,
     * two work buffers, input, output). */
    long s1 = file_size(argv[1]), s2 = file_size(argv[2]);
    if (s1 <= 0 || s2 <= 0) { fprintf(stderr, "cannot read the models\n"); return 1; }
    UINT32 blk = pool_mb ? pool_mb * 1024u * 1024u : (UINT32)(s1 > s2 ? s1 : s2) + 1024u * 1024u;
    blk = (blk + 0xFFFFF) & ~0xFFFFFu;
    if (hd_common_init(0) != HD_OK) DIE("hd_common_init\n");
    g_hd_common = 1;
    HD_COMMON_MEM_INIT_CONFIG *mc = calloc(1, sizeof(*mc));
    mc->pool_info[0].type = HD_COMMON_MEM_COMMON_POOL;
    mc->pool_info[0].blk_size = blk;
    mc->pool_info[0].blk_cnt = blk_cnt;
    mc->pool_info[0].ddr_id = DDR_ID0;
    HD_RESULT r = hd_common_mem_init(mc);
    free(mc);
    if (r != HD_OK) DIE("hd_common_mem_init(%u x %u) = %d\n", blk_cnt, blk, r);
    g_hd_mem = 1;
    VENDOR_AI3_DEV_CFG dc; memset(&dc, 0, sizeof(dc));
    if (vendor_ai3_dev_init(&dc) != HD_OK) DIE("vendor_ai3_dev_init\n");
    g_dev = 1;
    fprintf(stderr, "pool %u x %.0f MB; charset %d classes; warp %s\n", blk_cnt, blk / 1048576.0, g_ncs,
            g_warp_lanczos ? "lanczos4" : "bilinear");

    double t_open = now_ms();
    net_open(&g_lpd, argv[1], plugin_cpu);
    net_open(&g_ocr, argv[2], plugin_cpu);
    if (!g_hwdec && g_lpd.in.channel != 3) DIE("LPD input has %u channels, expected 3\n", g_lpd.in.channel);
    fprintf(stderr, "lpd: net_get input fmt 0x%08x channel %u\n", g_lpd.in.fmt, g_lpd.in.channel);
    {
        unsigned H = g_lpd.in.height;                  /* net 640 x H */
        if (g_lpd.in.width != 640 || H % 32) DIE("LPD net input %ux%u, expected 640 x 32k\n", g_lpd.in.width, H);
        g_na = (int)(H * 105 / 8);                     /* 13.125 * H */
        if (g_cw) {
            if ((unsigned)g_ch * 640u != H * (unsigned)g_cw)
                DIE("--canvas %dx%d does not scale to the net 640x%u\n", g_cw, g_ch, H);
            uintptr_t pa, va;
            UINT32 need = (UINT32)g_cw * g_ch * (g_hwdec ? 3 : 6) / 2;
            if (alloc_blk(need + IO_PAD, &pa, &va)) DIE("lpd: no block for the %dx%d canvas\n", g_cw, g_ch);
            g_lpd.in.pa = pa; g_lpd.in.va = va; g_lpd.in.size = need;
            g_lpd.in.width = g_cw; g_lpd.in.height = g_ch; g_lpd.in.line_ofs = g_cw;
            if (g_hwdec) {
                /* YUV420 input as the AI3 guide's net_set sample: a Y8 buffer whose
                 * p_next is the UV buffer */
                g_lpd.in.fmt = PXLFMT_Y8; g_lpd.in.channel = 1;
                g_lpd.in.size = (UINT32)g_cw * g_ch;
                g_lpd_uv = g_lpd.in;
                g_lpd_uv.fmt = PXLFMT_UV;
                g_lpd_uv.pa = pa + (uintptr_t)g_cw * g_ch; g_lpd_uv.va = va + (uintptr_t)g_cw * g_ch;
                g_lpd_uv.size = (UINT32)g_cw * g_ch / 2;
                g_lpd_uv.p_next = NULL;
                g_lpd.in.p_next = &g_lpd_uv;
            }
        } else if (H != 640) DIE("LPD net is 640x%u: give its --canvas\n", H);
        fprintf(stderr, "lpd: %s -> net 640x%u, %d anchors, in fmt 0x%08x, out channel_ofs %llu\n",
                g_cw ? "canvas (NUE2 resize)" : "CPU letterbox", H, g_na, g_lpd.in.fmt,
                (unsigned long long)g_lpd.out.channel_ofs);
    }
    if (g_ocr.nin == 1 && (g_ocr.in.width != 128 || g_ocr.in.height != 32 || g_ocr.in.channel != 3))
        DIE("OCR input is %ux%ux%u, expected 128x32x3\n", g_ocr.in.width, g_ocr.in.height, g_ocr.in.channel);
    if (g_ocr.nin == 2 && (g_ocr.in.width != PW || g_ocr.in.height != PH || g_ocr.in.channel != 3 ||
                           g_ocr.in2.width != 128 || g_ocr.in2.height != 32 || g_ocr.in2.channel != 2))
        DIE("GridSample OCR inputs are %ux%ux%u + %ux%ux%u, expected %dx%dx3 + 128x32x2\n", g_ocr.in.width,
            g_ocr.in.height, g_ocr.in.channel, g_ocr.in2.width, g_ocr.in2.height, g_ocr.in2.channel, PW, PH);
    if (g_hwdec) {
        uintptr_t pa, va;
        g_bs_max = 4u * 1024 * 1024;
        if (alloc_blk(g_bs_max, &pa, &va)) DIE("no block for the JPEG bitstream\n");
        g_bs_pa = pa; g_bs_va = (unsigned char *)va; g_bs_blk = blk_of(pa);
        g_dec_w = (g_cw + 63) & ~63; g_dec_h = (g_ch + 15) & ~15;
        g_yuv_max = (UINT32)g_dec_w * g_dec_h * 3 / 2;
        if (alloc_blk(g_yuv_max, &pa, &va)) DIE("no block for the decoded frame\n");
        g_yuv_pa = pa; g_yuv_va = (unsigned char *)va; g_yuv_blk = blk_of(pa);
        if (hd_videodec_init() != 0) DIE("hd_videodec_init (is kflow_videodec loaded?)\n");
        g_vinit = 1;
        if (hd_videodec_open(VDEC_IN0, VDEC_OUT0, &g_vpath) != 0) DIE("hd_videodec_open\n");
        g_vopen = 1;
        vdec_path_config_t pc;
        memset(&pc, 0, sizeof(pc));
        pc.codec_type = VDEC_CODEC_JPEG; pc.w = (uint32_t)g_dec_w; pc.h = (uint32_t)g_dec_h;
        if (hd_videodec_set(g_vpath, VDEC_PARAM_PATH_CONFIG, &pc) != 0) DIE("hd_videodec_set(PATH_CONFIG)\n");
        uint32_t codec = VDEC_CODEC_JPEG;
        if (hd_videodec_set(g_vpath, VDEC_PARAM_IN, &codec) != 0) DIE("hd_videodec_set(IN)\n");
        if (hd_videodec_start(g_vpath) != 0) DIE("hd_videodec_start\n");
        g_vstart = 1;
        fprintf(stderr, "hw-jpeg: decoder up, output buffer %dx%d, bitstream %u KB\n", g_dec_w, g_dec_h, g_bs_max / 1024);
    }
    if (g_lpd.out.size < (UINT32)(17 * g_na * 2)) DIE("LPD output is %u bytes, expected >= %d\n", g_lpd.out.size, 17 * g_na * 2);
    if (g_ocr.out.size < (UINT32)(8 * (g_ncs + 1) * 2))
        DIE("OCR output is %u bytes; does %s match this model?\n", g_ocr.out.size, argv[3]);
    fprintf(stderr, "both models open in %.0f ms\n", now_ms() - t_open);

    int done = 0; double sum = 0, t;
    if (list) {
        FILE *lf = fopen(list, "r");
        if (!lf) DIE("cannot open %s\n", list);
        char line[1024];
        while (fgets(line, sizeof(line), lf)) {
            char *nl = strchr(line, '\n'); if (nl) *nl = 0;
            if (!line[0]) continue;
            done += run_image(line, &o, &t); sum += t;
        }
        fclose(lf);
    }
    for (int i = 0; i < nimg; i++) { done += run_image(imgs[i], &o, &t); sum += t; }
    if (done) fprintf(stderr, "%d image(s), mean %.1f ms per image (decode..plate)%s\n", done, sum / done,
                      g_hwdec ? (g_sw_fallback ? "; some JPEGs fell back to stb_image" : "; all JPEGs decoded in hardware") : "");
    if (g_sw_fallback) fprintf(stderr, "%d image(s) decoded by stb_image after the hardware refused them\n", g_sw_fallback);
    cleanup();
    return 0;
}
