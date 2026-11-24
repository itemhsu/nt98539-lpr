/* vdec_probe: decode JPEGs with the NT98539A hardware decoder (hd_videodec).
 *
 *   insmod /lib/modules/$(uname -r)/hdal/kflow_videodec/kflow_videodec.ko   (once)
 *   vdec_probe [--dump] a.jpg [b.jpg ...]
 *
 * Built against the 64-bit HDAL headers shipped in the Novaic toolchain
 * (Tool/release.tar.bz2: toolchain/repo/proprietary/runtime64/hdal/include);
 * their function set matches the board's /usr/lib/libhdal.so and their
 * HD_VIDEO_PXLFMT_RGB888_PLANAR equals what net_get reports on the board.
 *
 * Per image: the JPEG goes into a common-pool block and push_in_buf decodes
 * it into a YUV420 frame (Y plane + interleaved UV plane) in a second block
 * that the caller supplies -- on this IPC part the decoder allocates no
 * output itself (a NULL frame gives "invalid raw addr, Y(0x0)" in dmesg).
 * pull_out_buf then returns that frame with its dims and line offsets.
 * Prints dims, line offsets and the time per decode; --dump writes
 * <jpg>.y and <jpg>.uv (loff-strided planes) next to the input.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stddef.h>
#include "hdal.h"
#include "hd_common.h"
#include "hd_videodec.h"

/* The board's libhdal (v3.500.1) is newer than these 3.0 headers; read off its
 * _hd_videodec_dewrap_bs / _dewrap_frame: HD_VIDEODEC_BS.blk is 8 bytes at 64
 * (header: 4 bytes at 60), HD_VIDEO_FRAME.blk is at 192 (header: 160) and the
 * frame has extra 8-byte fields at 136..152 -- total_blk_size, extrawrite_bgw,
 * extrawrite_bgh, in the order the library's own "pushin_frame:" debug line
 * prints them. The decoder refuses a user frame whose total_blk_size is 0
 * ("invalid raw addr ... size(0)"). Everything else sits at the header's
 * offsets. These wrappers put the new fields where the library reads them. */
typedef struct { HD_VIDEODEC_BS b; UINTPTR blk; } BOARD_BS;                 /* blk @ 64 */
#define FRAME_TOTAL_BLK_SIZE 136
typedef struct { HD_VIDEO_FRAME f; UINT8 gap[24]; UINTPTR blk; UINT8 tail[256]; } BOARD_FRAME; /* blk @ 192 */
_Static_assert(offsetof(BOARD_BS, blk) == 64, "bs blk");
_Static_assert(offsetof(BOARD_FRAME, blk) == 192, "frame blk");

#ifndef MAKEFOURCC
#define MAKEFOURCC(a, b, c, d) ((UINT32)(a) | ((UINT32)(b) << 8) | ((UINT32)(c) << 16) | ((UINT32)(d) << 24))
#endif

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

#define CHECK(x) do { HD_RESULT _r = (x); if (_r != HD_OK) { fprintf(stderr, "FAIL %s = %d\n", #x, _r); goto out; } } while (0)

int main(int argc, char **argv)
{
    int dump = 0, rc = 1;
    HD_PATH_ID path = 0;
    int opened = 0, started = 0, vinit = 0, minit = 0;
    HD_COMMON_MEM_VB_BLK bsblk = HD_COMMON_MEM_VB_INVALID_BLK, yblk = HD_COMMON_MEM_VB_INVALID_BLK;
    void *bsva = NULL;
    const UINT32 BS_MAX = 4 * 1024 * 1024, MAXW = 1920, MAXH = 1248;

    CHECK(hd_common_init(0));
    HD_COMMON_MEM_INIT_CONFIG *mc = calloc(1, sizeof(*mc));
    mc->pool_info[0].type = HD_COMMON_MEM_COMMON_POOL;
    mc->pool_info[0].blk_size = 4 * 1024 * 1024;            /* >= 1920x1248 YUV420 (3.6 MB) */
    mc->pool_info[0].blk_cnt = 8;
    mc->pool_info[0].ddr_id = DDR_ID0;
    HD_RESULT mr = hd_common_mem_init(mc);
    free(mc);
    CHECK(mr);
    minit = 1;
    CHECK(hd_videodec_init());
    vinit = 1;
    CHECK(hd_videodec_open(HD_VIDEODEC_0_IN_0, HD_VIDEODEC_0_OUT_0, &path));
    opened = 1;
    HD_VIDEODEC_PATH_CONFIG pc;
    memset(&pc, 0, sizeof(pc));
    pc.max_mem.codec_type = HD_CODEC_TYPE_JPEG;
    pc.max_mem.dim.w = MAXW;
    pc.max_mem.dim.h = MAXH;
    CHECK(hd_videodec_set(path, HD_VIDEODEC_PARAM_PATH_CONFIG, &pc));
    HD_VIDEODEC_IN vin;
    memset(&vin, 0, sizeof(vin));
    vin.codec_type = HD_CODEC_TYPE_JPEG;
    CHECK(hd_videodec_set(path, HD_VIDEODEC_PARAM_IN, &vin));
    CHECK(hd_videodec_start(path));
    started = 1;

    bsblk = hd_common_mem_get_block(HD_COMMON_MEM_COMMON_POOL, BS_MAX, DDR_ID0);
    if (bsblk == HD_COMMON_MEM_VB_INVALID_BLK) { fprintf(stderr, "FAIL no bitstream block\n"); goto out; }
    UINTPTR bspa = hd_common_mem_blk2pa(bsblk);
    const UINT32 YUV_MAX = MAXW * MAXH * 3 / 2;
    yblk = hd_common_mem_get_block(HD_COMMON_MEM_COMMON_POOL, YUV_MAX, DDR_ID0);
    if (yblk == HD_COMMON_MEM_VB_INVALID_BLK) { fprintf(stderr, "FAIL no output block\n"); goto out; }
    UINTPTR ypa = hd_common_mem_blk2pa(yblk);
    bsva = hd_common_mem_mmap(HD_COMMON_MEM_MEM_TYPE_CACHE, bspa, BS_MAX);
    if (!bspa || !bsva) { fprintf(stderr, "FAIL bitstream mmap\n"); goto out; }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dump")) { dump = 1; continue; }
        FILE *f = fopen(argv[i], "rb");
        if (!f) { fprintf(stderr, "cannot read %s\n", argv[i]); continue; }
        size_t n = fread(bsva, 1, BS_MAX, f);
        fclose(f);
        if (n == 0 || n >= BS_MAX) { fprintf(stderr, "%s: size %zu\n", argv[i], n); continue; }
        hd_common_mem_flush_cache(bsva, (UINT32)n);

        double t0 = now_ms();
        BOARD_BS bs;
        memset(&bs, 0, sizeof(bs));
        bs.b.sign = MAKEFOURCC('V', 'S', 'T', 'M');
        bs.b.vcodec_format = HD_CODEC_TYPE_JPEG;
        bs.b.ddr_id = DDR_ID0;
        bs.b.phy_addr = bspa;
        bs.b.size = (UINT32)n;
        bs.blk = bsblk;
        BOARD_FRAME uo;
        memset(&uo, 0, sizeof(uo));
        uo.f.sign = MAKEFOURCC('V', 'F', 'R', 'M');
        uo.f.ddr_id = DDR_ID0;
        uo.f.pxlfmt = HD_VIDEO_PXLFMT_YUV420;
        uo.f.dim.w = MAXW;
        uo.f.dim.h = MAXH;
        uo.f.phy_addr[0] = ypa;
        uo.f.phy_addr[1] = ypa + MAXW * MAXH;
        uo.blk = yblk;
        UINT64 total = YUV_MAX;
        memcpy((UINT8 *)&uo + FRAME_TOTAL_BLK_SIZE, &total, sizeof(total));
        HD_RESULT r = hd_videodec_push_in_buf(path, &bs.b, &uo.f, 500);
        if (r != HD_OK) { fprintf(stderr, "%s: push_in_buf = %d\n", argv[i], r); break; }
        BOARD_FRAME out;
        memset(&out, 0, sizeof(out));
        HD_VIDEO_FRAME *fr_ = &out.f;
        r = hd_videodec_pull_out_buf(path, fr_, 500);
        double t1 = now_ms();
        if (r != HD_OK) { fprintf(stderr, "%s: pull_out_buf = %d\n", argv[i], r); break; }
        HD_VIDEO_FRAME fr = out.f;
        printf("%s\t%ux%u\tfmt 0x%08x\tY %ux%u loff %u\tUV %ux%u loff %u\tdecode %.2f ms\n", argv[i],
               fr.dim.w, fr.dim.h, fr.pxlfmt, fr.pw[0], fr.ph[0], fr.loff[0], fr.pw[1], fr.ph[1], fr.loff[1], t1 - t0);
        if (dump) {
            /* UV's phy_addr is not page aligned, so map the whole output block
             * once and address both planes by offset from it */
            UINT8 *base = hd_common_mem_mmap(HD_COMMON_MEM_MEM_TYPE_CACHE, ypa, YUV_MAX);
            if (!base) { fprintf(stderr, "mmap output block failed\n"); }
            else {
                hd_common_mem_flush_cache(base, YUV_MAX);
                for (int p = 0; p < 2; p++) {
                    UINT32 sz = fr.loff[p] * (p ? fr.dim.h / 2 : fr.dim.h);
                    UINTPTR off = fr.phy_addr[p] - ypa;
                    if (off + sz > YUV_MAX) { fprintf(stderr, "plane %d outside the block\n", p); continue; }
                    char name[1024];
                    snprintf(name, sizeof(name), "%s.%s", argv[i], p ? "uv" : "y");
                    FILE *o = fopen(name, "wb");
                    if (o) { fwrite(base + off, 1, sz, o); fclose(o); }
                }
                printf("   Y @+0x%lx  UV @+0x%lx\n", (unsigned long)(fr.phy_addr[0] - ypa), (unsigned long)(fr.phy_addr[1] - ypa));
                hd_common_mem_munmap(base, YUV_MAX);
            }
        }
        hd_videodec_release_out_buf(path, fr_);
    }
    rc = 0;
out:
    if (started) hd_videodec_stop(path);
    if (opened) hd_videodec_close(path);
    if (bsva) hd_common_mem_munmap(bsva, BS_MAX);
    if (yblk != HD_COMMON_MEM_VB_INVALID_BLK) hd_common_mem_release_block(yblk);
    if (bsblk != HD_COMMON_MEM_VB_INVALID_BLK) hd_common_mem_release_block(bsblk);
    if (vinit) hd_videodec_uninit();
    if (minit) hd_common_mem_uninit();
    hd_common_uninit();
    return rc;
}
