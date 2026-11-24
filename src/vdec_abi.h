/* Hardware JPEG decoder (hd_videodec) ABI for the NT98539A's libhdal v3.500.1.
 *
 * Sources:
 *   hd_videodec.h / hd_type.h / hd_common.h, 64-bit HDAL 3.0 headers in the
 *   Novaic toolchain (Tool/release.tar.bz2: toolchain/repo/proprietary/
 *   runtime64/hdal/include) -- not redistributed; the values below were
 *   printed from them with a compiled probe.
 *   HD_VIDEO_PXLFMT_UV: runtime64/hdal/vendor/ai2/include/vendor_ai.h.
 *
 * The board's library is newer than those headers. Read off its
 * _hd_videodec_dewrap_bs / _hd_videodec_dewrap_frame (the code that copies the
 * caller's structs into the ioctl) and its "pushin_frame:" debug line:
 *   HD_VIDEODEC_BS.blk     8 bytes at 64   (header: 4 bytes at 60)
 *   HD_VIDEO_FRAME.blk     at 192          (header: 160)
 *   total_blk_size         at 136          (not in the header; 0 is refused
 *                                           with "invalid raw addr ... size(0)")
 *   extrawrite_bgw / _bgh  at 144 / 152    (left 0)
 * Every other field the library reads sits at the header's offset.
 *
 * Also: HD_COMMON_MEM_VB_BLK is a 64-bit handle (UINTPTR) in the 64-bit
 * headers. ai3_abi.h declares it INT32, which truncates the handle; the
 * *_64 aliases below bind the same symbols with the 64-bit types.
 */
#ifndef VDEC_ABI_H
#define VDEC_ABI_H

#include <stddef.h>
#include <stdint.h>

#define VDEC_IN0                 0x23000100u   /* HD_VIDEODEC_0_IN_0 */
#define VDEC_OUT0                0x23000001u   /* HD_VIDEODEC_0_OUT_0 */
#define VDEC_PARAM_PATH_CONFIG   2
#define VDEC_PARAM_IN            4
#define VDEC_CODEC_JPEG          0
#define PXLFMT_YUV420            0x520c0420u
#define PXLFMT_Y8                0x51080400u
#define PXLFMT_UV                0x51100044u
#define VDEC_FOURCC(a, b, c, d)  ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24))

typedef struct {                  /* HD_VIDEODEC_PATH_CONFIG, 140 bytes */
    uint32_t codec_type;          /* max_mem.codec_type */
    uint32_t w, h;                /* max_mem.dim */
    uint8_t  rest[140 - 12];      /* NVR-only fields and data_pool[], left 0 */
} vdec_path_config_t;

typedef struct {                  /* HD_VIDEODEC_BS as the board reads it */
    uint32_t sign;                /* 'VSTM' */
    uint32_t pad0;
    void    *p_next;
    uint32_t vcodec_format;       /* @16 */
    uint32_t pad1;
    uint64_t count;               /* @24 */
    uint64_t timestamp;           /* @32 */
    uint32_t ddr_id;              /* @40 */
    uint32_t pad2;
    uintptr_t phy_addr;           /* @48 */
    uint32_t size;                /* @56 */
    uint32_t pad3;
    uintptr_t blk;                /* @64 */
} vdec_bs_t;

typedef struct {                  /* HD_VIDEO_FRAME as the board reads/writes it */
    uint32_t sign;                /* 'VFRM' */
    uint32_t pad0;
    void    *p_next;
    uint32_t ddr_id;              /* @16 */
    uint32_t pxlfmt;              /* @20 */
    uint32_t w, h;                /* @24 dim */
    uint64_t count;               /* @32 */
    uint64_t timestamp;           /* @40 */
    uint32_t pw[4];               /* @48 */
    uint32_t ph[4];               /* @64 */
    uint32_t loff[4];             /* @80 */
    uintptr_t phy_addr[4];        /* @96 */
    uint32_t poc_info;            /* @128 */
    uint32_t pad1;
    uint64_t total_blk_size;      /* @136 */
    uint64_t extrawrite_bgw;      /* @144 */
    uint64_t extrawrite_bgh;      /* @152 */
    uint8_t  gap[192 - 160];
    uintptr_t blk;                /* @192 */
    uint8_t  tail[256];           /* room for whatever else pull_out_buf writes */
} vdec_frame_t;

_Static_assert(sizeof(vdec_path_config_t) == 140, "path config");
_Static_assert(offsetof(vdec_bs_t, phy_addr) == 48 && offsetof(vdec_bs_t, size) == 56 &&
               offsetof(vdec_bs_t, blk) == 64, "bitstream layout");
_Static_assert(offsetof(vdec_frame_t, w) == 24 && offsetof(vdec_frame_t, pw) == 48 &&
               offsetof(vdec_frame_t, loff) == 80 && offsetof(vdec_frame_t, phy_addr) == 96 &&
               offsetof(vdec_frame_t, total_blk_size) == 136 && offsetof(vdec_frame_t, blk) == 192,
               "frame layout");

extern int hd_videodec_init(void);
extern int hd_videodec_uninit(void);
extern int hd_videodec_open(uint32_t in_id, uint32_t out_id, uint32_t *p_path_id);
extern int hd_videodec_close(uint32_t path_id);
extern int hd_videodec_start(uint32_t path_id);
extern int hd_videodec_stop(uint32_t path_id);
extern int hd_videodec_set(uint32_t path_id, uint32_t id, void *p_param);
extern int hd_videodec_push_in_buf(uint32_t path_id, vdec_bs_t *bs, vdec_frame_t *user_out, int32_t wait_ms);
extern int hd_videodec_pull_out_buf(uint32_t path_id, vdec_frame_t *frame, int32_t wait_ms);
extern int hd_videodec_release_out_buf(uint32_t path_id, vdec_frame_t *frame);

/* 64-bit block handles: the same symbols as ai3_abi.h's, with UINTPTR types */
extern uintptr_t hd_mem_get_block_64(uint32_t pool, uint32_t size, uint32_t ddr) __asm__("hd_common_mem_get_block");
extern uintptr_t hd_mem_blk2pa_64(uintptr_t blk) __asm__("hd_common_mem_blk2pa");
extern int hd_mem_release_block_64(uintptr_t blk) __asm__("hd_common_mem_release_block");
#define HD_MEM_INVALID_BLK_64 ((uintptr_t)-1)

#endif
