/* AI3 ABI for NT98539A, from Novatek's own documents in SDK 05.00.260416.01.
 *
 * Sources, per declaration:
 *   hdal           code/hdal/include/hd_common.h, NT9852x board SDK v2.04.002
 *   AI3 structs    Novatek_AI3_Programming_Guide, ch. 7 (declared verbatim)
 *   CFG_BUF        Novatek_NPU_Runtime_Programming_Guide 7.3.2 -- the AI3
 *                  guide lists the type but never declares it; the Runtime
 *                  guide's parameter table names its three members and says
 *                  pa and va "在 64 bits 系统中，要传入有效的 64 bits 地址"
 *                  while size is "数据大小 (bytes)"
 *   call sequence  Novatek_NPU_Runtime_Programming_Guide 6.1.7 (worked
 *                  example) and AI3 Programming Guide ch. 3
 *
 * Two array indices are named in the examples but never given values:
 * AI3_PROC_BUF_RONLYBUF / _WORKBUF and AI3_PLUGIN_CPU. dev_get() writes the
 * required sizes back into proc_mem.buf[], so the probe prints all
 * AI3_PROC_BUF_MAX entries and reads the indices off that -- see ai3_bench.c
 * stage 3. Nothing is guessed into a DMA descriptor.
 */
#ifndef AI3_ABI_H
#define AI3_ABI_H

#include <stdint.h>

typedef int32_t  INT32;
typedef uint32_t UINT32;
typedef uint64_t UINT64;
typedef char     CHAR;
typedef float    FLOAT;
typedef int32_t  HD_RESULT;
typedef UINT32   HD_VIDEO_PXLFMT;

#define HD_OK 0

/* ---- hdal, from hd_common.h ---------------------------------------------- */

#define HD_COMMON_MEM_VB_INVALID_BLK   (-1)
#define HD_COMMON_MEM_SHARED_POOL_NUM  1
/* The header says 128. Over-allocate the config struct instead of trusting
 * this to match the board's 2026 hdal: we only ever fill pool_info[0]. */
#define HD_COMMON_MEM_MAX_POOL_NUM     256

typedef INT32 HD_COMMON_MEM_VB_BLK;

typedef enum { DDR_ID_SYSTEM = 0, DDR_ID0 = 0, DDR_ID1 = 1 } HD_COMMON_MEM_DDR_ID;

typedef enum {
    HD_COMMON_MEM_COMMON_POOL = 1,
} HD_COMMON_MEM_POOL_TYPE;

typedef enum {
    HD_COMMON_MEM_MEM_TYPE_CACHE    = 0,
    HD_COMMON_MEM_MEM_TYPE_NONCACHE = 1,
} HD_COMMON_MEM_MEM_TYPE;

typedef struct {
    HD_COMMON_MEM_DDR_ID    ddr_id;
    HD_COMMON_MEM_POOL_TYPE type;
    UINT32                  blk_size;
    UINT32                  blk_cnt;
    UINT32                  start_addr;
    HD_COMMON_MEM_POOL_TYPE shared_pool[HD_COMMON_MEM_SHARED_POOL_NUM];
} HD_COMMON_MEM_POOL_INFO;

typedef struct {
    HD_COMMON_MEM_POOL_INFO pool_info[HD_COMMON_MEM_MAX_POOL_NUM];
} HD_COMMON_MEM_INIT_CONFIG;

extern HD_RESULT hd_common_init(UINT32 sys_config_type);
extern HD_RESULT hd_common_uninit(void);
extern HD_RESULT hd_common_mem_init(HD_COMMON_MEM_INIT_CONFIG *p_cfg);
extern HD_RESULT hd_common_mem_uninit(void);
extern HD_COMMON_MEM_VB_BLK hd_common_mem_get_block(HD_COMMON_MEM_POOL_TYPE pool_type,
                                                    UINT32 blk_size,
                                                    HD_COMMON_MEM_DDR_ID ddr);
extern HD_RESULT hd_common_mem_release_block(HD_COMMON_MEM_VB_BLK blk);
extern UINT32 hd_common_mem_blk2pa(HD_COMMON_MEM_VB_BLK blk);
/* phy_addr really is UINT32 here, even on the 64-bit part. */
extern void *hd_common_mem_mmap(HD_COMMON_MEM_MEM_TYPE mem_type, UINT32 phy_addr, UINT32 size);
extern HD_RESULT hd_common_mem_munmap(void *virt_addr, unsigned int size);
extern HD_RESULT hd_common_mem_flush_cache(void *virt_addr, unsigned int size);

/* ---- AI3 ----------------------------------------------------------------- */

#define AI3_BUF_INFO_MAX     8
#define AI3_DEV_INFO_MAX     64
#define AI3_DEV_PLUGIN_MAX   8
#define AI3_MODEL_CFG_MAX    16
#define AI3_NET_INFO_MAX     32
#define AI3_PROC_BUF_MAX     8
#define AI3_PROC_CFG_MAX     64

/* Runtime guide 7.3.2: pa and va are 64-bit on a 64-bit part, size is bytes. */
typedef struct _VENDOR_AI3_CFG_BUF {
    uintptr_t pa;
    uintptr_t va;
    UINT32    size;
} VENDOR_AI3_CFG_BUF;

typedef struct _VENDOR_AI3_PROC_MEM {
    VENDOR_AI3_CFG_BUF buf[AI3_PROC_BUF_MAX];
} VENDOR_AI3_PROC_MEM;

typedef struct _VENDOR_AI3_BUF {
    UINT32      sign;
    UINT32      chunk_size;
    uintptr_t   pa;
    uintptr_t   va;
    UINT32      size;
    UINT32      size_real;
    HD_VIDEO_PXLFMT fmt;
    UINT32      width;
    UINT32      height;
    UINT32      channel;
    UINT32      batch_num;
    UINT32      time;
    UINT64      line_ofs;
    UINT64      channel_ofs;
    UINT64      batch_ofs;
    UINT64      time_ofs;
    union { UINT32 reserve; FLOAT scale_ratio; };
    INT32       zero_point;
    CHAR        layout[8];
    CHAR       *name;
    struct _VENDOR_AI3_BUF *p_next;
    UINT32      info[AI3_BUF_INFO_MAX];
} VENDOR_AI3_BUF;

typedef struct _VENDOR_AI3_DEV_CFG {
    UINT32 ctrl;
    UINT32 mode;
    UINT64 info[AI3_DEV_INFO_MAX];
} VENDOR_AI3_DEV_CFG;

typedef struct _VENDOR_AI3_MODEL_INFO {
    VENDOR_AI3_CFG_BUF  model_buf;
    VENDOR_AI3_PROC_MEM proc_mem;
    UINT32              type;
    UINT32              attr;
    UINT32              ctrl;
    UINT32              mode;
    UINT64              config[AI3_MODEL_CFG_MAX];
} VENDOR_AI3_MODEL_INFO;

typedef struct _VENDOR_AI3_PROC_CFG {
    VENDOR_AI3_CFG_BUF  model_buf;
    VENDOR_AI3_PROC_MEM proc_mem;
    UINT32              ctrl;
    UINT32              mode;
    void               *plugin[AI3_DEV_PLUGIN_MAX];
    UINT64              config[AI3_PROC_CFG_MAX];
} VENDOR_AI3_PROC_CFG;

typedef struct _VENDOR_AI3_NET_INFO {
    UINT32  layer_cnt;
    UINT32  bind_cnt;
    UINT32  buf_cnt;
    UINT32  in_buf_cnt;
    UINT32  out_buf_cnt;
    UINT32 *in_path_list;
    UINT32 *out_path_list;
    UINT32  info[AI3_NET_INFO_MAX];
} VENDOR_AI3_NET_INFO;

/* The numbering is the NPU Runtime Programming Guide's VENDOR_NPU_CFG_ID
 * (§7.3.1), not the AI3 guide's listing -- the AI3 guide lists VER2 and
 * TOOLVER right after VER, which would push MODEL_INFO to 3, and that is not
 * what libvendor_ai3.so on NT98539A does. Confirmed against the device
 * (SDK 05.00.260416.01, runtime 01.64.260128.00):
 *
 *    0  VER        returns "01.64.260128.00" and "01.50.251230.00"
 *    1  MODEL_INFO fills proc_mem.buf[0..1].size and leaves model_buf intact
 *    2  IO_CNT     wrote 1 and 1 for a single-input single-output model
 *    3  IO_INFO    refuses without in_buf_info, as documented
 *    7  TOOLVER    returns the model's "05.00.260416.01"
 *
 * The AI3 guide's own listing is abridged in any case: the Runtime Porting
 * Guide gives VENDOR_AI3_CFG_ISP_SHADOW_UB = 25, and id 25 does answer on
 * this build, where the listed order would have placed it at 10.
 */
typedef enum {
    VENDOR_AI3_CFG_VER = 0,
    VENDOR_AI3_CFG_MODEL_INFO,          /* 1 */
    VENDOR_AI3_CFG_IO_CNT,              /* 2 */
    VENDOR_AI3_CFG_IO_INFO,             /* 3 */
    VENDOR_AI3_CFG_TOOLVER = 7,
    VENDOR_AI3_CFG_ISP_SHADOW_UB = 25,
} VENDOR_AI3_CFG_ID;

/* net_set/net_get take either one of these, or a path id straight out of
 * net_info.in_path_list / out_path_list -- Runtime guide 7.5.1:
 *   "net_set(XXXX_PARAM_ID) -> net_set(input_path), net_set(output_path)
 *    -> net_proc()"
 */
typedef enum {
    VENDOR_AI3_NET_PARAM_SHAPE_DIM_IMM = 0,
    VENDOR_AI3_NET_PARAM_BATCH_N_IMM,
    VENDOR_AI3_NET_PARAM_JOB_PRI,
    VENDOR_AI3_NET_PARAM_CORE_MASK,
} VENDOR_AI3_NET_PARAM_ID;

extern HD_RESULT vendor_ai3_dev_init(VENDOR_AI3_DEV_CFG *p_dev_cfg);
extern HD_RESULT vendor_ai3_dev_uninit(void);
extern HD_RESULT vendor_ai3_dev_get(UINT32 cfg_id, void *p_param);
extern HD_RESULT vendor_ai3_net_open(UINT32 *p_proc_id,
                                     VENDOR_AI3_PROC_CFG *p_proc_cfg,
                                     VENDOR_AI3_NET_INFO *p_net_info);
extern HD_RESULT vendor_ai3_net_close(UINT32 proc_id);
extern HD_RESULT vendor_ai3_net_start(UINT32 proc_id);
extern HD_RESULT vendor_ai3_net_stop(UINT32 proc_id);
extern HD_RESULT vendor_ai3_net_get(UINT32 proc_id, UINT32 param_id, void *p_param);
extern HD_RESULT vendor_ai3_net_set(UINT32 proc_id, UINT32 param_id, void *p_param);
extern HD_RESULT vendor_ai3_net_proc(UINT32 proc_id);

/* libvendor_ai3_pub.so; the examples pass this as plugin[AI3_PLUGIN_CPU]. */
extern void *vendor_ai_cpu1_get_engine(void);

#endif /* AI3_ABI_H */
