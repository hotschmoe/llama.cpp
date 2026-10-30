#ifndef HVX_COPY_H
#define HVX_COPY_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx-base.h"

#define hvx_splat_pragma(x) _Pragma(#x)
#define hvx_splat_loop_body(dst_type, vec_store, unroll_cnt)     \
    do {                                                         \
        dst_type * restrict vdst = (dst_type *) dst;             \
                                                                 \
        uint32_t nvec = n / (128 / elem_size);                   \
        uint32_t nloe = n % (128 / elem_size);                   \
                                                                 \
        uint32_t i = 0;                                          \
                                                                 \
        hvx_splat_pragma(unroll(unroll_cnt))                     \
        for (; i < nvec; i++) {                                  \
            vdst[i] = src;                                       \
        }                                                        \
        if (nloe) {                                              \
            vec_store((void *) &vdst[i], nloe * elem_size, src); \
        }                                                        \
    } while(0)

static inline void hvx_splat_a(void * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) dst % 128 == 0);
    hvx_splat_loop_body(HVX_Vector, hvx_vec_store_a, 4);
}

static inline void hvx_splat_u(void * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_size) {
    hvx_splat_loop_body(HVX_UVector, hvx_vec_store_u, 4);
}

static inline void hvx_splat_f32_a(void * restrict dst, float v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f32(v), n, sizeof(float));
}

static inline void hvx_splat_f32_u(void * restrict dst, float v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f32(v), n, sizeof(float));
}

static inline void hvx_splat_f16_a(void * restrict dst, _Float16 v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f16(v), n, sizeof(__fp16));
}

static inline void hvx_splat_f16_u(void * restrict dst, _Float16 v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, sizeof(__fp16));
}

static inline void hvx_splat_u16_a(void * restrict dst, uint16_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vh_vsplat_R(v), n, sizeof(uint16_t));
}

static inline void hvx_splat_u16_u(void * restrict dst, uint16_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vh_vsplat_R(v), n, sizeof(uint16_t));
}

static inline void hvx_splat_u8_a(void * restrict dst, uint8_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

static inline void hvx_splat_u8_u(void * restrict dst, uint8_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

#define hvx_copy_loop_body(dst_type, src_type, vec_store)            \
    do {                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                 \
        src_type * restrict vsrc = (src_type *) src;                 \
                                                                     \
        const uint32_t epv  = 128 / elem_size;                       \
        const uint32_t nvec = n / epv;                               \
        const uint32_t nloe = n % epv;                               \
                                                                     \
        uint32_t i = 0;                                              \
                                                                     \
        _Pragma("unroll(4)")                                         \
        for (; i < nvec; i++) { vdst[i] = vsrc[i]; }                 \
        if (nloe) {                                                  \
            vec_store((void *) &vdst[i], nloe * elem_size, vsrc[i]); \
        }                                                            \
    } while(0)

// Generic copy routines
static inline void hvx_copy_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_copy_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) dst % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_copy_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_copy_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    hvx_copy_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// copy n fp16 elements : destination and source are aligned to HVX Vector (128)
static inline void hvx_copy_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : destination is aligned, source is unaligned
static inline void hvx_copy_f16_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : destination is unaligned, source is aligned
static inline void hvx_copy_f16_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : destination and source are unaligned
static inline void hvx_copy_f16_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, sizeof(__fp16));
}

// copy n fp32 elements : destination and source are aligned to HVX Vector (128)
static inline void hvx_copy_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, sizeof(float));
}

// copy n fp32 elements : destination is aligned, source is unaligned
static inline void hvx_copy_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, sizeof(float));
}

// copy n fp32 elements : destination is unaligned, source is aligned
static inline void hvx_copy_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, sizeof(float));
}

// copy n fp32 elements : destination and source are unaligned
static inline void hvx_copy_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, sizeof(float));
}

//// fp32 -> fp16

#define hvx_copy_f16_f32_loop_body(dst_type, src_type, vec_store)                   \
    do {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        const uint32_t elem_size = sizeof(__fp16);                                  \
        const uint32_t epv  = 128 / elem_size;                                      \
        const uint32_t nvec = n / epv;                                              \
        const uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        for (; i < nvec; i++) {                                                     \
            vdst[i] = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);                 \
        }                                                                           \
        if (nloe) {                                                                 \
            HVX_Vector v = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);            \
            vec_store((void *) &vdst[i], nloe * elem_size, v);                      \
        }                                                                           \
    } while(0)

// copy/convert n fp32 elements into n fp16 elements : destination and source are aligned
static inline void hvx_copy_f16_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is aligned, source is unaligned
static inline void hvx_copy_f16_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is unaligned, source is aligned
static inline void hvx_copy_f16_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : destination and source are unaligned
static inline void hvx_copy_f16_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp16 -> fp32

#define hvx_copy_f32_f16_loop_body(dst_type, src_type, vec_store)                   \
    do {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        const HVX_Vector one = hvx_vec_splat_f16(1.0);                              \
                                                                                    \
        const uint32_t elem_size = sizeof(__fp16);                                  \
        const uint32_t epv  = 128 / elem_size;                                      \
        const uint32_t nvec = n / epv;                                              \
              uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        for (i = 0; i < nvec; ++i) {                                                \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
            vdst[i*2]   = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));                        \
            vdst[i*2+1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));                        \
        }                                                                           \
                                                                                    \
        if (nloe) {                                                                 \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
                                                                                    \
            HVX_Vector vd = Q6_V_lo_W(p);                                           \
            i = 2 * i;                                                              \
                                                                                    \
            if (nloe >= 32) {                                                       \
                vdst[i] = Q6_Vsf_equals_Vqf32(vd);                                  \
                nloe -= 32; ++i; vd = Q6_V_hi_W(p);                                 \
            }                                                                       \
                                                                                    \
            if (nloe) {                                                             \
                vd = Q6_Vsf_equals_Vqf32(vd);                                       \
                hvx_vec_store_u(&vdst[i], nloe * sizeof(float), vd);                \
            }                                                                       \
        }                                                                           \
    } while(0)

// copy/convert n fp16 elements into n fp32 elements : destination and source are aligned
static inline void hvx_copy_f32_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is aligned, source is unaligned
static inline void hvx_copy_f32_f16_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is unaligned, source is aligned
static inline void hvx_copy_f32_f16_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp16 elements into n fp32 elements : destination and source are unaligned
static inline void hvx_copy_f32_f16_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp32 -> int32

#define hvx_copy_i32_f32_loop_body(dst_type, src_type, vec_store) \
    do {                                                          \
        dst_type * restrict vdst = (dst_type *) dst;              \
        src_type * restrict vsrc = (src_type *) src;              \
                                                                  \
        const uint32_t elem_size = sizeof(int32_t);               \
        const uint32_t epv  = 128 / elem_size;                    \
        const uint32_t nvec = n / epv;                            \
        const uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        for (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vw_equals_Vsf(vsrc[i]);                  \
        }                                                         \
        if (nloe) {                                               \
            HVX_Vector v = Q6_Vw_equals_Vsf(vsrc[i]);             \
            vec_store((void *) &vdst[i], nloe * elem_size, v);    \
        }                                                         \
    } while(0)

// copy/convert n fp32 elements into n int32 elements : destination and source are aligned
static inline void hvx_copy_i32_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_i32_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n int32 elements : destination and source are unaligned
static inline void hvx_copy_i32_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_i32_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// int32 -> fp32

#define hvx_copy_f32_i32_loop_body(dst_type, src_type, vec_store) \
    do {                                                          \
        dst_type * restrict vdst = (dst_type *) dst;              \
        src_type * restrict vsrc = (src_type *) src;              \
                                                                  \
        const uint32_t elem_size = sizeof(float);                 \
        const uint32_t epv  = 128 / elem_size;                    \
        const uint32_t nvec = n / epv;                            \
        const uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        for (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vsf_equals_Vw(vsrc[i]);                  \
        }                                                         \
        if (nloe) {                                               \
            HVX_Vector v = Q6_Vsf_equals_Vw(vsrc[i]);             \
            vec_store((void *) &vdst[i], nloe * elem_size, v);    \
        }                                                         \
    } while(0)

// copy/convert n int32 elements into n fp32 elements : destination and source are aligned
static inline void hvx_copy_f32_i32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f32_i32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n int32 elements into n fp32 elements : destination and source are unaligned
static inline void hvx_copy_f32_i32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_f32_i32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp32 <-> bf16 (bit-exact with ggml_compute_fp32_to_bf16 / ggml_compute_bf16_to_fp32)

// Round each fp32 lane to bf16 precision, round-to-nearest-even, NaN forced quiet; the result keeps the
// bf16 value in the upper 16 bits of the lane and zero in the lower 16 (i.e. it is the rounded fp32 value).
// trunc != 0 is the env-gated negative control (truncation instead of RNE); it must never be on by default.
static inline HVX_Vector hvx_vec_round_bf16_f32(HVX_Vector v, bool trunc) {
    const HVX_Vector hi_mask = Q6_V_vsplat_R(0xffff0000u);
    const HVX_Vector c_nan   = Q6_V_vsplat_R(0x7f800000u);
    const HVX_Vector c_qnan  = Q6_V_vsplat_R(0x00400000u);

    HVX_Vector r;
    if (trunc) {
        r = Q6_V_vand_VV(v, hi_mask);
    } else {
        const HVX_Vector one   = Q6_V_vsplat_R(1u);
        const HVX_Vector c7fff = Q6_V_vsplat_R(0x7fffu);
        HVX_Vector lsb  = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(v, 16), one);
        HVX_Vector bias = Q6_Vw_vadd_VwVw(lsb, c7fff);
        r = Q6_V_vand_VV(Q6_Vw_vadd_VwVw(v, bias), hi_mask);
    }

    // NaN: keep the upper half of the payload and set the quiet bit (bf16 (u >> 16) | 64)
    HVX_Vector absv = Q6_V_vand_VV(v, Q6_V_vsplat_R(0x7fffffffu));
    HVX_VectorPred is_nan = Q6_Q_vcmp_gt_VuwVuw(absv, c_nan);
    HVX_Vector nanv = Q6_V_vor_VV(Q6_V_vand_VV(v, hi_mask), c_qnan);
    return Q6_V_vmux_QVV(is_nan, nanv, r);
}

// pack two fp32 vectors (already rounded, bf16 in the upper halves) into one vector of 64 bf16
static inline HVX_Vector hvx_vec_pack_bf16(HVX_Vector r0, HVX_Vector r1) {
    return Q6_Vh_vpacko_VwVw(r1, r0);
}

// widen the first 32 bf16 lanes (lo) / last 32 bf16 lanes (hi) of v to fp32 (exact)
static inline HVX_VectorPair hvx_vec_bf16_to_f32(HVX_Vector v) {
    HVX_VectorPair p = Q6_Ww_vunpack_Vh(v);
    return Q6_W_vcombine_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(p), 16), Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16));
}

// n fp32 -> n bf16, destination and source are aligned. The source buffer must be readable up to a whole
// number of 2 vectors (VTCM rows are 256-byte multiples) ; only n elements are written.
static inline void hvx_copy_bf16_f32_aa_x(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, bool trunc) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    HVX_Vector * restrict       vdst = (HVX_Vector *) dst;
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;

    const uint32_t nvec = n / 64;
    const uint32_t nloe = n % 64;

    uint32_t i = 0;
    for (; i < nvec; i++) {
        vdst[i] = hvx_vec_pack_bf16(hvx_vec_round_bf16_f32(vsrc[i * 2 + 0], trunc),
                                    hvx_vec_round_bf16_f32(vsrc[i * 2 + 1], trunc));
    }
    if (nloe) {
        HVX_Vector v = hvx_vec_pack_bf16(hvx_vec_round_bf16_f32(vsrc[i * 2 + 0], trunc),
                                         hvx_vec_round_bf16_f32(vsrc[i * 2 + 1], trunc));
        hvx_vec_store_a((void *) &vdst[i], nloe * sizeof(uint16_t), v);
    }
}

static inline void hvx_copy_bf16_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_bf16_f32_aa_x(dst, src, n, false);
}
static inline void hvx_copy_bf16_f32_trunc_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_bf16_f32_aa_x(dst, src, n, true);
}

// n bf16 -> n fp32, destination and source are aligned
static inline void hvx_copy_f32_bf16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    HVX_Vector * restrict       vdst = (HVX_Vector *) dst;
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;

    const uint32_t nvec = n / 64;
    uint32_t       nloe = n % 64;

    uint32_t i = 0;
    for (; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_bf16_to_f32(vsrc[i]);
        vdst[i * 2 + 0] = Q6_V_lo_W(p);
        vdst[i * 2 + 1] = Q6_V_hi_W(p);
    }
    if (nloe) {
        HVX_VectorPair p = hvx_vec_bf16_to_f32(vsrc[i]);
        i *= 2;
        if (nloe >= 32) {
            vdst[i++] = Q6_V_lo_W(p);
            nloe -= 32;
            if (nloe) {
                hvx_vec_store_a((void *) &vdst[i], nloe * sizeof(float), Q6_V_hi_W(p));
            }
        } else {
            hvx_vec_store_a((void *) &vdst[i], nloe * sizeof(float), Q6_V_lo_W(p));
        }
    }
}

// n fp32 -> n fp32 rounded to bf16 precision (== f32->bf16->f32), destination and source are aligned
static inline void hvx_round_bf16_f32_aa_x(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, bool trunc) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    HVX_Vector * restrict       vdst = (HVX_Vector *) dst;
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;

    const uint32_t nvec = n / 32;
    const uint32_t nloe = n % 32;

    uint32_t i = 0;
    for (; i < nvec; i++) {
        vdst[i] = hvx_vec_round_bf16_f32(vsrc[i], trunc);
    }
    if (nloe) {
        hvx_vec_store_a((void *) &vdst[i], nloe * sizeof(float), hvx_vec_round_bf16_f32(vsrc[i], trunc));
    }
}

static inline void hvx_round_bf16_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_round_bf16_f32_aa_x(dst, src, n, false);
}
static inline void hvx_round_bf16_f32_trunc_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_round_bf16_f32_aa_x(dst, src, n, true);
}

#endif // HVX_COPY_H
