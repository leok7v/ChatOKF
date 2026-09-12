// Compiled into default.metallib at build time; every kernel has a named
// Swift counterpart it must match (Q2_0.matvec, Kern.rmsnorm, GDN.step).
#include <metal_stdlib>
using namespace metal;


inline float siluf(float x)    { return x / (1.0f + exp(-x)); }
inline float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float softplusf(float x) {
    return max(x, 0.0f) + log(1.0f + exp(-fabs(x)));
}

struct SrqPair {
    float sIn; float loIn; float hiIn; float sOut; float loOut; float hiOut;
};

inline float srq_side(float v, float s, float lo, float hi) {
    return s != 0.0f ? clamp(rint(v / s), -128.0f, 127.0f) * s
                     : (lo < hi ? clamp(v, lo, hi) : v);
}

// woff is 64-bit because the weight buffer is the whole GGUF (>4 GB). The
// two-blocks-in-flight shape is TUNED; do not simplify without re-measuring.
struct GemvArgs { ulong woff; uint K; uint M; };

kernel void q2_0_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint NR0 = 8;
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 128;
    const ushort TPB = 8, SW = 16, STEP = 32 / TPB;   // 4 blocks in flight
    const ulong rowBytes = (ulong) nblk * 34;
    device const uchar * W = weights + a.woff;
    const ushort grp = tiisg / TPB;
    const ushort il  = (tiisg % TPB) * SW;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    // Two blocks per iteration in DISTINCT scalar arrays: a loop-indexed
    // yl[u][i] spills to local memory; UN=2 is the sweet spot, x4 spilled.
    uint ib = grp;
    for (; ib + STEP < nblk; ib += 2 * STEP) {
        const uint ib0 = ib, ib1 = ib + STEP;
        device const float * y0 = x + (ulong) ib0 * 128 + il;
        device const float * y1 = x + (ulong) ib1 * 128 + il;
        float yl0[16], yl1[16];
        float sy0 = 0.0f, sy1 = 0.0f;
        for (ushort i = 0; i < SW; i++) {
            yl0[i] = y0[i]; sy0 += y0[i];
            yl1[i] = y1[i]; sy1 += y1[i];
        }
        for (uint r = 0; r < NR0; r++) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * b0 =
                    W + row * rowBytes + (ulong) ib0 * 34;
                device const uchar * b1 =
                    W + row * rowBytes + (ulong) ib1 * 34;
                const float d0 = (float) (*(device const half *) b0);
                const float d1 = (float) (*(device const half *) b1);
                device const uchar * q0 = b0 + 2 + il / 4;
                device const uchar * q1 = b1 + 2 + il / 4;
                float lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;
                for (ushort i = 0; i < SW; i++) {
                    const uchar c0 = (q0[i >> 2] >> ((i & 3) * 2)) & 3;
                    const uchar c1 = (q1[i >> 2] >> ((i & 3) * 2)) & 3;
                    if (c0 & 1) { lo0 += yl0[i]; }
                    if (c0 & 2) { hi0 += yl0[i]; }
                    if (c1 & 1) { lo1 += yl1[i]; }
                    if (c1 & 2) { hi1 += yl1[i]; }
                }
                acc[r] += d0 * (lo0 + 2.0f * hi0 - sy0)
                        + d1 * (lo1 + 2.0f * hi1 - sy1);
            }
        }
    }
    for (; ib < nblk; ib += STEP) {           // tail (odd block count)
        device const float * y = x + (ulong) ib * 128 + il;
        float yl[16];
        float sy = 0.0f;
        for (ushort i = 0; i < SW; i++) {
            yl[i] = y[i];
            sy += y[i];
        }
        for (uint r = 0; r < NR0; r++) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * bp = W + row * rowBytes + (ulong) ib * 34;
                const float d = (float) (*(device const half *) bp);
                device const uchar * qs = bp + 2 + il / 4;
                float lo = 0.0f, hi = 0.0f;
                for (ushort i = 0; i < SW; i++) {
                    const uchar code = (qs[i >> 2] >> ((i & 3) * 2)) & 3;
                    const float xv = yl[i];
                    if (code & 1) { lo += xv; }
                    if (code & 2) { hi += xv; }
                }
                acc[r] += d * (lo + 2.0f * hi - sy);
            }
        }
    }
    for (uint r = 0; r < NR0; r++) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
    }
}


// out[n*M+m] matches GQ.matvec's column order.
struct block_q2_0 { half d; uchar qs[32]; };
struct block_q4_0 { half d; uchar qs[16]; };
struct block_q8_0 { half d; char qs[32]; };
struct block_iq4_nl { half d; uchar qs[16]; };

// The `_h` twins write half DIRECTLY, not via float. Q4_0 splits a byte into
// element j (LOW nibble) and j+16 (HIGH), so its sub-blocks interleave.
static inline void dq_q2_0_h(device const block_q2_0 * xb, short il,
                             thread half4x4 & reg) {
    device const uchar * qs = xb->qs;
    const half d = xb->d;
    const int bo = il * 4;
    for (int i = 0; i < 4; i++) {
        const uchar b = qs[bo + i];
        reg[i][0] = ((half) ((b >> 0) & 3) - 1.0h) * d;
        reg[i][1] = ((half) ((b >> 2) & 3) - 1.0h) * d;
        reg[i][2] = ((half) ((b >> 4) & 3) - 1.0h) * d;
        reg[i][3] = ((half) ((b >> 6) & 3) - 1.0h) * d;
    }
}


static inline void dq_q4_0_h(device const block_q4_0 * xb, short il,
                             thread half4x4 & reg) {
    device const uchar * qs = xb->qs;
    const half d = xb->d;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            const uchar byte = qs[i * 4 + j];
            const uchar code = il == 0 ? (byte & 0x0F) : (byte >> 4);
            reg[i][j] = ((half) code - 8.0h) * d;
        }
    }
}

static inline void dq_q8_0_h(device const block_q8_0 * xb, short il,
                             thread half4x4 & reg) {
    device const char * qs = xb->qs;
    const half d = xb->d;
    const int bo = il * 16;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            reg[i][j] = (half) qs[bo + i * 4 + j] * d;
        }
    }
}

template <typename Reg>
inline void simd_mm_slice(threadgroup const Reg * sa,
                          threadgroup const Reg * sb,
                          thread simdgroup_float8x8 (&mc)[8],
                          ushort sgitg) {
    simdgroup_matrix<Reg, 8, 8> ma[4], mb[2];
    threadgroup const Reg * lsma = sa + 4 * 64 * (sgitg % 2);
    threadgroup const Reg * lsmb = sb + 2 * 64 * (sgitg / 2);
    #pragma clang loop unroll(full)
    for (short ik = 0; ik < 4; ik++) {        // NK / 8
        simdgroup_barrier(mem_flags::mem_none);
        #pragma clang loop unroll(full)
        for (short i = 0; i < 4; i++) {
            simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
        }
        simdgroup_barrier(mem_flags::mem_none);
        #pragma clang loop unroll(full)
        for (short i = 0; i < 2; i++) {
            simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
        }
        simdgroup_barrier(mem_flags::mem_none);
        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; i++) {
            simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
        }
        lsma += 8 * 64;
        lsmb += 4 * 64;
    }
}

// The spill stages as F32 in the SAME shmem, which is why a partial tile
// needs 8192 B even at half precision.
inline void store_mm_tile(thread simdgroup_float8x8 (&mc)[8],
                          device float * dst, threadgroup uchar * shmem,
                          int r0, int r1, int M, int N,
                          short nr0, short nr1,
                          ushort tiitg, ushort sgitg) {
    const int NR0 = 64, NR1 = 32;
    if (r0 + NR0 <= M && r1 + NR1 <= N) {
        device float * C = dst + (r0 + 32 * (sgitg & 1))
            + (ulong) (r1 + 16 * (sgitg >> 1)) * M;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong) M * (i / 4),
                            M, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp = (threadgroup float *) shmem
            + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp + 8 * (i % 4) + 8 * NR0 * (i / 4),
                            NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float * D = dst + r0 + (ulong) (r1 + j) * M;
                threadgroup float * C = temp + j * NR0;
                for (int i = 0; i < nr0; i++) { D[i] = C[i]; }
            }
        }
    }
}

// Half tiles halve the staging memory and raise occupancy. DQ is a TEMPLATE
// parameter so the call resolves at compile time, not by devirtualization.
template <typename Block, typename Reg, short NSUB, int QK,
          void (*DQ)(device const Block *, short, thread matrix<Reg, 4, 4> &)>
inline void gemm_mm_impl(
        device const uchar * weights,
        device const float * X,
        device       float * dst,
        constant GemvArgs  & a,
        constant uint      & N,
        threadgroup uchar  * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort sgitg) {
    const int K = (int) a.K, M = (int) a.M;
    const int NR0 = 64, NR1 = 32, NK = 32, NL0 = NK / 16, NL1 = NK / 8;
    // The f32 tiles need 8192 for sa where half needs 4096; sb follows it.
    const ulong sbOff = sizeof(Reg) == 2 ? 4096 : 8192;
    threadgroup Reg * sa = (threadgroup Reg *) (shmem);
    threadgroup Reg * sb = (threadgroup Reg *) (shmem + sbOff);
    const int r0 = tgpig.y * NR0;          // first M row of this tile
    const int r1 = tgpig.x * NR1;          // first N col of this tile
    const short nr0 = (M - r0 < NR0) ? (short) (M - r0) : NR0;
    const short nr1 = ((int) N - r1 < NR1) ? (short) ((int) N - r1) : NR1;
    const short lr0 = ((short) tiitg / NL0) < nr0 ? ((short) tiitg / NL0)
                                                  : nr0 - 1;
    const short lr1 = ((short) tiitg / NL1) < nr1 ? ((short) tiitg / NL1)
                                                  : nr1 - 1;
    const short il0 = tiitg % NL0;
    short il = il0;
    const ulong rowBytes = (ulong) (K / QK) * sizeof(Block);
    device const Block * x = (device const Block *)
        (weights + a.woff + rowBytes * (r0 + lr0));
    const short iy = 8 * (tiitg % NL1);
    device const float * y = X + (ulong) (r1 + lr1) * K + iy;
    simdgroup_float8x8 mc[8];
    #pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        matrix<Reg, 4, 4> temp_a;
        DQ(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
        }
        {
            const short sx = tiitg % NL1;
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            threadgroup Reg * bp = sb + 64 * ib + 8 * ly;
            for (short i = 0; i < 8; i++) { bp[i] = (Reg) y[i]; }
        }
        // NSUB == 2 (Q4_0, Q8_0) advances every step; NSUB == 8 (Q2_0) walks
        // four steps inside one 128-weight block first.
        il = (il + 2 < NSUB) ? il + 2 : il % 2;
        x  = (il < 2) ? x + 1 : x;
        y += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simd_mm_slice(sa, sb, mc, sgitg);
    }
    store_mm_tile(mc, dst, shmem, r0, r1, M, (int) N, nr0, nr1,
                  tiitg, sgitg);
}

// A kernel cannot be a template in MSL, so each is a named entry point.
#define GEMM_MM_KERNEL(NAME, BLOCK, REG, NSUB, QK, DQ)                      \
kernel void NAME(                                                           \
        device const uchar * weights [[buffer(0)]],                         \
        device const float * X       [[buffer(1)]],                         \
        device       float * dst     [[buffer(2)]],                         \
        constant GemvArgs  & a       [[buffer(3)]],                         \
        constant uint      & N       [[buffer(4)]],                         \
        threadgroup uchar  * shmem   [[threadgroup(0)]],                    \
        uint3  tgpig [[threadgroup_position_in_grid]],                      \
        ushort tiitg [[thread_index_in_threadgroup]],                       \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {                  \
    gemm_mm_impl<BLOCK, REG, NSUB, QK, DQ>(                                 \
        weights, X, dst, a, N, shmem, tgpig, tiitg, sgitg);                 \
}

GEMM_MM_KERNEL(q2_0_gemm_mm_h, block_q2_0, half,  8, 128, dq_q2_0_h)
GEMM_MM_KERNEL(q4_0_gemm_mm_h, block_q4_0, half,  2,  32, dq_q4_0_h)
GEMM_MM_KERNEL(q8_0_gemm_mm_h, block_q8_0, half,  2,  32, dq_q8_0_h)

// A dense row read as 32-wide blocks, the two 16-halves `il` 0 and 1.
struct block_f16x32 { half v[32]; };
struct block_bf16x32 { ushort v[32]; };
struct block_f32x32 { float v[32]; };

static inline void dq_f16_h(device const block_f16x32 * xb, short il,
                            thread half4x4 & reg) {
    device const half * v = xb->v + il * 16;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) { reg[i][j] = v[i * 4 + j]; }
    }
}

static inline void dq_bf16_h(device const block_bf16x32 * xb, short il,
                             thread half4x4 & reg) {
    device const ushort * v = xb->v + il * 16;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            reg[i][j] = (half) as_type<float>((uint) v[i * 4 + j] << 16);
        }
    }
}

static inline void dq_f32_h(device const block_f32x32 * xb, short il,
                            thread half4x4 & reg) {
    device const float * v = xb->v + il * 16;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) { reg[i][j] = (half) v[i * 4 + j]; }
    }
}

GEMM_MM_KERNEL(f16_gemm_mm_h, block_f16x32, half, 2, 32, dq_f16_h)
GEMM_MM_KERNEL(bf16_gemm_mm_h, block_bf16x32, half, 2, 32, dq_bf16_h)
GEMM_MM_KERNEL(f32_gemm_mm_h, block_f32x32, half, 2, 32, dq_f32_h)

// Generated from ggml-common.h, hence a header. It sits BELOW simd_mm_slice
// and store_mm_tile because the IQ tile shares them.
#include "IQTables.h"


// Ported from LLM/Quantize, which is gated bit-for-bit against ggml. dq_sub
// decodes ONE 32-weight sub-block; gemv, dequant and embed build on it.
#define TY_Q2K     10
#define TY_Q3K     11
#define TY_Q4K     12
#define TY_Q5K     13
#define TY_Q6K     14
#define TY_IQ2XXS  16
#define TY_IQ2XS   17
#define TY_IQ3XXS  18
#define TY_IQ1S    19
#define TY_IQ3S    21
#define TY_IQ2S    22
#define TY_IQ4XS   23
#define TY_IQ1M    29

// A super-block is 50..144 bytes, so nothing inside one is reliably aligned
// for a ushort/uint cast. Every multi-byte field is assembled from bytes.
inline ushort iq_u16(device const uchar *p) {
    return (ushort) p[0] | ((ushort) p[1] << 8);
}

inline uint iq_u32(device const uchar *p) {
    return (uint) p[0] | ((uint) p[1] << 8)
         | ((uint) p[2] << 16) | ((uint) p[3] << 24);
}

inline float iq_f16(device const uchar *p) {
    return (float) as_type<half>(iq_u16(p));
}

inline float iq_sgn(uchar signs, uint j, float v) {
    return (signs & kmask_iq2xs[j]) ? -v : v;
}

inline void dq_sub_iq1s(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b);
    const ushort h = iq_u16(b + 34 + ib * 2);
    const float dl = d * (float) (2 * ((h >> 12) & 7) + 1);
    const float delta = (h & 0x8000) ? -0.125f : 0.125f;
    for (uint l = 0; l < 4; ++l) {
        const uint e = iq1s_grid_gpu[b[2 + ib * 4 + l]
                                     | (((h >> (3 * l)) & 7) << 8)];
        for (uint j = 0; j < 8; ++j) {
            const float n = (float) ((e >> (8 * (j % 4) + 4 * (j / 4))) & 0xF);
            w[l * 8 + j] = dl * (n - 1.0f + delta);
        }
    }
}

inline void dq_sub_iq1m(device const uchar *b, uint ib, thread float *w) {
    ushort sc[4];
    for (uint i = 0; i < 4; ++i) { sc[i] = iq_u16(b + 48 + i * 2); }
    const ushort bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0)
                      | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = (float) as_type<half>(bits);
    const ushort s = sc[ib / 2];
    const uint sh = 6 * (ib % 2);
    const float dl1 = d * (float) (2 * ((s >> sh) & 7) + 1);
    const float dl2 = d * (float) (2 * ((s >> (sh + 3)) & 7) + 1);
    const uchar h0 = b[32 + ib * 2], h1 = b[33 + ib * 2];
    for (uint l = 0; l < 4; ++l) {
        const uchar h = (l < 2) ? h0 : h1;
        const uint up = (l % 2 == 0) ? ((uint) h << 8) : ((uint) h << 4);
        const uint e = iq1s_grid_gpu[b[ib * 4 + l] | (up & 0x700)];
        const float delta = (h & ((l % 2 == 0) ? 0x08 : 0x80))
                          ? -0.125f : 0.125f;
        const float dl = (l < 2) ? dl1 : dl2;
        for (uint j = 0; j < 8; ++j) {
            const float n = (float) ((e >> (8 * (j % 4) + 4 * (j / 4))) & 0xF);
            w[l * 8 + j] = dl * (n - 1.0f + delta);
        }
    }
}

inline void dq_sub_iq2xxs(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b);
    const uint a0 = iq_u32(b + 2 + ib * 8);
    const uint a1 = iq_u32(b + 6 + ib * 8);
    const float db = d * (0.5f + (float) (a1 >> 28)) * 0.25f;
    for (uint l = 0; l < 4; ++l) {
        const uint gi = ((a0 >> (8 * l)) & 0xFF) * 2;
        const uint lo = iq2xxs_grid_u32[gi], hi = iq2xxs_grid_u32[gi + 1];
        const uchar signs = ksigns_iq2xs[(a1 >> (7 * l)) & 127];
        for (uint j = 0; j < 8; ++j) {
            const uint e = (j < 4) ? lo : hi;
            const float g = (float) ((e >> (8 * (j % 4))) & 0xFF);
            w[l * 8 + j] = iq_sgn(signs, j, db * g);
        }
    }
}

inline void dq_sub_iq2xs(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b);
    const uchar sc = b[66 + ib];
    const float db0 = d * (0.5f + (float) (sc & 0xF)) * 0.25f;
    const float db1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
    for (uint l = 0; l < 4; ++l) {
        const ushort q = iq_u16(b + 2 + (ib * 4 + l) * 2);
        const uint gi = (q & 511) * 2;
        const uint lo = iq2xs_grid_u32[gi], hi = iq2xs_grid_u32[gi + 1];
        const uchar signs = ksigns_iq2xs[q >> 9];
        const float dl = (l < 2) ? db0 : db1;
        for (uint j = 0; j < 8; ++j) {
            const uint e = (j < 4) ? lo : hi;
            const float g = (float) ((e >> (8 * (j % 4))) & 0xFF);
            w[l * 8 + j] = iq_sgn(signs, j, dl * g);
        }
    }
}

inline void dq_sub_iq2s(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b);
    const uchar sc = b[74 + ib];
    const uchar qh = b[66 + ib];
    const float db0 = d * (0.5f + (float) (sc & 0xF)) * 0.25f;
    const float db1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
    for (uint l = 0; l < 4; ++l) {
        const uint gi = (b[2 + ib * 4 + l]
                         | ((((uint) qh) << (8 - 2 * l)) & 0x300)) * 2;
        const uint lo = iq2s_grid_u32[gi], hi = iq2s_grid_u32[gi + 1];
        const uchar signs = b[34 + ib * 4 + l];
        const float dl = (l < 2) ? db0 : db1;
        for (uint j = 0; j < 8; ++j) {
            const uint e = (j < 4) ? lo : hi;
            const float g = (float) ((e >> (8 * (j % 4))) & 0xFF);
            w[l * 8 + j] = iq_sgn(signs, j, dl * g);
        }
    }
}

inline void dq_sub_iq3xxs(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b);
    const uint aux = iq_u32(b + 66 + ib * 4);
    const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
    for (uint l = 0; l < 4; ++l) {
        const uchar signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
        const uint g1 = iq3xxs_grid[b[2 + ib * 8 + 2 * l]];
        const uint g2 = iq3xxs_grid[b[2 + ib * 8 + 2 * l + 1]];
        for (uint j = 0; j < 4; ++j) {
            w[l * 8 + j] = iq_sgn(signs, j,
                db * (float) ((g1 >> (8 * j)) & 0xFF));
            w[l * 8 + j + 4] = iq_sgn(signs, j + 4,
                db * (float) ((g2 >> (8 * j)) & 0xFF));
        }
    }
}

inline void dq_sub_iq3s(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b);
    const uchar sc = b[106 + ib / 2];
    const float db = (ib % 2 == 0) ? d * (float) (1 + 2 * (sc & 0xF))
                                   : d * (float) (1 + 2 * (sc >> 4));
    const uchar qh = b[66 + ib];
    for (uint l = 0; l < 4; ++l) {
        const uint base = 2 + ib * 8 + 2 * l;
        const uint g1 = iq3s_grid[b[base]
            | ((((uint) qh) << (8 - 2 * l)) & 256)];
        const uint g2 = iq3s_grid[b[base + 1]
            | ((((uint) qh) << (7 - 2 * l)) & 256)];
        const uchar signs = b[74 + ib * 4 + l];
        for (uint j = 0; j < 4; ++j) {
            w[l * 8 + j] = iq_sgn(signs, j,
                db * (float) ((g1 >> (8 * j)) & 0xFF));
            w[l * 8 + j + 4] = iq_sgn(signs, j + 4,
                db * (float) ((g2 >> (8 * j)) & 0xFF));
        }
    }
}

// Byte 0 first (little endian), so element 4i of a sub-block is word i's low
// byte.
__attribute__((always_inline))
inline float4 nib4(uint word) {
    return float4(as_type<uchar4>(word));
}

inline void put4(thread float *w, uint at, float4 v) {
    w[at] = v.x;
    w[at + 1] = v.y;
    w[at + 2] = v.z;
    w[at + 3] = v.w;
}

// A 136-byte block is 8-byte aligned, so the 16 codes of a sub-block are two
// uint2 loads; the low nibble of byte j is element j and the high one j+16.
inline void dq_sub_iq4xs(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b);
    const ushort sh = iq_u16(b + 2);
    const uchar sl = b[4 + ib / 2];
    const int ls = (int) ((sl >> (4 * (ib % 2))) & 0xF)
                 | (int) (((sh >> (2 * ib)) & 3) << 4);
    const float dl = d * (float) (ls - 32);
    device const uint2 * q2 = (device const uint2 *) (b + 8 + ib * 16);
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 2; ++k) {
        const uint2 v = q2[k];
        #pragma clang loop unroll(full)
        for (uint m = 0; m < 2; ++m) {
            const uint word = m == 0 ? v.x : v.y;
            const uchar4 lo = as_type<uchar4>(word & 0x0F0F0F0Fu);
            const uchar4 hi = as_type<uchar4>((word >> 4) & 0x0F0F0F0Fu);
            const uint at = (k * 2 + m) * 4;
            put4(w, at, dl * float4(kvalues_iq4nl[lo.x], kvalues_iq4nl[lo.y],
                                    kvalues_iq4nl[lo.z], kvalues_iq4nl[lo.w]));
            put4(w, at + 16, dl * float4(kvalues_iq4nl[hi.x],
                                         kvalues_iq4nl[hi.y],
                                         kvalues_iq4nl[hi.z],
                                         kvalues_iq4nl[hi.w]));
        }
    }
}

inline void dq_sub_q2k(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b + 80), dmin = iq_f16(b + 82);
    const uint shift = 2 * (ib % 4);
    const uint qOff = 16 + (ib / 4) * 32;
    for (uint h = 0; h < 2; ++h) {
        const uchar sc = b[ib * 2 + h];
        const float dl = d * (float) (sc & 0xF);
        const float ml = dmin * (float) (sc >> 4);
        for (uint l = 0; l < 16; ++l) {
            const uchar q = b[qOff + h * 16 + l];
            w[h * 16 + l] = dl * (float) ((q >> shift) & 3) - ml;
        }
    }
}

inline void dq_sub_q3k(device const uchar *b, uint ib, thread float *w) {
    const float dAll = iq_f16(b + 108);
    uint aux[4];
    aux[0] = iq_u32(b + 96);
    aux[1] = iq_u32(b + 100);
    aux[2] = iq_u32(b + 104);
    const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    const uint tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const uint shift = 2 * (ib % 4);
    const uint qOff = 32 + (ib / 4) * 32;
    const uchar m = (uchar) (1u << ib);
    for (uint h = 0; h < 2; ++h) {
        const uint isc = ib * 2 + h;
        const int s = (int) (char) ((aux[isc / 4] >> (8 * (isc % 4))) & 0xFF);
        const float dl = dAll * (float) (s - 32);
        for (uint l = 0; l < 16; ++l) {
            const uint at = h * 16 + l;
            const float bump = (b[at] & m) ? 0.0f : 4.0f;
            w[at] = dl * ((float) ((b[qOff + at] >> shift) & 3) - bump);
        }
    }
}

// ggml's get_scale_min_k4 over the twelve packed pairs at byte 4.
inline float2 dq_scale_min(device const uchar *b, uint j) {
    float2 out;
    if (j < 4) {
        out.x = (float) (b[4 + j] & 63);
        out.y = (float) (b[8 + j] & 63);
    } else {
        out.x = (float) ((b[8 + j] & 0xF) | ((b[j] >> 6) << 4));
        out.y = (float) ((b[8 + j] >> 4) | ((b[4 + j] >> 6) << 4));
    }
    return out;
}

// A 144-byte block is 16-byte aligned, so a sub-block's 32 codes are two
// uint4 loads; even and odd sub-blocks share the bytes, low and high nibble.
inline void dq_sub_q4k(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b), dmin = iq_f16(b + 2);
    const float2 sm = dq_scale_min(b, ib);
    const float dv = d * sm.x, ov = dmin * sm.y;
    device const uint4 * q4 = (device const uint4 *) (b + 16 + (ib / 2) * 32);
    const uint sh = 4 * (ib & 1);
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 2; ++k) {
        const uint4 v = q4[k];
        put4(w, k * 16, dv * nib4((v.x >> sh) & 0x0F0F0F0Fu) - ov);
        put4(w, k * 16 + 4, dv * nib4((v.y >> sh) & 0x0F0F0F0Fu) - ov);
        put4(w, k * 16 + 8, dv * nib4((v.z >> sh) & 0x0F0F0F0Fu) - ov);
        put4(w, k * 16 + 12, dv * nib4((v.w >> sh) & 0x0F0F0F0Fu) - ov);
    }
}

// Q4_K's shape plus the fifth bit: bit ib of qh[l] lifts element l by 16.
inline void dq_sub_q5k(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b), dmin = iq_f16(b + 2);
    const float2 sm = dq_scale_min(b, ib);
    const float dv = d * sm.x, ov = dmin * sm.y;
    device const uint4 * q4 = (device const uint4 *) (b + 48 + (ib / 2) * 32);
    device const uint4 * h4 = (device const uint4 *) (b + 16);
    const uint sh = 4 * (ib & 1);
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 2; ++k) {
        const uint4 v = q4[k];
        const uint4 h = h4[k];
        const uint4 n = ((v >> sh) & 0x0F0F0F0Fu)
                      | (((h >> ib) & 0x01010101u) << 4);
        put4(w, k * 16, dv * nib4(n.x) - ov);
        put4(w, k * 16 + 4, dv * nib4(n.y) - ov);
        put4(w, k * 16 + 8, dv * nib4(n.z) - ov);
        put4(w, k * 16 + 12, dv * nib4(n.w) - ov);
    }
}

// A 210-byte block is only 2-byte aligned, so the codes come in as ushort
// pairs; low or high nibble by r, the two high bits from qh.
inline void dq_sub_q6k(device const uchar *b, uint ib, thread float *w) {
    const float d = iq_f16(b + 208);
    const uint n = ib / 4, r = ib % 4;
    device const ushort * ql = (device const ushort *)
        (b + n * 64 + (r % 2) * 32);
    device const ushort * qh = (device const ushort *) (b + 128 + n * 32);
    const uint scOff = 192 + n * 8 + 2 * r;
    const float s0 = d * (float) (int) (char) b[scOff];
    const float s1 = d * (float) (int) (char) b[scOff + 1];
    const uint shq = (r < 2) ? 0 : 4;
    const uint shh = 2 * r;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        const uint lw = (uint) ql[2 * i] | ((uint) ql[2 * i + 1] << 16);
        const uint hw = (uint) qh[2 * i] | ((uint) qh[2 * i + 1] << 16);
        const uint q = ((lw >> shq) & 0x0F0F0F0Fu)
                     | (((hw >> shh) & 0x03030303u) << 4);
        const float s = i < 4 ? s0 : s1;
        put4(w, i * 4, s * (nib4(q) - 32.0f));
    }
}

inline void dq_sub(uint ty, device const uchar *b, uint ib, thread float *w) {
    switch (ty) {
        case TY_IQ1S:   dq_sub_iq1s(b, ib, w);   break;
        case TY_IQ1M:   dq_sub_iq1m(b, ib, w);   break;
        case TY_IQ2XXS: dq_sub_iq2xxs(b, ib, w); break;
        case TY_IQ2XS:  dq_sub_iq2xs(b, ib, w);  break;
        case TY_IQ2S:   dq_sub_iq2s(b, ib, w);   break;
        case TY_IQ3XXS: dq_sub_iq3xxs(b, ib, w); break;
        case TY_IQ3S:   dq_sub_iq3s(b, ib, w);   break;
        case TY_IQ4XS:  dq_sub_iq4xs(b, ib, w);  break;
        case TY_Q2K:    dq_sub_q2k(b, ib, w);    break;
        case TY_Q3K:    dq_sub_q3k(b, ib, w);    break;
        case TY_Q5K:    dq_sub_q5k(b, ib, w);    break;
        case TY_Q6K:    dq_sub_q6k(b, ib, w);    break;
        case TY_Q4K:    dq_sub_q4k(b, ib, w);    break;
        default:
            for (uint j = 0; j < 32; ++j) { w[j] = NAN; }
            break;
    }
}

// A lane's SPAN sub-blocks dotted against R1 staged columns, the scale and
// minimum applied ONCE: Sum(dv*q - ov)*y = dv*Sum(qy) - ov*Sum(y).
__attribute__((always_inline))
inline float hsum(float4 v) { return v.x + v.y + v.z + v.w; }

inline uint kq_byte(uint u, uint i) { return (u >> (8u * i)) & 0xFFu; }

// Both sides three ops, so the select is free of a branch.
__attribute__((always_inline))
inline float2 kq_sm(uint4 h, uint j) {
    const uint k = j & 3u;
    const uint u = kq_byte(h.y, k), v = kq_byte(h.z, k), t = kq_byte(h.w, k);
    const uint sc = (j < 4u) ? (u & 63u) : ((t & 15u) | ((u >> 6) << 4));
    const uint mn = (j < 4u) ? (v & 63u) : ((t >> 4) | ((v >> 6) << 4));
    return float2((float) sc, (float) mn);
}

inline float kq_f16lo(uint u) {
    return (float) as_type<half>((ushort) (u & 0xFFFFu));
}

inline float kq_f16hi(uint u) {
    return (float) as_type<half>((ushort) (u >> 16));
}

// One decoded sub-block as eight float4 with the per-16 scale folded out:
// a column costs eight FMAs and four scalars.
struct KqSub {
    float4 f[8];
    float a0, a1, b0, b1;
};

template <ushort R1, ushort CS, typename Y>
__attribute__((always_inline))
inline void kq_dot(const thread KqSub &w, Y yl, thread const float *sy16,
                   thread float *s) {
    #pragma clang loop unroll(full)
    for (ushort c = 0; c < R1; ++c) {
        Y y = yl + c * CS;
        const float4 t0 = w.f[0] * y[0] + w.f[1] * y[1];
        const float4 u0 = w.f[2] * y[2] + w.f[3] * y[3];
        const float4 t1 = w.f[4] * y[4] + w.f[5] * y[5];
        const float4 u1 = w.f[6] * y[6] + w.f[7] * y[7];
        s[c] = w.a0 * hsum(t0 + u0) + w.a1 * hsum(t1 + u1)
             + w.b0 * sy16[c * 2] + w.b1 * sy16[c * 2 + 1];
    }
}

// Lanes 2j and 2j+1 own sub-blocks 2j and 2j+1, which share one 32-byte
// code span, so each loads half and the pair exchanges over the simdgroup.
__attribute__((always_inline))
inline void kq_dq_q4k(device const uchar *b, uint ib, thread KqSub &w) {
    const uint4 hd = *(device const uint4 *) b;
    const float d = kq_f16lo(hd.x), dmin = kq_f16hi(hd.x);
    const float2 sm = kq_sm(hd, ib);
    device const uint4 * q4 = (device const uint4 *) (b + 16 + (ib / 2) * 32);
    const uint4 mine = q4[ib & 1];
    const uint4 other = simd_shuffle_xor(mine, 1);
    const uint sh = 4 * (ib & 1);
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 2; ++k) {
        const uint4 v = (k == (ib & 1)) ? mine : other;
        const uint4 lo = (v >> sh) & 0x0F0F0F0Fu;
        w.f[k * 4] = nib4(lo.x);
        w.f[k * 4 + 1] = nib4(lo.y);
        w.f[k * 4 + 2] = nib4(lo.z);
        w.f[k * 4 + 3] = nib4(lo.w);
    }
    w.a0 = d * sm.x;
    w.a1 = w.a0;
    w.b0 = -dmin * sm.y;
    w.b1 = w.b0;
}

__attribute__((always_inline))
inline void kq_dq_q5k(device const uchar *b, uint ib, thread KqSub &w) {
    const uint4 hd = *(device const uint4 *) b;
    const float d = kq_f16lo(hd.x), dmin = kq_f16hi(hd.x);
    const float2 sm = kq_sm(hd, ib);
    device const uint4 * q4 = (device const uint4 *) (b + 48 + (ib / 2) * 32);
    device const uint4 * h4 = (device const uint4 *) (b + 16);
    const uint4 mine = q4[ib & 1];
    const uint4 other = simd_shuffle_xor(mine, 1);
    const uint sh = 4 * (ib & 1);
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 2; ++k) {
        const uint4 v = (k == (ib & 1)) ? mine : other;
        const uint4 h = h4[k];
        const uint4 lo = ((v >> sh) & 0x0F0F0F0Fu)
                       | (((h >> ib) & 0x01010101u) << 4);
        w.f[k * 4] = nib4(lo.x);
        w.f[k * 4 + 1] = nib4(lo.y);
        w.f[k * 4 + 2] = nib4(lo.z);
        w.f[k * 4 + 3] = nib4(lo.w);
    }
    w.a0 = d * sm.x;
    w.a1 = w.a0;
    w.b0 = -dmin * sm.y;
    w.b1 = w.b0;
}

__attribute__((always_inline))
inline void kq_dq_q6k(device const uchar *b, uint ib, thread KqSub &w) {
    const float d = iq_f16(b + 208);
    const uint n = ib / 4, r = ib % 4;
    device const packed_ushort4 * ql = (device const packed_ushort4 *)
        (b + n * 64 + (r % 2) * 32);
    device const packed_ushort4 * qh = (device const packed_ushort4 *)
        (b + 128 + n * 32);
    const ushort ss = *(device const ushort *) (b + 192 + n * 8 + 2 * r);
    const float s0 = d * (float) (char) (ss & 0xFFu);
    const float s1 = d * (float) (char) (ss >> 8);
    const uint shq = (r < 2) ? 0 : 4;
    const uint shh = 2 * r;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        const uint2 lw = as_type<uint2>(ushort4(ql[i]));
        const uint2 hw = as_type<uint2>(ushort4(qh[i]));
        w.f[2 * i] = nib4(((lw.x >> shq) & 0x0F0F0F0Fu)
                          | (((hw.x >> shh) & 0x03030303u) << 4));
        w.f[2 * i + 1] = nib4(((lw.y >> shq) & 0x0F0F0F0Fu)
                              | (((hw.y >> shh) & 0x03030303u) << 4));
    }
    w.a0 = s0;
    w.a1 = s1;
    w.b0 = -32.0f * s0;
    w.b1 = -32.0f * s1;
}

__attribute__((always_inline))
inline void kq_dq_iq4xs(device const uchar *b, uint ib, thread KqSub &w) {
    const float d = iq_f16(b);
    const ushort sh = iq_u16(b + 2);
    const uchar sl = b[4 + ib / 2];
    const int ls = (int) ((sl >> (4 * (ib % 2))) & 0xF)
                 | (int) (((sh >> (2 * ib)) & 3) << 4);
    device const uint2 * q2 = (device const uint2 *) (b + 8 + ib * 16);
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 2; ++k) {
        const uint2 v = q2[k];
        #pragma clang loop unroll(full)
        for (uint m = 0; m < 2; ++m) {
            const uint word = m == 0 ? v.x : v.y;
            const uchar4 lo = as_type<uchar4>(word & 0x0F0F0F0Fu);
            const uchar4 hi = as_type<uchar4>((word >> 4) & 0x0F0F0F0Fu);
            const uint at = k * 2 + m;
            w.f[at] = float4(kvalues_iq4nl[lo.x], kvalues_iq4nl[lo.y],
                             kvalues_iq4nl[lo.z], kvalues_iq4nl[lo.w]);
            w.f[at + 4] = float4(kvalues_iq4nl[hi.x], kvalues_iq4nl[hi.y],
                                 kvalues_iq4nl[hi.z], kvalues_iq4nl[hi.w]);
        }
    }
    w.a0 = d * (float) (ls - 32);
    w.a1 = w.a0;
    w.b0 = 0.0f;
    w.b1 = 0.0f;
}

template <ushort F4>
struct KqSlice;

template <typename D, ushort F4>
__attribute__((always_inline))
inline void kq_dq_generic_slice(device const uchar *b, uint ib, uint p,
                                thread KqSlice<F4> &w) {
    float v[32];
    D::dq(b, ib, v);
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < F4; ++j) {
        const uint at = (p * F4 + j) * 4;
        w.f[j] = float4(v[at], v[at + 1], v[at + 2], v[at + 3]);
    }
    w.a = 1.0f;
    w.b = 0.0f;
}

// A decoder with no fused form: materialize, then dot.
template <typename D>
__attribute__((always_inline))
inline void kq_dq_generic(device const uchar *b, uint ib, thread KqSub &w) {
    float v[32];
    D::dq(b, ib, v);
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < 8; ++j) {
        w.f[j] = float4(v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
    }
    w.a0 = 1.0f;
    w.a1 = 1.0f;
    w.b0 = 0.0f;
    w.b1 = 0.0f;
}

// F4 float4 of a sub-block's eight, for a kernel that cannot afford a
// column's 32 activations in registers; the minimum is applied per slice.
template <ushort F4>
struct KqSlice {
    float4 f[F4];
    float a, b;
};

template <ushort F4>
__attribute__((always_inline))
inline void kq_dq_q4k_slice(device const uchar *b, uint ib, uint p,
                            thread KqSlice<F4> &w) {
    const uint4 hd = *(device const uint4 *) b;
    const float d = kq_f16lo(hd.x), dmin = kq_f16hi(hd.x);
    const float2 sm = kq_sm(hd, ib);
    device const uint4 * q4 = (device const uint4 *) (b + 16 + (ib / 2) * 32);
    const uint4 mine = q4[ib & 1];
    const uint4 other = simd_shuffle_xor(mine, 1);
    const uint sh = 4 * (ib & 1);
    const uint h = (p * F4) / 4;
    const uint4 v = (h == (ib & 1)) ? mine : other;
    const uint4 lo = (v >> sh) & 0x0F0F0F0Fu;
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < F4; ++j) { w.f[j] = nib4(lo[(p * F4) % 4 + j]); }
    w.a = d * sm.x;
    w.b = -dmin * sm.y;
}

template <ushort F4>
__attribute__((always_inline))
inline void kq_dq_q5k_slice(device const uchar *b, uint ib, uint p,
                            thread KqSlice<F4> &w) {
    const uint4 hd = *(device const uint4 *) b;
    const float d = kq_f16lo(hd.x), dmin = kq_f16hi(hd.x);
    const float2 sm = kq_sm(hd, ib);
    device const uint4 * q4 = (device const uint4 *) (b + 48 + (ib / 2) * 32);
    device const uint4 * h4 = (device const uint4 *) (b + 16);
    const uint4 mine = q4[ib & 1];
    const uint4 other = simd_shuffle_xor(mine, 1);
    const uint sh = 4 * (ib & 1);
    const uint h = (p * F4) / 4;
    const uint4 v = (h == (ib & 1)) ? mine : other;
    const uint4 hb = h4[h];
    const uint4 lo = ((v >> sh) & 0x0F0F0F0Fu)
                   | (((hb >> ib) & 0x01010101u) << 4);
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < F4; ++j) { w.f[j] = nib4(lo[(p * F4) % 4 + j]); }
    w.a = d * sm.x;
    w.b = -dmin * sm.y;
}

template <ushort F4>
__attribute__((always_inline))
inline void kq_dq_q6k_slice(device const uchar *b, uint ib, uint p,
                            thread KqSlice<F4> &w) {
    const float d = iq_f16(b + 208);
    const uint n = ib / 4, r = ib % 4;
    device const packed_ushort4 * ql = (device const packed_ushort4 *)
        (b + n * 64 + (r % 2) * 32);
    device const packed_ushort4 * qh = (device const packed_ushort4 *)
        (b + 128 + n * 32);
    const uint h = (p * F4) / 4;
    const float sc = d * (float) (char) b[192 + n * 8 + 2 * r + h];
    const uint shq = (r < 2) ? 0 : 4;
    const uint shh = 2 * r;
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < F4; j += 2) {
        const uint i = (p * F4 + j) / 2;
        const uint2 lw = as_type<uint2>(ushort4(ql[i]));
        const uint2 hw = as_type<uint2>(ushort4(qh[i]));
        w.f[j] = nib4(((lw.x >> shq) & 0x0F0F0F0Fu)
                      | (((hw.x >> shh) & 0x03030303u) << 4));
        w.f[j + 1] = nib4(((lw.y >> shq) & 0x0F0F0F0Fu)
                          | (((hw.y >> shh) & 0x03030303u) << 4));
    }
    w.a = sc;
    w.b = -32.0f * sc;
}

template <ushort F4>
__attribute__((always_inline))
inline void kq_dq_iq4xs_slice(device const uchar *b, uint ib, uint p,
                              thread KqSlice<F4> &w) {
    const float d = iq_f16(b);
    const ushort sh = iq_u16(b + 2);
    const uchar sl = b[4 + ib / 2];
    const int ls = (int) ((sl >> (4 * (ib % 2))) & 0xF)
                 | (int) (((sh >> (2 * ib)) & 3) << 4);
    device const uint * q1 = (device const uint *) (b + 8 + ib * 16);
    const uint h = (p * F4) / 4;
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < F4; ++j) {
        const uint at = (p * F4) % 4 + j;
        const uchar4 q = as_type<uchar4>((q1[at] >> (4 * h)) & 0x0F0F0F0Fu);
        w.f[j] = float4(kvalues_iq4nl[q.x], kvalues_iq4nl[q.y],
                        kvalues_iq4nl[q.z], kvalues_iq4nl[q.w]);
    }
    w.a = d * (float) (ls - 32);
    w.b = 0.0f;
}

// Q8_0 and IQ4_NL are 32-weight blocks, eight per 256-weight span, so
// sub-block ib is block ib; a block is 2-byte aligned, codes as packed shorts.
__attribute__((always_inline))
inline uint2 kq_bytes8(device const uchar *p) {
    return as_type<uint2>(ushort4(*(device const packed_ushort4 *) p));
}

__attribute__((always_inline))
inline float4 kq_iq4nl(uint nib) {
    const uchar4 q = as_type<uchar4>(nib);
    return float4(kvalues_iq4nl[q.x], kvalues_iq4nl[q.y],
                  kvalues_iq4nl[q.z], kvalues_iq4nl[q.w]);
}

__attribute__((always_inline))
inline void kq_dq_iq4_nl(device const uchar *b, uint ib, thread KqSub &w) {
    device const uchar * p = b + ib * 18;
    const float d = (float) *(device const half *) p;
    const uint2 lo = kq_bytes8(p + 2), hi = kq_bytes8(p + 10);
    const uint4 wd = uint4(lo.x, lo.y, hi.x, hi.y);
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < 4; ++j) {
        w.f[j] = kq_iq4nl(wd[j] & 0x0F0F0F0Fu);
        w.f[j + 4] = kq_iq4nl((wd[j] >> 4) & 0x0F0F0F0Fu);
    }
    w.a0 = d;
    w.a1 = d;
    w.b0 = 0.0f;
    w.b1 = 0.0f;
}

__attribute__((always_inline))
inline void kq_dq_q8_0(device const uchar *b, uint ib, thread KqSub &w) {
    device const uchar * p = b + ib * 34;
    const float d = (float) *(device const half *) p;
    #pragma clang loop unroll(full)
    for (ushort k = 0; k < 4; ++k) {
        const uint2 q = kq_bytes8(p + 2 + 8 * k);
        w.f[2 * k] = float4(as_type<char4>(q.x));
        w.f[2 * k + 1] = float4(as_type<char4>(q.y));
    }
    w.a0 = d;
    w.a1 = d;
    w.b0 = 0.0f;
    w.b1 = 0.0f;
}

template <ushort F4>
__attribute__((always_inline))
inline void kq_dq_iq4_nl_slice(device const uchar *b, uint ib, uint p,
                               thread KqSlice<F4> &w) {
    device const uchar * bp = b + ib * 18;
    const float d = (float) *(device const half *) bp;
    const uint2 lo = kq_bytes8(bp + 2), hi = kq_bytes8(bp + 10);
    const uint4 wd = uint4(lo.x, lo.y, hi.x, hi.y);
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < F4; ++j) {
        const uint e = p * F4 + j;
        w.f[j] = kq_iq4nl((wd[e % 4] >> (4 * (e / 4))) & 0x0F0F0F0Fu);
    }
    w.a = d;
    w.b = 0.0f;
}

template <ushort F4>
__attribute__((always_inline))
inline void kq_dq_q8_0_slice(device const uchar *b, uint ib, uint p,
                             thread KqSlice<F4> &w) {
    device const uchar * bp = b + ib * 34;
    const float d = (float) *(device const half *) bp;
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < F4; ++j) {
        const uint e = p * F4 + j;
        const ushort2 q = *(device const packed_ushort2 *) (bp + 2 + 4 * e);
        w.f[j] = float4(as_type<char4>(as_type<uint>(q)));
    }
    w.a = d;
    w.b = 0.0f;
}

// Whether a decoder's span can end before a full 256: the packed blocks
// serve rows of any multiple of 32, the super-block types never do.
template <typename D> struct kq_tail {
    static constant constexpr bool value = false;
};

struct DqQ4K {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_q4k_slice<F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_q4k(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_q4k(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqQ5K {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_q5k_slice<F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_q5k(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_q5k(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqQ6K {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_q6k_slice<F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_q6k(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_q6k(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqQ2K {
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_q2k(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqQ2K>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqQ3K {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqQ3K, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_q3k(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqQ3K>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq2xxs {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqIq2xxs, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq2xxs(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqIq2xxs>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq2xs {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqIq2xs, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq2xs(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqIq2xs>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq2s {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqIq2s, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq2s(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqIq2s>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq3xxs {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqIq3xxs, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq3xxs(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqIq3xxs>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq3s {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqIq3s, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq3s(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqIq3s>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq1s {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqIq1s, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq1s(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqIq1s>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq1m {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_generic_slice<DqIq1m, F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq1m(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_generic<DqIq1m>(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq4xs {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_iq4xs_slice<F4>(b, ib, p, w);
    }
    static void dq(device const uchar *b, uint ib, thread float *w) {
        dq_sub_iq4xs(b, ib, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_iq4xs(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};

struct DqQ80 {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_q8_0_slice<F4>(b, ib, p, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_q8_0(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
struct DqIq4nl {
    template <ushort F4>
    __attribute__((always_inline))
    static void dqSlice(device const uchar *b, uint ib, uint p,
                        thread KqSlice<F4> &w) {
        kq_dq_iq4_nl_slice<F4>(b, ib, p, w);
    }
    static constant constexpr ushort span = 1;
    template <ushort R1, ushort SPAN, ushort CS, typename Y>
    __attribute__((always_inline))
    static void dot(device const uchar *b, uint ib, Y yl,
                    thread const float *sy16, thread float *s) {
        KqSub w;
        kq_dq_iq4_nl(b, ib, w);
        kq_dot<R1, CS>(w, yl, sy16, s);
    }
};
template <> struct kq_tail<DqQ80> {
    static constant constexpr bool value = true;
};
template <> struct kq_tail<DqIq4nl> {
    static constant constexpr bool value = true;
};

// `blk` is the byte count of a 256-weight span, handed down from
// GGUF.rowByteCount so the stride has one home.
struct IQArgs { ulong woff; uint K; uint M; uint ty; uint blk; };
struct IQEmbedArgs {
    ulong woff; uint K; uint M; uint rowBytes; uint ty; uint blk;
};

template <typename D>
void iq_dq_row_impl(device const uchar * weights, device float * out,
                    constant IQArgs & a, uint gid) {
    if (gid < a.K / 256) {
        device const uchar * bp = weights + a.woff + (ulong) gid * a.blk;
        device float * o = out + (ulong) gid * 256;
        float w[32];
        for (uint sb = 0; sb < 8; ++sb) {
            D::dq(bp, sb, w);
            for (uint j = 0; j < 32; ++j) { o[sb * 32 + j] = w[j]; }
        }
    }
}

#define IQ_DQROW_KERNEL(NAME, D)                                   \
kernel void NAME(                                                  \
        device const uchar * weights [[buffer(0)]],                \
        device       float * out     [[buffer(1)]],                \
        constant IQArgs    & a       [[buffer(2)]],                \
        uint gid [[thread_position_in_grid]]) {                    \
    iq_dq_row_impl<D>(weights, out, a, gid);                       \
}

IQ_DQROW_KERNEL(q2_k_dequant_row, DqQ2K)
IQ_DQROW_KERNEL(q4_k_dequant_row, DqQ4K)
IQ_DQROW_KERNEL(q6_k_dequant_row, DqQ6K)

template <typename D>
void iq_embed_impl(device const uchar * weights, device const int * ids,
                   device float * out, constant IQEmbedArgs & a, uint gid) {
    const uint nblk = a.K / 256;
    if (gid < a.M * nblk) {
        const uint n = gid / nblk, ib = gid % nblk;
        device const uchar * bp = weights + a.woff
            + (ulong) ids[n] * a.rowBytes + (ulong) ib * a.blk;
        device float * o = out + (ulong) n * a.K + ib * 256;
        float w[32];
        for (uint sb = 0; sb < 8; ++sb) {
            D::dq(bp, sb, w);
            for (uint j = 0; j < 32; ++j) { o[sb * 32 + j] = w[j]; }
        }
    }
}

#define IQ_EMBED_KERNEL(NAME, D)                                   \
kernel void NAME(                                                  \
        device const uchar   * weights [[buffer(0)]],              \
        device const int     * ids     [[buffer(1)]],              \
        device       float   * out     [[buffer(2)]],              \
        constant IQEmbedArgs & a       [[buffer(3)]],              \
        uint gid [[thread_position_in_grid]]) {                    \
    iq_embed_impl<D>(weights, ids, out, a, gid);                   \
}

IQ_EMBED_KERNEL(q2_k_embed_batch, DqQ2K)
IQ_EMBED_KERNEL(q4_k_embed_batch, DqQ4K)
IQ_EMBED_KERNEL(q6_k_embed_batch, DqQ6K)

kernel void iq_dequant_row(
        device const uchar * weights [[buffer(0)]],
        device       float * out     [[buffer(1)]],
        constant IQArgs    & a       [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.K / 256) {
        device const uchar * bp = weights + a.woff + (ulong) gid * a.blk;
        device float * o = out + (ulong) gid * 256;
        float w[32];
        for (uint sb = 0; sb < 8; ++sb) {
            dq_sub(a.ty, bp, sb, w);
            for (uint j = 0; j < 32; ++j) { o[sb * 32 + j] = w[j]; }
        }
    }
}

kernel void iq_embed_batch(
        device const uchar   * weights [[buffer(0)]],
        device const int     * ids     [[buffer(1)]],
        device       float   * out     [[buffer(2)]],
        constant IQEmbedArgs & a       [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const uint nblk = a.K / 256;
    if (gid < a.M * nblk) {
        const uint n = gid / nblk, ib = gid % nblk;
        device const uchar * bp = weights + a.woff
            + (ulong) ids[n] * a.rowBytes + (ulong) ib * a.blk;
        device float * o = out + (ulong) n * a.K + ib * 256;
        float w[32];
        for (uint sb = 0; sb < 8; ++sb) {
            dq_sub(a.ty, bp, sb, w);
            for (uint j = 0; j < 32; ++j) { o[sb * 32 + j] = w[j]; }
        }
    }
}

// The same 64x32x32 tile as gemm_mm_impl with the block walk made runtime:
// `blk` is the stride and `ty` picks the decoder.
template <typename Reg>
inline void iq_gemm_impl(
        device const uchar * weights,
        device const float * X,
        device       float * dst,
        constant IQArgs    & a,
        constant uint      & N,
        threadgroup uchar  * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort sgitg) {
    const int K = (int) a.K, M = (int) a.M;
    const int NR0 = 64, NR1 = 32, NK = 32, NL0 = NK / 16, NL1 = NK / 8;
    const ulong sbOff = sizeof(Reg) == 2 ? 4096 : 8192;
    threadgroup Reg * sa = (threadgroup Reg *) (shmem);
    threadgroup Reg * sb = (threadgroup Reg *) (shmem + sbOff);
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;
    const short nr0 = (M - r0 < NR0) ? (short) (M - r0) : NR0;
    const short nr1 = ((int) N - r1 < NR1) ? (short) ((int) N - r1) : NR1;
    const short lr0 = ((short) tiitg / NL0) < nr0 ? ((short) tiitg / NL0)
                                                  : nr0 - 1;
    const short lr1 = ((short) tiitg / NL1) < nr1 ? ((short) tiitg / NL1)
                                                  : nr1 - 1;
    const short il0 = tiitg % NL0;
    short il = il0;
    const ulong rowBytes = (ulong) (K / 256) * a.blk;
    device const uchar * row = weights + a.woff + rowBytes * (r0 + lr0);
    uint blkIdx = 0;
    const short iy = 8 * (tiitg % NL1);
    device const float * y = X + (ulong) (r1 + lr1) * K + iy;
    simdgroup_float8x8 mc[8];
    #pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    float w[32];
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        matrix<Reg, 4, 4> temp_a;
        dq_sub(a.ty, row + (ulong) blkIdx * a.blk, il / 2, w);
        const short half0 = (il % 2) * 16;
        for (short i = 0; i < 16; i++) {
            temp_a[i / 4][i % 4] = (Reg) w[half0 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short slot = 8 * sx + sy;
            sa[64 * slot + 8 * ly + lx] = temp_a[i / 4][i % 4];
        }
        {
            const short sx = tiitg % NL1;
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short slot = 4 * sx + sy;
            threadgroup Reg * bp = sb + 64 * slot + 8 * ly;
            for (short i = 0; i < 8; i++) { bp[i] = (Reg) y[i]; }
        }
        il = (il + 2 < 16) ? il + 2 : il % 2;
        blkIdx = (il < 2) ? blkIdx + 1 : blkIdx;
        y += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simd_mm_slice(sa, sb, mc, sgitg);
    }
    store_mm_tile(mc, dst, shmem, r0, r1, M, (int) N, nr0, nr1,
                  tiitg, sgitg);
}

kernel void iq_gemm_mm_h(
        device const uchar * weights [[buffer(0)]],
        device const float * X       [[buffer(1)]],
        device       float * dst     [[buffer(2)]],
        constant IQArgs    & a       [[buffer(3)]],
        constant uint      & N       [[buffer(4)]],
        threadgroup uchar  * shmem   [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    iq_gemm_impl<half>(weights, X, dst, a, N, shmem, tgpig, tiitg, sgitg);
}

// iq1_s_gemv's SHAPE without its ternary shortcut: eight threads per block,
// activations staged once per sub-block, ONE decoder inlined by template.
template <typename D>
void kq_gemv_impl(
        device const uchar * weights,
        device const float * x,
        device       float * out,
        constant IQArgs    & a,
        uint3  tgpig,
        ushort tiisg) {
    const ushort SPAN = D::span;
    const ushort L = 8 / SPAN;
    const ushort LANES = L * 2;
    const ushort G = 32 / LANES;
    const ushort RPG = 8 / G;
    const ushort owner = tiisg % L;
    const ushort bg = (tiisg / L) % 2;
    const ushort rg = tiisg / LANES;
    const ushort sub = owner * SPAN;
    const uint nblk = (a.K + 255) / 256;
    const ulong rowBytes = (ulong) a.K * a.blk / 256;
    device const uchar * W = weights + a.woff;
    const uint r0 = tgpig.x * 8 + rg * RPG;
    float acc[4] = { 0, 0, 0, 0 };
    for (uint ib = bg; ib < nblk; ib += 2) {
        const bool live = !kq_tail<D>::value || ib * 256 + sub * 32 < a.K;
        device const float4 * y = (device const float4 *)
            (x + (ulong) ib * 256 + sub * 32);
        float4 yl[8 * SPAN];
        float sy16[2 * SPAN];
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < 8 * SPAN; ++i) { yl[i] = live ? y[i] : 0.0f; }
        #pragma clang loop unroll(full)
        for (ushort h = 0; h < 2 * SPAN; ++h) {
            sy16[h] = hsum(yl[h * 4] + yl[h * 4 + 1]
                           + yl[h * 4 + 2] + yl[h * 4 + 3]);
        }
        #pragma clang loop unroll(full)
        for (ushort rr = 0; rr < RPG; ++rr) {
            const uint row = r0 + rr;
            if (row < a.M && live) {
                device const uchar * bp = W + (ulong) row * rowBytes
                                            + (ulong) ib * a.blk;
                float s;
                D::template dot<1, SPAN, 8>(bp, sub, yl, sy16, &s);
                acc[rr] += s;
            }
        }
    }
    #pragma clang loop unroll(full)
    for (ushort rr = 0; rr < RPG; ++rr) {
        float v = acc[rr];
        #pragma clang loop unroll(full)
        for (ushort sh = 1; sh < LANES; sh <<= 1) {
            v += simd_shuffle_xor(v, sh);
        }
        if (tiisg % LANES == 0 && r0 + rr < a.M) { out[r0 + rr] = v; }
    }
}


#define KQ_GEMV_KERNEL(NAME, D)                                    \
kernel void NAME(                                                  \
        device const uchar * weights [[buffer(0)]],                \
        device const float * x       [[buffer(1)]],                \
        device       float * out     [[buffer(2)]],                \
        constant IQArgs    & a       [[buffer(3)]],                \
        uint3  tgpig [[threadgroup_position_in_grid]],             \
        ushort tiisg [[thread_index_in_simdgroup]]) {              \
    kq_gemv_impl<D>(weights, x, out, a, tgpig, tiisg);             \
}


KQ_GEMV_KERNEL(q4_k_gemv, DqQ4K)
KQ_GEMV_KERNEL(q5_k_gemv, DqQ5K)
KQ_GEMV_KERNEL(q6_k_gemv, DqQ6K)
KQ_GEMV_KERNEL(q2_k_gemv, DqQ2K)
KQ_GEMV_KERNEL(q3_k_gemv, DqQ3K)
KQ_GEMV_KERNEL(iq2_xxs_gemv, DqIq2xxs)
KQ_GEMV_KERNEL(iq2_xs_gemv, DqIq2xs)
KQ_GEMV_KERNEL(iq2_s_gemv, DqIq2s)
KQ_GEMV_KERNEL(iq3_xxs_gemv, DqIq3xxs)
KQ_GEMV_KERNEL(iq3_s_gemv, DqIq3s)
KQ_GEMV_KERNEL(iq4_xs_gemv, DqIq4xs)
KQ_GEMV_KERNEL(q8_0_gemv, DqQ80)
KQ_GEMV_KERNEL(iq4_nl_gemv, DqIq4nl)

// The narrow-batch twin of kq_gemv: each sub-block decoded ONCE and dotted
// against R1 columns, so a verify streams the trunk once, not per column.
template <typename D, ushort R1, ushort SG>
void kq_gemm_nb_impl(
        device const uchar * weights,
        device const float * X,
        device       float * out,
        constant IQArgs    & a,
        threadgroup float4 * ys,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    const ushort LANES = 16;
    const ushort RPG = 4;
    const ushort CS = 128;
    const ushort F4 = R1 >= 4 ? 2 : 4;
    const ushort sub = tiisg % 8;
    const ushort bg = (tiisg / 8) % 2;
    const ushort rg = tiisg / LANES;
    const uint nblk = (a.K + 255) / 256;
    const uint k4 = a.K / 4;
    const ulong rowBytes = (ulong) a.K * a.blk / 256;
    device const uchar * W = weights + a.woff;
    device const float4 * X4 = (device const float4 *) X;
    const uint r0 = tgpig.x * (8 * SG) + sgitg * 8 + rg * RPG;
    float acc[RPG * R1];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < RPG * R1; ++i) { acc[i] = 0.0f; }
    for (uint ib0 = 0; ib0 < nblk; ib0 += 2) {
        for (ushort e = tiitg; e < R1 * CS; e += 32 * SG) {
            const ushort c = e / CS, i = e % CS;
            const uint at = ib0 * 64 + i;
            if (at < k4) { ys[e] = X4[(ulong) c * k4 + at]; }
        }
        if (SG == 1) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const uint ib = ib0 + bg;
        const bool live = kq_tail<D>::value ? ib * 256 + sub * 32 < a.K
                                            : ib < nblk;
        threadgroup const float4 * yl = ys + bg * 64 + sub * 8;
        #pragma clang loop unroll(full)
        for (uint p = 0; p < 8 / F4; ++p) {
            float4 yp[R1 * F4];
            float sy[R1];
            #pragma clang loop unroll(full)
            for (ushort c = 0; c < R1; ++c) {
                float4 t = 0.0f;
                #pragma clang loop unroll(full)
                for (ushort j = 0; j < F4; ++j) {
                    yp[c * F4 + j] = yl[c * CS + p * F4 + j];
                    t += yp[c * F4 + j];
                }
                sy[c] = hsum(t);
            }
            #pragma clang loop unroll(full)
            for (ushort rr = 0; rr < RPG; ++rr) {
                const uint row = r0 + rr;
                if (row < a.M && live) {
                    device const uchar * bp = W + (ulong) row * rowBytes
                                                + (ulong) ib * a.blk;
                    KqSlice<F4> w;
                    D::template dqSlice<F4>(bp, sub, p, w);
                    #pragma clang loop unroll(full)
                    for (ushort c = 0; c < R1; ++c) {
                        float4 t = w.f[0] * yp[c * F4];
                        #pragma clang loop unroll(full)
                        for (ushort j = 1; j < F4; ++j) {
                            t += w.f[j] * yp[c * F4 + j];
                        }
                        acc[rr * R1 + c] += w.a * hsum(t) + w.b * sy[c];
                    }
                }
            }
        }
        if (SG == 1) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < RPG * R1; ++i) {
        float v = acc[i];
        #pragma clang loop unroll(full)
        for (ushort sh = 1; sh < LANES; sh <<= 1) {
            v += simd_shuffle_xor(v, sh);
        }
        const uint row = r0 + i / R1;
        if (tiisg % LANES == 0 && row < a.M) {
            out[(ulong) (i % R1) * a.M + row] = v;
        }
    }
}

// The old Q4_0 kernel's shape: no threadgroup staging, sixteen rows a
// simdgroup, and the row bound a select on the sum rather than a branch.
template <typename D, ushort R1>
void kq_gemm_nw_impl(
        device const uchar * weights,
        device const float * X,
        device       float * out,
        constant IQArgs    & a,
        uint3  tgpig,
        ushort tiisg) {
    const ushort F4 = 2;
    const ushort RPG = 4;
    const ushort sub = tiisg % 8;
    const ushort rg = tiisg / 8;
    const uint nblk = (a.K + 255) / 256;
    const uint k4 = a.K / 4;
    const ulong rowBytes = (ulong) a.K * a.blk / 256;
    device const uchar * W = weights + a.woff;
    device const float4 * X4 = (device const float4 *) X;
    const uint r0 = tgpig.x * 16 + rg * RPG;
    float acc[RPG * R1];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < RPG * R1; ++i) { acc[i] = 0.0f; }
    for (uint ib = 0; ib < nblk; ++ib) {
        const bool live = kq_tail<D>::value ? ib * 256 + sub * 32 < a.K
                                            : true;
        const uint at = ib * 64 + sub * 8;
        #pragma clang loop unroll(full)
        for (uint p = 0; p < 8 / F4; ++p) {
            float4 yp[R1 * F4];
            float sy[R1];
            #pragma clang loop unroll(full)
            for (ushort c = 0; c < R1; ++c) {
                float4 t = 0.0f;
                #pragma clang loop unroll(full)
                for (ushort j = 0; j < F4; ++j) {
                    const uint e = at + p * F4 + j;
                    yp[c * F4 + j] = live ? X4[(ulong) c * k4 + e] : 0.0f;
                    t += yp[c * F4 + j];
                }
                sy[c] = hsum(t);
            }
            #pragma clang loop unroll(full)
            for (ushort rr = 0; rr < RPG; ++rr) {
                const uint row = min(r0 + rr, a.M - 1);
                const float keep = (r0 + rr < a.M && live) ? 1.0f : 0.0f;
                device const uchar * bp = W + (ulong) row * rowBytes
                                            + (ulong) ib * a.blk;
                KqSlice<F4> w;
                D::template dqSlice<F4>(bp, sub, p, w);
                #pragma clang loop unroll(full)
                for (ushort c = 0; c < R1; ++c) {
                    float4 t = w.f[0] * yp[c * F4];
                    #pragma clang loop unroll(full)
                    for (ushort j = 1; j < F4; ++j) {
                        t += w.f[j] * yp[c * F4 + j];
                    }
                    acc[rr * R1 + c] += keep * (w.a * hsum(t) + w.b * sy[c]);
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < RPG * R1; ++i) {
        float v = acc[i];
        v += simd_shuffle_down(v, 4);
        v += simd_shuffle_down(v, 2);
        v += simd_shuffle_down(v, 1);
        const uint row = r0 + i / R1;
        if (sub == 0 && row < a.M) {
            out[(ulong) (i % R1) * a.M + row] = v;
        }
    }
}

#define KQ_NW_KERNEL(NAME, D, R1)                                  \
kernel void NAME(                                                  \
        device const uchar * weights [[buffer(0)]],                \
        device const float * X       [[buffer(1)]],                \
        device       float * out     [[buffer(2)]],                \
        constant IQArgs    & a       [[buffer(3)]],                \
        uint3  tgpig [[threadgroup_position_in_grid]],             \
        ushort tiisg [[thread_index_in_simdgroup]]) {              \
    kq_gemm_nw_impl<D, R1>(weights, X, out, a, tgpig, tiisg);      \
}

KQ_NW_KERNEL(q4_k_gemm_nw_r3, DqQ4K, 3)
KQ_NW_KERNEL(q5_k_gemm_nw_r3, DqQ5K, 3)
KQ_NW_KERNEL(q6_k_gemm_nw_r3, DqQ6K, 3)
KQ_NW_KERNEL(iq4_xs_gemm_nw_r3, DqIq4xs, 3)

#define KQ_NB_KERNEL(NAME, D, R1, SG)                              \
kernel void NAME(                                                  \
        device const uchar * weights [[buffer(0)]],                \
        device const float * X       [[buffer(1)]],                \
        device       float * out     [[buffer(2)]],                \
        constant IQArgs    & a       [[buffer(3)]],                \
        threadgroup float4 * ys      [[threadgroup(0)]],           \
        uint3  tgpig [[threadgroup_position_in_grid]],             \
        ushort tiitg [[thread_index_in_threadgroup]],              \
        ushort tiisg [[thread_index_in_simdgroup]],                \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {         \
    kq_gemm_nb_impl<D, R1, SG>(weights, X, out, a, ys, tgpig,      \
                               tiitg, tiisg, sgitg);               \
}

#define KQ_NB_KERNELS(T, D)                                        \
KQ_NB_KERNEL(T##_gemm_nb_r2s1, D, 2, 1)                            \
KQ_NB_KERNEL(T##_gemm_nb_r2s8, D, 2, 8)                            \
KQ_NB_KERNEL(T##_gemm_nb_r3s1, D, 3, 1)                            \
KQ_NB_KERNEL(T##_gemm_nb_r3s8, D, 3, 8)                            \
KQ_NB_KERNEL(T##_gemm_nb_r4s8, D, 4, 8)                            \
KQ_NB_KERNEL(T##_gemm_nb_r5s8, D, 5, 8)

KQ_NB_KERNELS(q4_k, DqQ4K)
KQ_NB_KERNELS(q5_k, DqQ5K)
KQ_NB_KERNELS(q6_k, DqQ6K)
KQ_NB_KERNELS(iq4_xs, DqIq4xs)
KQ_NB_KERNELS(q8_0, DqQ80)
KQ_NB_KERNELS(iq4_nl, DqIq4nl)
KQ_NB_KERNELS(q3_k, DqQ3K)
KQ_NB_KERNELS(iq2_s, DqIq2s)
KQ_NB_KERNELS(iq3_xxs, DqIq3xxs)
KQ_NB_KERNELS(iq3_s, DqIq3s)
KQ_NB_KERNELS(iq2_xxs, DqIq2xxs)
KQ_NB_KERNELS(iq2_xs, DqIq2xs)
KQ_NB_KERNELS(iq1_s, DqIq1s)
KQ_NB_KERNELS(iq1_m, DqIq1m)

// Outside dq_sub's thirteen-way switch, in q2_0_gemv's shape: the ternary
// grid ({0,1,2} nibbles) collapses to one multiply per sub-block.
kernel void iq1_s_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant IQArgs    & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint NR0 = 8;
    const ushort TPB = 8, STEP = 32 / TPB;
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 256;
    const ulong rowBytes = (ulong) nblk * 50;
    device const uchar * W = weights + a.woff;
    const ushort grp = tiisg / TPB;
    const ushort sub = tiisg % TPB;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (uint ib = grp; ib < nblk; ib += STEP) {
        device const float * y = x + (ulong) ib * 256 + sub * 32;
        float yl[32];
        float sy = 0.0f;
        for (ushort i = 0; i < 32; i++) { yl[i] = y[i]; sy += y[i]; }
        for (uint r = 0; r < NR0; r++) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * bp = W + (ulong) row * rowBytes
                                            + (ulong) ib * 50;
                const float d = iq_f16(bp);
                const ushort h = iq_u16(bp + 34 + sub * 2);
                const float dl = d * (float) (2 * ((h >> 12) & 7) + 1);
                const float delta = (h & 0x8000) ? -0.125f : 0.125f;
                float lo = 0.0f, hi = 0.0f;
                for (ushort l = 0; l < 4; l++) {
                    const uint e = iq1s_grid_gpu[bp[2 + sub * 4 + l]
                        | (((h >> (3 * l)) & 7) << 8)];
                    for (ushort j = 0; j < 8; j++) {
                        const uint n = (e >> (8 * (j % 4) + 4 * (j / 4))) & 0xF;
                        const float v = yl[l * 8 + j];
                        if (n == 1) { lo += v; }
                        if (n == 2) { hi += v; }
                    }
                }
                acc[r] += dl * (lo + 2.0f * hi + (delta - 1.0f) * sy);
            }
        }
    }
    for (uint r = 0; r < NR0; r++) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
    }
}

kernel void iq1_m_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant IQArgs    & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint NR0 = 8;
    const ushort TPB = 8, STEP = 32 / TPB;
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 256;
    const ulong rowBytes = (ulong) nblk * 56;
    device const uchar * W = weights + a.woff;
    const ushort grp = tiisg / TPB;
    const ushort sub = tiisg % TPB;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (uint ib = grp; ib < nblk; ib += STEP) {
        device const float * y = x + (ulong) ib * 256 + sub * 32;
        float yl[32];
        for (ushort i = 0; i < 32; i++) { yl[i] = y[i]; }
        for (uint r = 0; r < NR0; r++) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * bp = W + (ulong) row * rowBytes
                                            + (ulong) ib * 56;
                ushort sc[4];
                for (ushort i = 0; i < 4; i++) { sc[i] = iq_u16(bp + 48 + i * 2); }
                const ushort bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0)
                                  | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
                const float d = (float) as_type<half>(bits);
                const ushort sv = sc[sub / 2];
                const ushort shf = 6 * (sub % 2);
                const float dl1 = d * (float) (2 * ((sv >> shf) & 7) + 1);
                const float dl2 = d * (float) (2 * ((sv >> (shf + 3)) & 7) + 1);
                const uchar h0 = bp[32 + sub * 2], h1 = bp[33 + sub * 2];
                float s = 0.0f;
                for (ushort l = 0; l < 4; l++) {
                    const uchar h = (l < 2) ? h0 : h1;
                    const uint up = (l % 2 == 0) ? ((uint) h << 8)
                                                 : ((uint) h << 4);
                    const uint e = iq1s_grid_gpu[bp[sub * 4 + l] | (up & 0x700)];
                    const float delta = (h & ((l % 2 == 0) ? 0x08 : 0x80))
                                      ? -0.125f : 0.125f;
                    float lo = 0.0f, hi = 0.0f, sy = 0.0f;
                    for (ushort j = 0; j < 8; j++) {
                        const uint n = (e >> (8 * (j % 4) + 4 * (j / 4))) & 0xF;
                        const float v = yl[l * 8 + j];
                        sy += v;
                        if (n == 1) { lo += v; }
                        if (n == 2) { hi += v; }
                    }
                    s += ((l < 2) ? dl1 : dl2)
                       * (lo + 2.0f * hi + (delta - 1.0f) * sy);
                }
                acc[r] += s;
            }
        }
    }
    for (uint r = 0; r < NR0; r++) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
    }
}

struct block_q4_K { uchar b[144]; };
struct block_q5_K { uchar b[176]; };
struct block_q6_K { uchar b[210]; };
struct block_iq4_XS { uchar b[136]; };

// Slice `il` (0..15) of a super-block is sub-block il/2, half il%2.
static inline void dq_q4_k_h(device const block_q4_K * xb, short il,
                             thread half4x4 & reg) {
    device const uchar * b = xb->b;
    const uint ib = il / 2, h = il % 2;
    const uint4 hd = *(device const uint4 *) b;
    const float d = kq_f16lo(hd.x), dmin = kq_f16hi(hd.x);
    const float2 sm = kq_sm(hd, ib);
    const float dv = d * sm.x, ov = dmin * sm.y;
    device const uint4 * q4 = (device const uint4 *) (b + 16 + (ib / 2) * 32);
    const uint4 v = q4[h];
    const uint sh = 4 * (ib & 1);
    const uint4 lo = (v >> sh) & 0x0F0F0F0Fu;
    reg[0] = half4(dv * nib4(lo.x) - ov);
    reg[1] = half4(dv * nib4(lo.y) - ov);
    reg[2] = half4(dv * nib4(lo.z) - ov);
    reg[3] = half4(dv * nib4(lo.w) - ov);
}

static inline void dq_q5_k_h(device const block_q5_K * xb, short il,
                             thread half4x4 & reg) {
    device const uchar * b = xb->b;
    const uint ib = il / 2, h = il % 2;
    const uint4 hd = *(device const uint4 *) b;
    const float d = kq_f16lo(hd.x), dmin = kq_f16hi(hd.x);
    const float2 sm = kq_sm(hd, ib);
    const float dv = d * sm.x, ov = dmin * sm.y;
    device const uint4 * q4 = (device const uint4 *) (b + 48 + (ib / 2) * 32);
    device const uint4 * h4 = (device const uint4 *) (b + 16);
    const uint4 v = q4[h];
    const uint4 hb = h4[h];
    const uint sh = 4 * (ib & 1);
    const uint4 n = ((v >> sh) & 0x0F0F0F0Fu)
                  | (((hb >> ib) & 0x01010101u) << 4);
    reg[0] = half4(dv * nib4(n.x) - ov);
    reg[1] = half4(dv * nib4(n.y) - ov);
    reg[2] = half4(dv * nib4(n.z) - ov);
    reg[3] = half4(dv * nib4(n.w) - ov);
}

static inline void dq_q6_k_h(device const block_q6_K * xb, short il,
                             thread half4x4 & reg) {
    device const uchar * b = xb->b;
    const uint ib = il / 2, h = il % 2;
    const float d = iq_f16(b + 208);
    const uint n = ib / 4, r = ib % 4;
    device const packed_ushort4 * ql = (device const packed_ushort4 *)
        (b + n * 64 + (r % 2) * 32);
    device const packed_ushort4 * qh = (device const packed_ushort4 *)
        (b + 128 + n * 32);
    const float s = d * (float) (int) (char) b[192 + n * 8 + 2 * r + h];
    const uint shq = (r < 2) ? 0 : 4;
    const uint shh = 2 * r;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; ++i) {
        const uint2 lw = as_type<uint2>(ushort4(ql[2 * h + i]));
        const uint2 hw = as_type<uint2>(ushort4(qh[2 * h + i]));
        const uint q0 = ((lw.x >> shq) & 0x0F0F0F0Fu)
                      | (((hw.x >> shh) & 0x03030303u) << 4);
        const uint q1 = ((lw.y >> shq) & 0x0F0F0F0Fu)
                      | (((hw.y >> shh) & 0x03030303u) << 4);
        reg[2 * i] = half4(s * (nib4(q0) - 32.0f));
        reg[2 * i + 1] = half4(s * (nib4(q1) - 32.0f));
    }
}

static inline void dq_iq4_xs_h(device const block_iq4_XS * xb, short il,
                               thread half4x4 & reg) {
    device const uchar * b = xb->b;
    const uint ib = il / 2, h = il % 2;
    const float d = iq_f16(b);
    const ushort sh = iq_u16(b + 2);
    const uchar sl = b[4 + ib / 2];
    const int ls = (int) ((sl >> (4 * (ib % 2))) & 0xF)
                 | (int) (((sh >> (2 * ib)) & 3) << 4);
    const float dl = d * (float) (ls - 32);
    device const uint * q1 = (device const uint *) (b + 8 + ib * 16);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        const uchar4 q = as_type<uchar4>((q1[i] >> (4 * h)) & 0x0F0F0F0Fu);
        reg[i] = half4(dl * float4(kvalues_iq4nl[q.x], kvalues_iq4nl[q.y],
                                   kvalues_iq4nl[q.z], kvalues_iq4nl[q.w]));
    }
}

GEMM_MM_KERNEL(q4_k_gemm_mm_h, block_q4_K, half, 16, 256, dq_q4_k_h)
GEMM_MM_KERNEL(q5_k_gemm_mm_h, block_q5_K, half, 16, 256, dq_q5_k_h)
GEMM_MM_KERNEL(q6_k_gemm_mm_h, block_q6_K, half, 16, 256, dq_q6_k_h)
GEMM_MM_KERNEL(iq4_xs_gemm_mm_h, block_iq4_XS, half, 16, 256, dq_iq4_xs_h)

// The codebook types decode a whole sub-block and keep the staged half.
template <typename D, typename Block>
static inline void dq_generic_h(device const Block * xb, short il,
                                thread half4x4 & reg) {
    float v[32];
    D::dq(xb->b, il / 2, v);
    const short at = (il % 2) * 16;
    for (int i = 0; i < 4; i++) {
        reg[i] = half4(v[at + i * 4], v[at + i * 4 + 1], v[at + i * 4 + 2],
                       v[at + i * 4 + 3]);
    }
}

struct block_q3_K { uchar b[110]; };
struct block_iq2_S { uchar b[82]; };
struct block_iq3_XXS { uchar b[98]; };
struct block_iq3_S { uchar b[110]; };
struct block_iq2_XXS { uchar b[66]; };
struct block_iq2_XS { uchar b[74]; };
struct block_iq1_S { uchar b[50]; };
struct block_iq1_M { uchar b[56]; };

static inline void dq_q3_k_h(device const block_q3_K * xb, short il,
                             thread half4x4 & reg) {
    dq_generic_h<DqQ3K, block_q3_K>(xb, il, reg);
}
static inline void dq_iq2_s_h(device const block_iq2_S * xb, short il,
                              thread half4x4 & reg) {
    dq_generic_h<DqIq2s, block_iq2_S>(xb, il, reg);
}
static inline void dq_iq3_xxs_h(device const block_iq3_XXS * xb, short il,
                                thread half4x4 & reg) {
    dq_generic_h<DqIq3xxs, block_iq3_XXS>(xb, il, reg);
}
static inline void dq_iq3_s_h(device const block_iq3_S * xb, short il,
                              thread half4x4 & reg) {
    dq_generic_h<DqIq3s, block_iq3_S>(xb, il, reg);
}
static inline void dq_iq2_xxs_h(device const block_iq2_XXS * xb, short il,
                                thread half4x4 & reg) {
    dq_generic_h<DqIq2xxs, block_iq2_XXS>(xb, il, reg);
}
static inline void dq_iq2_xs_h(device const block_iq2_XS * xb, short il,
                               thread half4x4 & reg) {
    dq_generic_h<DqIq2xs, block_iq2_XS>(xb, il, reg);
}
static inline void dq_iq1_s_h(device const block_iq1_S * xb, short il,
                              thread half4x4 & reg) {
    dq_generic_h<DqIq1s, block_iq1_S>(xb, il, reg);
}
static inline void dq_iq1_m_h(device const block_iq1_M * xb, short il,
                              thread half4x4 & reg) {
    dq_generic_h<DqIq1m, block_iq1_M>(xb, il, reg);
}

GEMM_MM_KERNEL(q3_k_gemm_mm_h, block_q3_K, half, 16, 256, dq_q3_k_h)
GEMM_MM_KERNEL(iq2_s_gemm_mm_h, block_iq2_S, half, 16, 256, dq_iq2_s_h)
GEMM_MM_KERNEL(iq3_xxs_gemm_mm_h, block_iq3_XXS, half, 16, 256, dq_iq3_xxs_h)
GEMM_MM_KERNEL(iq3_s_gemm_mm_h, block_iq3_S, half, 16, 256, dq_iq3_s_h)
GEMM_MM_KERNEL(iq2_xxs_gemm_mm_h, block_iq2_XXS, half, 16, 256, dq_iq2_xxs_h)
GEMM_MM_KERNEL(iq2_xs_gemm_mm_h, block_iq2_XS, half, 16, 256, dq_iq2_xs_h)
GEMM_MM_KERNEL(iq1_s_gemm_mm_h, block_iq1_S, half, 16, 256, dq_iq1_s_h)
GEMM_MM_KERNEL(iq1_m_gemm_mm_h, block_iq1_M, half, 16, 256, dq_iq1_m_h)


// The token table stays in ORIGINAL order: `token_ordering` VALUES are token
// ids, so the gather is scattered and no inverse permutation applies.
struct AssistTopArgs { uint n; uint k; };

// n is small (2048), so staging costs 8 KB and every pass is one reduction.
kernel void assist_top_clusters(
        device const float    * x   [[buffer(0)]],
        device       uint     * out [[buffer(1)]],
        constant AssistTopArgs & a  [[buffer(2)]],
        uint tpitg [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    threadgroup float s[2048];
    threadgroup float bv[32];
    threadgroup uint  bi[32];
    for (uint i = tpitg; i < a.n; i += ntg) { s[i] = x[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint pass = 0; pass < a.k; pass++) {
        float best = -INFINITY;
        uint  at = 0;
        for (uint i = tpitg; i < a.n; i += ntg) {
            if (s[i] > best) { best = s[i]; at = i; }
        }
        const uint lane = tpitg % 32;
        const uint warp = tpitg / 32;
        for (uint off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(best, off);
            const uint  oi = simd_shuffle_down(at, off);
            if (ov > best || (ov == best && oi < at)) { best = ov; at = oi; }
        }
        if (lane == 0) { bv[warp] = best; bi[warp] = at; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tpitg == 0) {
            float tv = bv[0];
            uint  ti = bi[0];
            for (uint w = 1; w < ntg / 32; w++) {
                if (bv[w] > tv || (bv[w] == tv && bi[w] < ti)) {
                    tv = bv[w];
                    ti = bi[w];
                }
            }
            out[pass] = ti;
            s[ti] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

struct AssistPickArgs {
    ulong woff;
    ulong ooff;
    uint  dim;
    uint  per;
    uint  clusters;
    uint  rowBytes;
};

// The same Q4_0 layout q4_0_dequant_row reads: low nibble j, high j+16.
inline float assist_row_dot(device const uchar * row, device const float * h,
                            uint dim) {
    float acc = 0.0f;
    const uint blocks = dim / 32;
    for (uint b = 0; b < blocks; b++) {
        device const uchar * bp = row + b * 18;
        const float d = (float) (*(device const half *) bp);
        device const uchar * qs = bp + 2;
        const uint base = b * 32;
        float s = 0.0f;
        for (uint i = 0; i < 16; i++) {
            const uchar byte = qs[i];
            s += ((float) (byte & 0x0F) - 8.0f) * h[base + i];
            s += ((float) (byte >> 4) - 8.0f) * h[base + i + 16];
        }
        acc += s * d;
    }
    return acc;
}

kernel void assist_cluster_argmax(
        device const uchar     * weights  [[buffer(0)]],
        device const float     * h        [[buffer(1)]],
        device const uchar     * obase    [[buffer(2)]],
        device const uint      * clusters [[buffer(3)]],
        device       int       * out      [[buffer(4)]],
        constant AssistPickArgs & a       [[buffer(5)]],
        uint tpitg [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    threadgroup float bv[32];
    threadgroup int   bi[32];
    device const uchar * table = weights + a.woff;
    device const float * ordering =
        (device const float *) (obase + a.ooff);
    const uint total = a.clusters * a.per;
    float best = -INFINITY;
    int   at = 0;
    for (uint i = tpitg; i < total; i += ntg) {
        const uint ordered = clusters[i / a.per] * a.per + (i % a.per);
        const int token = (int) rint(ordering[ordered]);
        const float score = assist_row_dot(
            table + (ulong) token * a.rowBytes, h, a.dim);
        if (score > best || (score == best && token < at)) {
            best = score;
            at = token;
        }
    }
    const uint lane = tpitg % 32;
    const uint warp = tpitg / 32;
    for (uint off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(best, off);
        const int   oi = simd_shuffle_down(at, off);
        if (ov > best || (ov == best && oi < at)) { best = ov; at = oi; }
    }
    if (lane == 0) { bv[warp] = best; bi[warp] = at; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tpitg == 0) {
        float tv = bv[0];
        int   ti = bi[0];
        for (uint w = 1; w < ntg / 32; w++) {
            if (bv[w] > tv || (bv[w] == tv && bi[w] < ti)) {
                tv = bv[w];
                ti = bi[w];
            }
        }
        out[0] = ti;
    }
}

static inline void dq_iq4_nl_h(device const block_iq4_nl * xb, short il,
                               thread half4x4 & reg) {
    device const uchar * qs = xb->qs;
    const half d = xb->d;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            const uchar byte = qs[i * 4 + j];
            const uchar code = il == 0 ? (byte & 0x0F) : (byte >> 4);
            reg[i][j] = (half) kvalues_iq4nl[code] * d;
        }
    }
}

GEMM_MM_KERNEL(iq4_nl_gemm_mm_h, block_iq4_nl, half,  2,  32, dq_iq4_nl_h)

// `woff` already points at the wanted row's first block.
kernel void q2_0_dequant_row(
        device const uchar * weights [[buffer(0)]],
        device       float * out     [[buffer(1)]],
        constant GemvArgs  & a       [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.K) {
        const uint ib = gid / 128;
        const uint j  = gid % 128;
        device const uchar * bp = weights + a.woff + (ulong) ib * 34;
        const float d = (float) (*(device const half *) bp);
        const uchar code = (bp[2 + (j >> 2)] >> ((j & 3) * 2)) & 3;
        out[gid] = (float) ((int) code - 1) * d;
    }
}


// The same codebook and offset gemma's INT4 QAT uses, which is what makes the
// repack a code-for-code transfer. Mirrors GQ.matvec's q4_0 arm, the oracle.
template <bool S>
inline void q4_0_gemv_impl(
        device const uchar * weights,
        device const float * x,
        device       float * out,
        constant GemvArgs  & a,
        SrqPair c,
        uint3  tgpig,
        ushort tiisg) {
    const uint NR0 = 8;
    const uint row0 = tgpig.x * NR0;
    const uint nblk = a.K / 32;
    const ulong rowBytes = (ulong) nblk * 18;
    device const uchar * W = weights + a.woff;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (uint ib = tiisg; ib < nblk; ib += 32) {
        device const float * y = x + (ulong) ib * 32;
        float yl[16], yh[16];
        for (ushort i = 0; i < 16; i++) {
            yl[i] = S ? srq_side(y[i], c.sIn, c.loIn, c.hiIn) : y[i];
            yh[i] = S ? srq_side(y[i + 16], c.sIn, c.loIn, c.hiIn)
                      : y[i + 16];
        }
        for (uint r = 0; r < NR0; r++) {
            const uint row = row0 + r;
            if (row < a.M) {
                device const uchar * bp = W + row * rowBytes + (ulong) ib * 18;
                const float d = (float) (*(device const half *) bp);
                device const uchar * qs = bp + 2;
                float s = 0.0f;
                for (ushort i = 0; i < 16; i++) {
                    const uchar b = qs[i];
                    s += ((float) (b & 0x0F) - 8.0f) * yl[i];
                    s += ((float) (b >> 4) - 8.0f) * yh[i];
                }
                acc[r] += s * d;
            }
        }
    }
    for (uint r = 0; r < NR0; r++) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) {
            out[row0 + r] = S ? srq_side(s, c.sOut, c.loOut, c.hiOut) : s;
        }
    }
}

kernel void q4_0_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const SrqPair none = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    q4_0_gemv_impl<false>(weights, x, out, a, none, tgpig, tiisg);
}

kernel void q4_0_gemv_srq(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        constant SrqPair   & c       [[buffer(4)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const SrqPair pair = c;
    q4_0_gemv_impl<true>(weights, x, out, a, pair, tgpig, tiisg);
}

// The narrow-batch twin of q4_0_gemv, for the TIED lm_head a verify sweeps.
// A block is walked in four 4-byte groups so activations stay in registers.
template<ushort R1, ushort NR0>
void q4_0_gemm_nb_impl(
        device const uchar * weights,
        device const float * X,
        device       float * out,
        constant GemvArgs  & a,
        uint3  tgpig,
        ushort tiisg) {
    const ushort NX = 8;
    const ushort NY = 32 / NX;
    const ushort tx = tiisg % NX;
    const ushort ty = tiisg / NX;
    const uint row0 = (tgpig.x * NY + ty) * NR0;
    const uint nblk = a.K / 32;
    const ulong rowBytes = (ulong) nblk * 18;
    device const uchar * W = weights + a.woff;
    float acc[NR0 * R1];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < NR0 * R1; i++) { acc[i] = 0.0f; }
    for (uint ib = tx; ib < nblk; ib += NX) {
        float dd[NR0];
        device const uchar * qs[NR0];
        // A row past M reads row M-1 with a zero scale, never out of bounds.
        #pragma clang loop unroll(full)
        for (ushort r = 0; r < NR0; r++) {
            device const uchar * bp = W
                + min(row0 + r, a.M - 1) * rowBytes + (ulong) ib * 18;
            dd[r] = (row0 + r < a.M)
                ? (float) (*(device const half *) bp) : 0.0f;
            qs[r] = bp + 2;
        }
        #pragma clang loop unroll(full)
        for (ushort g = 0; g < 4; g++) {
            float4 ylo[R1], yhi[R1];
            #pragma clang loop unroll(full)
            for (ushort c = 0; c < R1; c++) {
                device const float4 * y = (device const float4 *)
                    (X + (ulong) c * a.K + (ulong) ib * 32);
                ylo[c] = y[g];
                yhi[c] = y[4 + g];
            }
            #pragma clang loop unroll(full)
            for (ushort r = 0; r < NR0; r++) {
                float4 lo, hi;
                #pragma clang loop unroll(full)
                for (ushort i = 0; i < 4; i++) {
                    const uchar b = qs[r][g * 4 + i];
                    lo[i] = (float) (b & 0x0F) - 8.0f;
                    hi[i] = (float) (b >> 4) - 8.0f;
                }
                #pragma clang loop unroll(full)
                for (ushort c = 0; c < R1; c++) {
                    acc[r * R1 + c] += dd[r]
                        * (dot(lo, ylo[c]) + dot(hi, yhi[c]));
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < NR0 * R1; i++) {
        acc[i] += simd_shuffle_down(acc[i], 4);
        acc[i] += simd_shuffle_down(acc[i], 2);
        acc[i] += simd_shuffle_down(acc[i], 1);
    }
    if (tx == 0) {
        for (ushort r = 0; r < NR0; r++) {
            if (row0 + r < a.M) {
                for (ushort c = 0; c < R1; c++) {
                    out[(ulong) c * a.M + row0 + r] = acc[r * R1 + c];
                }
            }
        }
    }
}

#define Q40_NB_KERNEL(NAME, R1, NR0)                                   \
kernel void NAME(                                                      \
        device const uchar * weights [[buffer(0)]],                    \
        device const float * X       [[buffer(1)]],                    \
        device       float * out     [[buffer(2)]],                    \
        constant GemvArgs  & a       [[buffer(3)]],                    \
        uint3  tgpig [[threadgroup_position_in_grid]],                 \
        ushort tiisg [[thread_index_in_simdgroup]]) {                  \
    q4_0_gemm_nb_impl<R1, NR0>(weights, X, out, a, tgpig, tiisg);      \
}

Q40_NB_KERNEL(q4_0_gemm_nb_r2, 2, 4)
Q40_NB_KERNEL(q4_0_gemm_nb_r3, 3, 4)
Q40_NB_KERNEL(q4_0_gemm_nb_r4, 4, 4)
Q40_NB_KERNEL(q4_0_gemm_nb_r5, 5, 4)

// One weight read serves R1 columns: prefill without matrix units is
// otherwise one weight stream per token. A column stages a quarter block.
template<ushort R1, ushort NR0>
void q2_0_gemm_nb_impl(
        device const uchar * weights,
        device const float * X,
        device       float * out,
        constant GemvArgs  & a,
        uint3  tgpig,
        ushort tiisg) {
    const ushort NX = 8;
    const ushort NY = 32 / NX;
    const ushort tx = tiisg % NX;
    const ushort ty = tiisg / NX;
    const uint row0 = (tgpig.x * NY + ty) * NR0;
    const uint nblk = a.K / 128;
    const ulong rowBytes = (ulong) nblk * 34;
    device const uchar * W = weights + a.woff;
    float acc[NR0 * R1];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < NR0 * R1; i++) { acc[i] = 0.0f; }
    for (uint ib = tx; ib < nblk; ib += NX) {
        float dd[NR0];
        device const uchar * qs[NR0];
        // A row past M reads row M-1 with a zero scale, never out of bounds.
        #pragma clang loop unroll(full)
        for (ushort r = 0; r < NR0; r++) {
            device const uchar * bp = W
                + min(row0 + r, a.M - 1) * rowBytes + (ulong) ib * 34;
            dd[r] = (row0 + r < a.M)
                ? (float) (*(device const half *) bp) : 0.0f;
            qs[r] = bp + 2;
        }
        float sy[R1];
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < R1; c++) { sy[c] = 0.0f; }
        for (ushort g = 0; g < 32; g++) {
            float4 yv[R1];
            #pragma clang loop unroll(full)
            for (ushort c = 0; c < R1; c++) {
                device const float4 * y = (device const float4 *)
                    (X + (ulong) c * a.K + (ulong) ib * 128);
                yv[c] = y[g];
                sy[c] += yv[c].x + yv[c].y + yv[c].z + yv[c].w;
            }
            #pragma clang loop unroll(full)
            for (ushort r = 0; r < NR0; r++) {
                const uchar b = qs[r][g];
                float4 lo, hi;
                #pragma clang loop unroll(full)
                for (ushort i = 0; i < 4; i++) {
                    const uchar code = (b >> (i * 2)) & 3;
                    lo[i] = (code & 1) ? 1.0f : 0.0f;
                    hi[i] = (code & 2) ? 1.0f : 0.0f;
                }
                #pragma clang loop unroll(full)
                for (ushort c = 0; c < R1; c++) {
                    acc[r * R1 + c] += dd[r]
                        * (dot(lo, yv[c]) + 2.0f * dot(hi, yv[c]));
                }
            }
        }
        #pragma clang loop unroll(full)
        for (ushort r = 0; r < NR0; r++) {
            #pragma clang loop unroll(full)
            for (ushort c = 0; c < R1; c++) {
                acc[r * R1 + c] -= dd[r] * sy[c];
            }
        }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < NR0 * R1; i++) {
        acc[i] += simd_shuffle_down(acc[i], 4);
        acc[i] += simd_shuffle_down(acc[i], 2);
        acc[i] += simd_shuffle_down(acc[i], 1);
    }
    if (tx == 0) {
        for (ushort r = 0; r < NR0; r++) {
            if (row0 + r < a.M) {
                for (ushort c = 0; c < R1; c++) {
                    out[(ulong) c * a.M + row0 + r] = acc[r * R1 + c];
                }
            }
        }
    }
}

#define Q20_NB_KERNEL(NAME, R1, NR0)                                   \
kernel void NAME(                                                      \
        device const uchar * weights [[buffer(0)]],                    \
        device const float * X       [[buffer(1)]],                    \
        device       float * out     [[buffer(2)]],                    \
        constant GemvArgs  & a       [[buffer(3)]],                    \
        uint3  tgpig [[threadgroup_position_in_grid]],                 \
        ushort tiisg [[thread_index_in_simdgroup]]) {                  \
    q2_0_gemm_nb_impl<R1, NR0>(weights, X, out, a, tgpig, tiisg);      \
}

Q20_NB_KERNEL(q2_0_gemm_nb_r2, 2, 4)
Q20_NB_KERNEL(q2_0_gemm_nb_r3, 3, 4)
Q20_NB_KERNEL(q2_0_gemm_nb_r4, 4, 4)
Q20_NB_KERNEL(q2_0_gemm_nb_r5, 5, 4)


kernel void q4_0_dequant_row(
        device const uchar * weights [[buffer(0)]],
        device       float * out     [[buffer(1)]],
        constant GemvArgs  & a       [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.K) {
        const uint ib = gid / 32;
        const uint j  = gid % 32;
        device const uchar * bp = weights + a.woff + (ulong) ib * 18;
        const float d = (float) (*(device const half *) bp);
        const uchar b = bp[2 + (j % 16)];
        const uchar code = (j < 16) ? (b & 0x0F) : (b >> 4);
        out[gid] = ((float) code - 8.0f) * d;
    }
}

// BF16 is the top 16 bits of f32. Every gemma norm is stored so, and read
// through the f32 path two weights fuse into one garbage float.
inline float bf16_at(device const uchar * p, uint i) {
    const ushort bits = *(device const ushort *) (p + 2 * i);
    return as_type<float>((uint) bits << 16);
}

kernel void q8_0_dequant_row(
        device const uchar * weights [[buffer(0)]],
        device       float * out     [[buffer(1)]],
        constant GemvArgs  & a       [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.K) {
        const uint ib = gid / 32;
        const uint j  = gid % 32;
        device const uchar * bp = weights + a.woff + (ulong) ib * 34;
        const float d = (float) (*(device const half *) bp);
        const char q = ((device const char *) (bp + 2))[j];
        out[gid] = (float) q * d;
    }
}

kernel void bf16_dequant_row(
        device const uchar * weights [[buffer(0)]],
        device       float * out     [[buffer(1)]],
        constant GemvArgs  & a       [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.K) {
        out[gid] = bf16_at(weights + a.woff, gid);
    }
}

kernel void f16_dequant_row(
        device const uchar * weights [[buffer(0)]],
        device       float * out     [[buffer(1)]],
        constant GemvArgs  & a       [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.K) {
        out[gid] = (float) ((device const half *) (weights + a.woff))[gid];
    }
}

kernel void f32_dequant_row(
        device const uchar * weights [[buffer(0)]],
        device       float * out     [[buffer(1)]],
        constant GemvArgs  & a       [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.K) {
        out[gid] = ((device const float *) (weights + a.woff))[gid];
    }
}

// modules_to_not_convert keeps a few matrices full width; mirrors
// GQ.denseMatvec.
kernel void bf16_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint NR0 = 8;
    const uint row0 = tgpig.x * NR0;
    const ulong rowBytes = (ulong) a.K * 2;
    device const uchar * W = weights + a.woff;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (uint r = 0; r < NR0; r++) {
        const uint row = row0 + r;
        if (row < a.M) {
            device const uchar * rp = W + (ulong) row * rowBytes;
            float s = 0.0f;
            for (uint k = tiisg; k < a.K; k += 32) {
                s += bf16_at(rp, k) * x[k];
            }
            acc[r] = s;
        }
    }
    for (uint r = 0; r < NR0; r++) {
        const float s = simd_sum(acc[r]);
        if (tiisg == 0 && row0 + r < a.M) { out[row0 + r] = s; }
    }
}

kernel void f16_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig.x;
    if (row < a.M) {
        device const half * rp = (device const half *) (weights + a.woff)
                               + (ulong) row * a.K;
        device const half4 * r4 = (device const half4 *) rp;
        device const float4 * x4 = (device const float4 *) x;
        const uint k4 = a.K / 4;
        float4 acc = 0.0f;
        for (uint k = tiisg; k < k4; k += 32) { acc += float4(r4[k]) * x4[k]; }
        float s = acc.x + acc.y + acc.z + acc.w;
        for (uint k = k4 * 4 + tiisg; k < a.K; k += 32) {
            s += (float) rp[k] * x[k];
        }
        s = simd_sum(s);
        if (tiisg == 0) { out[row] = s; }
    }
}

kernel void f32_gemv(
        device const uchar * weights [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant GemvArgs  & a       [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig.x;
    if (row < a.M) {
        device const float * rp = (device const float *) (weights + a.woff)
                                + (ulong) row * a.K;
        device const float4 * r4 = (device const float4 *) rp;
        device const float4 * x4 = (device const float4 *) x;
        const uint k4 = a.K / 4;
        float4 acc = 0.0f;
        for (uint k = tiisg; k < k4; k += 32) { acc += r4[k] * x4[k]; }
        float s = acc.x + acc.y + acc.z + acc.w;
        for (uint k = k4 * 4 + tiisg; k < a.K; k += 32) { s += rp[k] * x[k]; }
        s = simd_sum(s);
        if (tiisg == 0) { out[row] = s; }
    }
}

// `shmem` needs one float per simdgroup and is CLOBBERED, so two reductions
// in one kernel must pass disjoint regions; see vit_layernorm.
inline float tg_reduce_sum(float v, threadgroup float * shmem,
                           uint ntg, uint sgi, uint tii) {
    float s = simd_sum(v);
    if (tii == 0) { shmem[sgi] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0) {
        float t = (tii < (ntg + 31) / 32) ? shmem[tii] : 0.0f;
        t = simd_sum(t);
        if (tii == 0) { shmem[0] = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shmem[0];
}

// llama.cpp build_norm multiplies the stored weight directly (no 1+w).
struct NormArgs { ulong woff; uint n; float eps; };

// A draft only needs the index, so the 248320-wide row stays on the GPU. Ties
// go to the lower index, matching the host argmax it replaces.
kernel void argmax_rows(
        device const float * x    [[buffer(0)]],
        device       int   * out  [[buffer(1)]],
        constant     uint  & n    [[buffer(2)]],
        uint  tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_position_in_threadgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    threadgroup float bestv[32];
    threadgroup int   besti[32];
    device const float * row = x + (ulong) tgpig * n;
    float bv = -INFINITY;
    int bi = 0;
    for (uint i = tpitg; i < n; i += ntg) {
        const float v = row[i];
        if (v > bv) { bv = v; bi = (int) i; }
    }
    const ushort lane = tpitg % 32;
    const ushort warp = tpitg / 32;
    for (ushort off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(bv, off);
        const int   oi = simd_shuffle_down(bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { bestv[warp] = bv; besti[warp] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tpitg == 0) {
        float tv = bestv[0];
        int ti = besti[0];
        for (ushort w = 1; w < ntg / 32; w++) {
            if (bestv[w] > tv || (bestv[w] == tv && besti[w] < ti)) {
                tv = bestv[w];
                ti = besti[w];
            }
        }
        out[tgpig] = ti;
    }
}

kernel void rmsnorm(
        device const float * x       [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device       float * y       [[buffer(2)]],
        constant NormArgs  & a       [[buffer(3)]],
        threadgroup float  * shmem   [[threadgroup(0)]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += ntg) { ss += x[i] * x[i]; }
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.n + a.eps);
    device const float * w = (device const float *) (weights + a.woff);
    for (uint i = tid; i < a.n; i += ntg) { y[i] = x[i] * scale * w[i]; }
}

// To a SEPARATE output, since prefill's x is kept for the residual; rows sit
// xStride / yStride apart so a caller can read or write half a wide row.
struct NormBatchArgs {
    ulong woff; uint n; float eps; uint xStride; uint yStride;
};

kernel void rmsnorm_batch(
        device const float * x       [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device       float * y       [[buffer(2)]],
        constant NormBatchArgs & a   [[buffer(3)]],
        threadgroup float  * shmem   [[threadgroup(0)]],
        uint  row  [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    device const float * xr = x + (ulong) row * a.xStride;
    device       float * yr = y + (ulong) row * a.yStride;
    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += ntg) { ss += xr[i] * xr[i]; }
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.n + a.eps);
    device const float * w = (device const float *) (weights + a.woff);
    for (uint i = tid; i < a.n; i += ntg) { yr[i] = xr[i] * scale * w[i]; }
}

kernel void add_rmsnorm_batch(
        device       float * x       [[buffer(0)]],
        device const float * r       [[buffer(1)]],
        device const uchar * weights [[buffer(2)]],
        device       float * y       [[buffer(3)]],
        constant NormBatchArgs & a   [[buffer(4)]],
        threadgroup float  * shmem   [[threadgroup(0)]],
        uint  row  [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    device       float * xr = x + (ulong) row * a.xStride;
    device const float * rr = r + (ulong) row * a.xStride;
    device       float * yr = y + (ulong) row * a.yStride;
    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += ntg) {
        const float v = xr[i] + rr[i];
        xr[i] = v;
        ss += v * v;
    }
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.n + a.eps);
    device const float * w = (device const float *) (weights + a.woff);
    for (uint i = tid; i < a.n; i += ntg) { yr[i] = xr[i] * scale * w[i]; }
}

// L2Norm divides by sqrt(sum+eps), no mean, which is the difference from
// RMSNorm. `xoff` lets q|k|v share a buffer.
struct RowArgs { ulong woff; uint d; uint xoff; float eps; };

kernel void rmsnorm_rows(
        device       float * x       [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        constant RowArgs   & a       [[buffer(2)]],
        threadgroup float  * shmem   [[threadgroup(0)]],
        uint  row  [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    device float * r = x + a.xoff + row * a.d;
    float ss = 0.0f;
    for (uint i = tid; i < a.d; i += ntg) { ss += r[i] * r[i]; }
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.d + a.eps);
    device const float * w = (device const float *) (weights + a.woff);
    for (uint i = tid; i < a.d; i += ntg) { r[i] = r[i] * scale * w[i]; }
}

kernel void rmsnorm_rows_silu(
        device       float * x       [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device const float * z       [[buffer(2)]],
        constant RowArgs   & a       [[buffer(3)]],
        threadgroup float  * shmem   [[threadgroup(0)]],
        uint  row  [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    device float * r = x + a.xoff + row * a.d;
    device const float * zr = z + a.xoff + row * a.d;
    float ss = 0.0f;
    for (uint i = tid; i < a.d; i += ntg) { ss += r[i] * r[i]; }
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.d + a.eps);
    device const float * w = (device const float *) (weights + a.woff);
    for (uint i = tid; i < a.d; i += ntg) {
        r[i] = r[i] * scale * w[i] * siluf(zr[i]);
    }
}

kernel void l2norm_rows(
        device       float * x     [[buffer(0)]],
        constant RowArgs   & a     [[buffer(1)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint  row  [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    device float * r = x + a.xoff + row * a.d;
    float ss = 0.0f;
    for (uint i = tid; i < a.d; i += ntg) { ss += r[i] * r[i]; }
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq + a.eps);
    for (uint i = tid; i < a.d; i += ntg) { r[i] *= scale; }
}

// One dispatch replaces the 2*N per-token calls that made GDN prefill
// CPU-bound.
struct RowBatchArgs { uint d; uint rowsPerTok; uint tokStride; float eps; };

kernel void l2norm_rows_batch(
        device       float * x     [[buffer(0)]],
        constant RowBatchArgs & a  [[buffer(1)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint  gr   [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    const uint n = gr / a.rowsPerTok;
    const uint local = gr % a.rowsPerTok;
    device float * r = x + (ulong) n * a.tokStride + (ulong) local * a.d;
    float ss = 0.0f;
    for (uint i = tid; i < a.d; i += ntg) { ss += r[i] * r[i]; }
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq + a.eps);
    for (uint i = tid; i < a.d; i += ntg) { r[i] *= scale; }
}

// Separate kernels rather than a type flag on NormArgs/RowArgs, so the
// ternary path's kernels stay byte-identical.
kernel void rmsnorm_bf16(
        device const float * x       [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device       float * y       [[buffer(2)]],
        constant NormArgs  & a       [[buffer(3)]],
        threadgroup float  * shmem   [[threadgroup(0)]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += ntg) { ss += x[i] * x[i]; }
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.n + a.eps);
    device const uchar * w = weights + a.woff;
    for (uint i = tid; i < a.n; i += ntg) {
        y[i] = x[i] * scale * bf16_at(w, i);
    }
}

kernel void rmsnorm_rows_bf16(
        device       float * x       [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        constant RowArgs   & a       [[buffer(2)]],
        threadgroup float  * shmem   [[threadgroup(0)]],
        uint  row  [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    device float * r = x + a.xoff + row * a.d;
    float ss = 0.0f;
    for (uint i = tid; i < a.d; i += ntg) { ss += r[i] * r[i]; }
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.d + a.eps);
    device const uchar * w = weights + a.woff;
    for (uint i = tid; i < a.d; i += ntg) {
        r[i] = r[i] * scale * bf16_at(w, i);
    }
}

struct NormAddArgs {
    ulong woffPost; ulong woffPre; uint n; float eps; float s; uint pre;
};

kernel void norm_add_norm_bf16(
        device       float * x       [[buffer(0)]],
        device const float * c       [[buffer(1)]],
        device const uchar * wPost   [[buffer(2)]],
        device const uchar * wPre    [[buffer(3)]],
        device       float * y       [[buffer(4)]],
        constant NormAddArgs & a     [[buffer(5)]],
        threadgroup float  * shmem   [[threadgroup(0)]],
        uint  row  [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    device       float * xr = x + (ulong) row * a.n;
    device const float * cr = c + (ulong) row * a.n;
    device       float * yr = y + (ulong) row * a.n;
    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += ntg) { ss += cr[i] * cr[i]; }
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.n + a.eps);
    device const uchar * wp = wPost + a.woffPost;
    float s2 = 0.0f;
    for (uint i = tid; i < a.n; i += ntg) {
        const float t = cr[i] * scale * bf16_at(wp, i);
        const float v = (xr[i] + t) * a.s;
        xr[i] = v;
        s2 += v * v;
    }
    if (a.pre != 0) {
        const float sumsq2 = tg_reduce_sum(s2, shmem + 32, ntg, sgi, tii);
        const float scale2 = 1.0f / sqrt(sumsq2 / (float) a.n + a.eps);
        device const uchar * wq = wPre + a.woffPre;
        for (uint i = tid; i < a.n; i += ntg) {
            yr[i] = xr[i] * scale2 * bf16_at(wq, i);
        }
    }
}

// gemma's v_norm is Gemma4RMSNorm(with_scale=False), a real op that leaves
// no tensor in the checkpoint; divides by the RMS (note the mean).
kernel void rmsnorm_rows_noweight(
        device       float * x     [[buffer(0)]],
        constant RowArgs   & a     [[buffer(1)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint  row  [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    device float * r = x + a.xoff + row * a.d;
    float ss = 0.0f;
    for (uint i = tid; i < a.d; i += ntg) { ss += r[i] * r[i]; }
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.d + a.eps);
    for (uint i = tid; i < a.d; i += ntg) { r[i] *= scale; }
}

// TRAP: pairs (j, j + headDim/2), NOT rope_neox's (i, i + nRot/2).
struct RopeGemmaArgs {
    uint headDim; uint nHead; uint rotated; float base; uint pos;
};

// The PAIRING stays the CALLER's, deliberately.
inline void rope_pair(device float * x, uint i1, uint i2,
                      float c, float s) {
    const float p = x[i1];
    const float q = x[i2];
    x[i1] = p * c - q * s;
    x[i2] = p * s + q * c;
}

kernel void rope_gemma(
        device float           * x [[buffer(0)]],
        constant RopeGemmaArgs & a [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.nHead * a.rotated) {
        const uint hf = a.headDim / 2;
        const uint h = gid / a.rotated;
        const uint j = gid % a.rotated;
        const uint hoff = h * a.headDim;
        const float freq = pow(a.base, -2.0f * (float) j / (float) a.headDim);
        const float ang = (float) a.pos * freq;
        rope_pair(x, hoff + j, hoff + j + hf, cos(ang), sin(ang));
    }
}

kernel void rmsnorm_batch_bf16(
        device const float * x       [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device       float * y       [[buffer(2)]],
        constant NormArgs  & a       [[buffer(3)]],
        threadgroup float  * shmem   [[threadgroup(0)]],
        uint  row  [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    device const float * xr = x + (ulong) row * a.n;
    device       float * yr = y + (ulong) row * a.n;
    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += ntg) { ss += xr[i] * xr[i]; }
    const float sumsq = tg_reduce_sum(ss, shmem, ntg, sgi, tii);
    const float scale = 1.0f / sqrt(sumsq / (float) a.n + a.eps);
    device const uchar * w = weights + a.woff;
    for (uint i = tid; i < a.n; i += ntg) {
        yr[i] = xr[i] * scale * bf16_at(w, i);
    }
}

// TWO-dimensional: the first half rotates by the patch's x, the second by y.
// TRAP: pairs WITHIN each axis half, not across the head like vit_rope.
struct GVRopeArgs {
    uint rowStride; uint off; uint headDim; uint nHead; uint N;
};

kernel void gemma_vit_rope(
        device float        * x    [[buffer(0)]],
        device const float  * cosT [[buffer(1)]],
        device const float  * sinT [[buffer(2)]],
        constant GVRopeArgs & a    [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const uint hd = a.headDim;
    const uint per = hd / 2;        // channels driven by one axis
    const uint hf = per / 2;        // rotated pairs per axis
    const uint pairs = 2 * hf;
    const uint perRow = a.nHead * pairs;
    if (gid < a.N * perRow) {
        const uint p = gid / perRow;
        const uint r = gid % perRow;
        const uint h = r / pairs;
        const uint pi = r % pairs;
        const uint axis = pi / hf;
        const uint j = pi % hf;
        const uint base = p * a.rowStride + a.off + h * hd + axis * per;
        const uint t = p * hd + axis * per + j;
        rope_pair(x, base + j, base + j + hf, cosT[t], sinT[t]);
    }
}

// TRAP: padding patches must be MASKED, not merely zeroed; a padded key still
// has a finite score, so left unmasked it takes real weight.
struct GVAttnArgs { uint n; uint hd; uint nHead; };

kernel void gemma_vit_attn(
        device const float * q     [[buffer(0)]],
        device const float * k     [[buffer(1)]],
        device const float * v     [[buffer(2)]],
        device const float * mask  [[buffer(3)]],
        device       float * out   [[buffer(4)]],
        constant GVAttnArgs & a    [[buffer(5)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint   tgid  [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint hd = a.hd;
    const uint row = tgid / a.nHead;
    const uint head = tgid % a.nHead;
    const uint stride = a.nHead * hd;
    threadgroup float * qs = shmem;
    device const float * qh = q + (ulong) row * stride + head * hd;
    for (uint i = tiisg; i < hd; i += 32) { qs[i] = qh[i]; }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY, l = 0.0f;
    float acc[8];
    const ushort SL = (ushort) ((hd + 31) / 32);
    for (ushort u = 0; u < SL; u++) { acc[u] = 0.0f; }
    for (uint t0 = 0; t0 < a.n; t0 += 32) {
        const uint t = t0 + tiisg;
        float sc = -INFINITY;
        if (t < a.n && mask[t] == 0.0f) {
            device const float * kt = k + (ulong) t * stride + head * hd;
            float p = 0.0f;
            for (uint i = 0; i < hd; i++) { p += qs[i] * kt[i]; }
            sc = p;
        }
        const float mNew = max(m, simd_max(sc));
        const float corr = exp(m - mNew);
        const float w = sc > -INFINITY ? exp(sc - mNew) : 0.0f;
        l = l * corr + simd_sum(w);
        for (ushort u = 0; u < SL; u++) { acc[u] *= corr; }
        const uint tk = min(32u, a.n - t0);
        for (ushort j = 0; j < tk; j++) {
            const float wt = simd_broadcast(w, j);
            device const float * vt =
                v + (ulong) (t0 + j) * stride + head * hd;
            for (ushort u = 0; u < SL; u++) {
                const uint i = tiisg + 32 * u;
                if (i < hd) { acc[u] += wt * vt[i]; }
            }
        }
        m = mNew;
    }
    const float inv = 1.0f / l;
    device float * o = out + (ulong) row * stride + head * hd;
    for (ushort u = 0; u < SL; u++) {
        const uint i = tiisg + 32 * u;
        if (i < hd) { o[i] = acc[u] * inv; }
    }
}

// Carries embed_scale, the per-layer embed scale, 1/sqrt(dim) on the
// per-layer projection, and layer_scalar (the WHOLE layer output, last).
kernel void scale_inplace(device float * x [[buffer(0)]],
                          constant uint & n [[buffer(1)]],
                          constant float & s [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid < n) { x[gid] *= s; }
}

// The per-layer input is the mean of table row and projected hidden, 1/sqrt(2).
kernel void add_scaled(device float * a [[buffer(0)]],
                       device const float * b [[buffer(1)]],
                       constant uint & n [[buffer(2)]],
                       constant float & s [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < n) { a[gid] = (a[gid] + b[gid]) * s; }
}

// gemma activates with gelu where the ternary path uses silu.
struct GeluMulArgs { uint n; uint boff; };

kernel void gelu_mul(device float * a [[buffer(0)]],
                     device const float * b [[buffer(1)]],
                     constant GeluMulArgs & g [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < g.n) {
        const float v = a[gid];
        const float t =
            precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v));
        a[gid] = 0.5f * v * (1.0f + t) * b[g.boff + gid];
    }
}

// A and B have DIFFERENT row strides; one dispatch instead of ~9000 a chunk.
struct GeluMulRowsArgs {
    uint n; uint rows; uint aStride; uint bStride; uint boff;
};

kernel void gelu_mul_rows(
        device       float          * a [[buffer(0)]],
        device const float          * b [[buffer(1)]],
        constant GeluMulRowsArgs    & g [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < g.rows * g.n) {
        const uint r = gid / g.n;
        const uint i = gid % g.n;
        const uint ai = r * g.aStride + i;
        const float v = a[ai];
        const float t =
            precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v));
        a[ai] = 0.5f * v * (1.0f + t) * b[r * g.bStride + g.boff + i];
    }
}

// tanh(x/cap)*cap on the logits: nothing in gemma's reference dump exceeds
// +/-30 because the model caps them here, not because the weights are small.
kernel void softcap(device float * x [[buffer(0)]],
                    constant uint & n [[buffer(1)]],
                    constant float & cap [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < n) { x[gid] = precise::tanh(x[gid] / cap) * cap; }
}

// Partial NEOX RoPE on the first nRot dims of each head (Kern.ropeNeox).
struct RopeArgs { uint headDim; uint nHead; uint nRot; float base; uint pos; };

kernel void rope_neox(
        device float     * x   [[buffer(0)]],
        constant RopeArgs & a  [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const uint hf = a.nRot / 2;
    if (gid < a.nHead * hf) {
        const uint h = gid / hf;
        const uint i = gid % hf;
        const uint hoff = h * a.headDim;
        const float freq = pow(a.base, -2.0f * (float) i / (float) a.nRot);
        const float ang = (float) a.pos * freq;
        rope_pair(x, hoff + i, hoff + i + hf, cos(ang), sin(ang));
    }
}

// FFN gate*up; the GDN o*silu(z) is mul_silu.
kernel void silu_mul(device float * a [[buffer(0)]],
                     device const float * b [[buffer(1)]],
                     constant uint & n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < n) { a[gid] = siluf(a[gid]) * b[gid]; }
}

// a[i] *= silu(b[i])   (o *= silu(z))
kernel void mul_silu(device float * a [[buffer(0)]],
                     device const float * b [[buffer(1)]],
                     constant uint & n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < n) { a[gid] = a[gid] * siluf(b[gid]); }
}

kernel void accum_outer(device float * h [[buffer(0)]],
                        device const float * x [[buffer(1)]],
                        constant uint & n [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x < n && gid.y < n) {
        h[(ulong) gid.y * n + gid.x] += x[gid.y] * x[gid.x];
    }
}

kernel void accum_sq(device float * dst [[buffer(0)]],
                     device const float * src [[buffer(1)]],
                     constant uint & n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < n) { dst[gid] += src[gid] * src[gid]; }
}

kernel void add_inplace(device float * x [[buffer(0)]],
                        device const float * y [[buffer(1)]],
                        constant uint & n [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < n) { x[gid] += y[gid]; }
}

// dt (ssm_dt.bias) and aNeg (ssm_a = -exp(A_log)) are SEPARATE f32 tensors
// and take a buffer each, since nothing puts two tensors in one window.
struct GateArgs { ulong dtOff; ulong aOff; uint nV; };

kernel void gdn_gate(
        device const float * bPre    [[buffer(0)]],
        device const float * aPre    [[buffer(1)]],
        device const uchar * dtW     [[buffer(2)]],
        device const uchar * aW      [[buffer(3)]],
        device       float * beta    [[buffer(4)]],
        device       float * g       [[buffer(5)]],
        constant GateArgs  & a       [[buffer(6)]],
        constant uint      & total   [[buffer(7)]],
        uint gid [[thread_position_in_grid]]) {
    // `total` is nV or N*nV; dt/aNeg are per value head, so index by gid % nV.
    if (gid < total) {
        device const float * dt   = (device const float *) (dtW + a.dtOff);
        device const float * aNeg = (device const float *) (aW + a.aOff);
        const uint h = gid % a.nV;
        beta[gid] = sigmoidf(bPre[gid]);
        g[gid] = softplusf(aPre[gid] + dt[h]) * aNeg[h];
    }
}

// window per channel c: [convState(3), qkvMix[c]], w(j,c) at c*4+j. Each
// thread reads its channel's ring then overwrites it, no cross-thread hazard.
struct ConvArgs { ulong cwOff; uint convDim; uint dConv; };

kernel void gdn_conv(
        device const float * qkvMix   [[buffer(0)]],
        device       float * convState[[buffer(1)]],
        device const uchar * weights  [[buffer(2)]],
        device       float * out      [[buffer(3)]],
        constant ConvArgs  & a        [[buffer(4)]],
        uint c [[thread_position_in_grid]]) {
    if (c < a.convDim) {
        device const float * cw =
            (device const float *) (weights + a.cwOff) + c * a.dConv;
        const uint kc = a.dConv;
        float acc = 0.0f;
        for (uint j = 0; j + 1 < kc; j++) {
            acc += convState[j * a.convDim + c] * cw[j];
        }
        acc += qkvMix[c] * cw[kc - 1];
        out[c] = siluf(acc);
        for (uint j = 0; j + 2 < kc; j++) {
            convState[j * a.convDim + c] =
                convState[(j + 1) * a.convDim + c];
        }
        if (kc >= 2) { convState[(kc - 2) * a.convDim + c] = qkvMix[c]; }
    }
}

// qFull is [q(hd) | gate(hd)] per head (Attn.step split loop).
struct SplitArgs { uint hd; uint nH; };

kernel void split_qgate(
        device const float * qFull [[buffer(0)]],
        device       float * q     [[buffer(1)]],
        device       float * gate  [[buffer(2)]],
        constant SplitArgs & a     [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.hd * a.nH) {
        const uint h = gid / a.hd;
        const uint i = gid % a.hd;
        const uint src = h * a.hd * 2;
        q[gid]    = qFull[src + i];
        gate[gid] = qFull[src + a.hd + i];
    }
}

struct AttnArgs {
    uint hd; uint nH; uint nKV; uint T; uint kvDim; uint P;
    float scale; uint gated; uint lo;
};

// A page table of device pointers held as raw gpuAddresses; KV_MAXP * P
// bounds the context (1M positions). K/V are HALF, accumulated f32.
#define KV_MAXP 2048
struct KVTable { device const half * pages[KV_MAXP]; };

// The same arithmetic serves K and V; it was written twice.
inline device const half * kv_row(device const KVTable & tab, uint t,
                                  uint P, uint kvDim, uint kvh, uint hd) {
    return tab.pages[t / P] + (t % P) * kvDim + kvh * hd;
}

// TRAP: vectorize only when hd % 4 == 0, which is also what makes the
// kvh * hd row base 8B-aligned. The scalar tail covers the rest.
inline float attn_score(threadgroup const float * qs,
                        device const half * kt, uint hd, float scale) {
    const uint hd4 = (hd % 4 == 0) ? hd : 0;
    threadgroup const float4 * q4 = (threadgroup const float4 *) qs;
    device const half4 * k4 = (device const half4 *) kt;
    float4 p4 = 0.0f;
    for (uint i = 0; i < hd4 / 4; i++) {
        p4 += q4[i] * float4(k4[i]);
    }
    float p = p4.x + p4.y + p4.z + p4.w;
    for (uint i = hd4; i < hd; i++) {
        p += qs[i] * (float) kt[i];
    }
    return p * scale;
}

// An empty stripe has m = -INF, so exp(m - M) = 0 and adds nothing.
inline void attn_combine(threadgroup const float * redM,
                         threadgroup const float * redL,
                         threadgroup const float * redAcc,
                         device const float * gate,
                         device       float * out,
                         uint hd, uint gated, ushort NSG, ushort tiisg) {
    const ushort SL = (ushort) ((hd + 31) / 32);   // dim slices per lane (<=8)
    float M = -INFINITY;
    for (ushort g = 0; g < NSG; g++) {
        M = max(M, redM[g]);
    }
    float L = 0.0f;
    for (ushort g = 0; g < NSG; g++) {
        L += redL[g] * exp(redM[g] - M);
    }
    const float inv = 1.0f / L;
    for (ushort u = 0; u < SL; u++) {
        const uint j = tiisg + 32 * u;
        if (j < hd) {
            float o = 0.0f;
            for (ushort g = 0; g < NSG; g++) {
                o += redAcc[g * hd + j] * exp(redM[g] - M);
            }
            o *= inv;
            // Dense qwen3 has no output gate; `gated` selects, gate unread when
            // 0.
            out[j] = gated ? o * sigmoidf(gate[j]) : o;
        }
    }
}

// q, gate and out arrive ALREADY OFFSET to this (row, head).
inline void attn_stripes(
        device const float   * qh,
        device const KVTable & kT,
        device const KVTable & vT,
        uint hd, uint kvh, uint kvDim, uint P, float scale,
        uint lo, uint T,
        threadgroup float    * shmem,
        ushort sgitg,
        ushort tiisg) {
    const ushort NSG = 4, TK = 32;
    // shmem = qs[hd] | redM[NSG] | redL[NSG] | redAcc[NSG*hd], O(hd) not O(T).
    threadgroup float * qs     = shmem;
    threadgroup float * redM   = qs + hd;
    threadgroup float * redL   = redM + NSG;
    threadgroup float * redAcc = redL + NSG;
    for (uint i = sgitg * 32 + tiisg; i < hd; i += NSG * 32) {
        qs[i] = qh[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Inline: every step mutates m, l and acc4 together.
    float m = -INFINITY, l = 0.0f;
    // The V accumulator is FOUR-WIDE: lane L owns dims 4L..4L+3 of each 128-dim
    // slice, so one half4 load per lane per key covers four scalar loads.
    const ushort VS = (ushort) ((hd + 127) / 128);
    float4 acc4[4];
    for (ushort u = 0; u < VS; u++) {
        acc4[u] = 0.0f;
    }
    for (uint t0 = lo + sgitg * TK; t0 < T; t0 += NSG * TK) {
        const uint tk = min((uint) TK, T - t0);
        // lane tiisg owns key t0+tiisg
        float sc = -INFINITY;
        if (tiisg < tk) {
            device const half * kt =
                kv_row(kT, t0 + tiisg, P, kvDim, kvh, hd);
            sc = attn_score(qs, kt, hd, scale);
        }
        const float mNew = max(m, simd_max(sc));
        const float corr = exp(m - mNew);
        const float w = sc > -INFINITY ? exp(sc - mNew) : 0.0f;
        l = l * corr + simd_sum(w);
        for (ushort u = 0; u < VS; u++) {
            acc4[u] *= corr;
        }
        for (ushort t = 0; t < tk; t++) {
            const float wt = simd_broadcast(w, t);
            device const half4 * v4 = (device const half4 *)
                kv_row(vT, t0 + t, P, kvDim, kvh, hd);
            for (ushort u = 0; u < VS; u++) {
                const uint j4 = tiisg + 32 * u;
                if (4 * j4 < hd) {
                    acc4[u] += wt * float4(v4[j4]);
                }
            }
        }
        m = mNew;
    }
    if (tiisg == 0) {
        redM[sgitg] = m;
        redL[sgitg] = l;
    }
    // redAcc stays a flat [NSG,hd] array, so attn_combine sees a flat layout.
    for (ushort u = 0; u < VS; u++) {
        const uint j = 4 * (tiisg + 32 * u);
        if (j < hd) {
            redAcc[sgitg * hd + j + 0] = acc4[u].x;
            redAcc[sgitg * hd + j + 1] = acc4[u].y;
            redAcc[sgitg * hd + j + 2] = acc4[u].z;
            redAcc[sgitg * hd + j + 3] = acc4[u].w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void attn_flash(
        device const float   * qh,
        device const KVTable & kT,
        device const KVTable & vT,
        device const float   * gate,
        device       float   * out,
        uint hd, uint kvh, uint kvDim, uint P, float scale, uint gated,
        uint lo, uint T,
        threadgroup float    * shmem,
        ushort sgitg,
        ushort tiisg) {
    const ushort NSG = 4;
    attn_stripes(qh, kT, vT, hd, kvh, kvDim, P, scale, lo, T, shmem,
                 sgitg, tiisg);
    threadgroup const float * redM = shmem + hd;
    threadgroup const float * redL = redM + NSG;
    threadgroup const float * redAcc = redL + NSG;
    if (sgitg == 0) {
        attn_combine(redM, redL, redAcc, gate, out, hd, gated, NSG, tiisg);
    }
}

inline void attn_partial(threadgroup const float * redM,
                         threadgroup const float * redL,
                         threadgroup const float * redAcc,
                         device float * part,
                         uint hd, ushort NSG, ushort tiisg) {
    const ushort SL = (ushort) ((hd + 31) / 32);
    float M = -INFINITY;
    for (ushort g = 0; g < NSG; g++) { M = max(M, redM[g]); }
    float L = 0.0f;
    for (ushort g = 0; g < NSG; g++) {
        L += redM[g] > -INFINITY ? redL[g] * exp(redM[g] - M) : 0.0f;
    }
    if (tiisg == 0) {
        part[0] = M;
        part[1] = L;
    }
    for (ushort u = 0; u < SL; u++) {
        const uint j = tiisg + 32 * u;
        if (j < hd) {
            float o = 0.0f;
            for (ushort g = 0; g < NSG; g++) {
                o += redM[g] > -INFINITY
                    ? redAcc[g * hd + j] * exp(redM[g] - M) : 0.0f;
            }
            part[2 + j] = o;
        }
    }
}

struct AttnSplitArgs {
    uint hd; uint nH; uint nKV; uint T; uint kvDim; uint P;
    float scale; uint gated; uint lo; uint splits; uint span;
};

kernel void attn_head_part(
        device const float   * q     [[buffer(0)]],
        device const KVTable & kT    [[buffer(1)]],
        device const KVTable & vT    [[buffer(2)]],
        device       float   * parts [[buffer(3)]],
        constant AttnSplitArgs & a   [[buffer(4)]],
        threadgroup float    * shmem [[threadgroup(0)]],
        uint   tg    [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const ushort NSG = 4;
    const uint h = tg / a.splits;
    const uint s = tg % a.splits;
    const uint group = a.nH / a.nKV;
    const uint lo = a.lo + s * a.span;
    const uint hi = min(a.T, lo + a.span);
    attn_stripes(q + h * a.hd, kT, vT, a.hd, h / group, a.kvDim, a.P,
                 a.scale, lo, hi, shmem, sgitg, tiisg);
    threadgroup const float * redM = shmem + a.hd;
    threadgroup const float * redL = redM + NSG;
    threadgroup const float * redAcc = redL + NSG;
    if (sgitg == 0) {
        attn_partial(redM, redL, redAcc, parts + (ulong) tg * (a.hd + 2),
                     a.hd, NSG, tiisg);
    }
}

kernel void attn_reduce(
        device const float   * parts [[buffer(0)]],
        device const float   * gate  [[buffer(1)]],
        device       float   * out   [[buffer(2)]],
        constant AttnSplitArgs & a   [[buffer(3)]],
        uint   h     [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint hd = a.hd;
    const ushort SL = (ushort) ((hd + 31) / 32);
    device const float * base = parts + (ulong) h * a.splits * (hd + 2);
    float M = -INFINITY;
    for (uint s = 0; s < a.splits; s++) { M = max(M, base[s * (hd + 2)]); }
    float L = 0.0f;
    for (uint s = 0; s < a.splits; s++) {
        const float Ms = base[s * (hd + 2)];
        L += Ms > -INFINITY ? base[s * (hd + 2) + 1] * exp(Ms - M) : 0.0f;
    }
    const float inv = 1.0f / L;
    const uint off = h * hd;
    for (ushort u = 0; u < SL; u++) {
        const uint j = tiisg + 32 * u;
        if (j < hd) {
            float o = 0.0f;
            for (uint s = 0; s < a.splits; s++) {
                const float Ms = base[s * (hd + 2)];
                o += Ms > -INFINITY
                    ? base[s * (hd + 2) + 2 + j] * exp(Ms - M) : 0.0f;
            }
            o *= inv;
            out[off + j] = a.gated ? o * sigmoidf(gate[off + j]) : o;
        }
    }
}

kernel void attn_head(
        device const float   * q     [[buffer(0)]],
        device const KVTable & kT    [[buffer(1)]],
        device const KVTable & vT    [[buffer(2)]],
        device const float   * gate  [[buffer(3)]],
        device       float   * out   [[buffer(4)]],
        constant AttnArgs    & a     [[buffer(5)]],
        threadgroup float    * shmem [[threadgroup(0)]],
        uint   h     [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint group = a.nH / a.nKV;
    const uint off = h * a.hd;
    attn_flash(q + off, kT, vT, gate + off, out + off,
               a.hd, h / group, a.kvDim, a.P, a.scale, a.gated, a.lo, a.T,
               shmem, sgitg, tiisg);
}

// The f32 -> half narrowing happens HERE, once, so every later read moves half.
struct KVArgs { uint kvDim; uint pos; };

kernel void kv_append(
        device const float * kCur [[buffer(0)]],
        device const float * vCur [[buffer(1)]],
        device       half  * K    [[buffer(2)]],
        device       half  * V    [[buffer(3)]],
        constant KVArgs    & a    [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.kvDim) {
        K[a.pos * a.kvDim + gid] = (half) kCur[gid];
        V[a.pos * a.kvDim + gid] = (half) vCur[gid];
    }
}

// BATCHED (prefill): activations token-major [N, dim]. The recurrences
// (conv, scan) loop N internally; the parallel ops grid N.

// woff = token_embd base, K = nEmbd, M = N, rowBytes = the type's row stride.
struct EmbedArgs { ulong woff; uint K; uint M; uint rowBytes; };

kernel void embed_batch(
        device const uchar * weights [[buffer(0)]],
        device const int   * ids     [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant EmbedArgs & a       [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.M * a.K) {
        const uint n = gid / a.K;
        const uint k = gid % a.K;
        device const uchar * bp = weights + a.woff
            + (ulong) ids[n] * a.rowBytes + (ulong) (k / 128) * 34;
        const float d = (float) (*(device const half *) bp);
        const uint j = k % 128;
        const uchar code = (bp[2 + j / 4] >> ((j & 3) * 2)) & 3;
        out[gid] = (float) ((int) code - 1) * d;
    }
}


kernel void q4_0_embed_batch(
        device const uchar * weights [[buffer(0)]],
        device const int   * ids     [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant EmbedArgs & a       [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.M * a.K) {
        const uint n = gid / a.K;
        const uint k = gid % a.K;
        device const uchar * bp = weights + a.woff
            + (ulong) ids[n] * a.rowBytes + (ulong) (k / 32) * 18;
        const float d = (float) (*(device const half *) bp);
        const uint j = k % 32;
        const uchar b = bp[2 + (j % 16)];
        const uchar code = (j < 16) ? (b & 0x0F) : (b >> 4);
        out[gid] = ((float) code - 8.0f) * d;
    }
}

kernel void q8_0_embed_batch(
        device const uchar * weights [[buffer(0)]],
        device const int   * ids     [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant EmbedArgs & a       [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.M * a.K) {
        const uint n = gid / a.K;
        const uint k = gid % a.K;
        device const uchar * bp = weights + a.woff
            + (ulong) ids[n] * a.rowBytes + (ulong) (k / 32) * 34;
        const float d = (float) (*(device const half *) bp);
        const char q = ((device const char *) (bp + 2))[k % 32];
        out[gid] = (float) q * d;
    }
}

kernel void bf16_embed_batch(
        device const uchar * weights [[buffer(0)]],
        device const int   * ids     [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant EmbedArgs & a       [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.M * a.K) {
        const uint n = gid / a.K;
        const uint k = gid % a.K;
        device const uchar * rp = weights + a.woff
            + (ulong) ids[n] * a.rowBytes;
        out[gid] = bf16_at(rp, k);
    }
}

kernel void f16_embed_batch(
        device const uchar * weights [[buffer(0)]],
        device const int   * ids     [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant EmbedArgs & a       [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.M * a.K) {
        const uint n = gid / a.K;
        const uint k = gid % a.K;
        device const uchar * rp = weights + a.woff
            + (ulong) ids[n] * a.rowBytes;
        out[gid] = (float) ((device const half *) rp)[k];
    }
}

kernel void f32_embed_batch(
        device const uchar * weights [[buffer(0)]],
        device const int   * ids     [[buffer(1)]],
        device       float * out     [[buffer(2)]],
        constant EmbedArgs & a       [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.M * a.K) {
        const uint n = gid / a.K;
        const uint k = gid % a.K;
        device const uchar * rp = weights + a.woff
            + (ulong) ids[n] * a.rowBytes;
        out[gid] = ((device const float *) rp)[k];
    }
}

// The ring lives in registers across the batch; mirrors gdn_conv looped.
struct ConvBatchArgs {
    ulong cwOff; uint convDim; uint dConv; uint N;
    uint slot0; uint slots; uint stateElems;
};

kernel void gdn_conv_batch(
        device const float * qkvMixN [[buffer(0)]],
        device       float * convState [[buffer(1)]],
        device const uchar * weights [[buffer(2)]],
        device       float * outN    [[buffer(3)]],
        constant ConvBatchArgs & a   [[buffer(4)]],
        uint c [[thread_position_in_grid]]) {
    if (c < a.convDim) {
        device const float * cw =
            (device const float *) (weights + a.cwOff) + c * a.dConv;
        const uint kc = a.dConv;
        float ring[3];
        device float * Sin = convState + (ulong) a.slot0 * a.stateElems;
        for (uint j = 0; j + 1 < kc; j++) {
            ring[j] = Sin[j * a.convDim + c];
        }
        for (uint n = 0; n < a.N; n++) {
            const float cur = qkvMixN[n * a.convDim + c];
            float acc = 0.0f;
            for (uint j = 0; j + 1 < kc; j++) { acc += ring[j] * cw[j]; }
            acc += cur * cw[kc - 1];
            outN[n * a.convDim + c] = siluf(acc);
            for (uint j = 0; j + 2 < kc; j++) { ring[j] = ring[j + 1]; }
            if (kc >= 2) { ring[kc - 2] = cur; }
            // the window after token n, from registers.
            if (a.slots > 1 || n + 1 == a.N) {
                device float * So = convState
                    + (ulong) ((a.slot0 + 1 + n) % a.slots) * a.stateElems;
                for (uint j = 0; j + 1 < kc; j++) {
                    So[j * a.convDim + c] = ring[j];
                }
            }
        }
    }
}

// The scan READS slot0 and writes the state after token t to slot
// (slot0+1+t) % slots from registers; slots == 1 is the in-place update.
struct ScanBatchArgs {
    uint nV; uint nK; uint dS; float qScale;
    uint N; uint convDim; uint keyDim; uint valueDim;
    uint slot0; uint slots; uint stateElems;
};

kernel void gdn_scan_batch(
        device const float * convOutN [[buffer(0)]],
        device const float * gN       [[buffer(1)]],
        device const float * betaN    [[buffer(2)]],
        device       float * S        [[buffer(3)]],
        device       float * oN       [[buffer(4)]],
        constant ScanBatchArgs & a    [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    const uint dS = a.dS;
    if (gid < a.nV * dS) {
        const uint hv = gid / dS;
        const uint j  = gid % dS;
        const uint hk = hv % a.nK;
        device float * Sh = S + (ulong) a.slot0 * a.stateElems + hv * dS * dS;
        for (uint n = 0; n < a.N; n++) {
            // carry into the next slot BEFORE updating, so slot0
            // survives as the pre-pass state this kernel may be rolled to.
            if (a.slots > 1) {
                device float * So = S
                    + (ulong) ((a.slot0 + 1 + n) % a.slots) * a.stateElems
                    + hv * dS * dS;
                for (uint i = 0; i < dS; i++) {
                    So[i * dS + j] = Sh[i * dS + j];
                }
                Sh = So;
            }
            device const float * row = convOutN + n * a.convDim;
            const uint kOff = hk * dS, vOff = hv * dS;
            device const float * q = row + kOff;
            device const float * k = row + a.keyDim + kOff;
            device const float * v = row + 2 * a.keyDim + vOff;
            const float gamma = exp(gN[n * a.nV + hv]);
            const float b = betaN[n * a.nV + hv];
            float sk = 0.0f;
            for (uint i = 0; i < dS; i++) {
                const float s = Sh[i * dS + j] * gamma;
                Sh[i * dS + j] = s;
                sk += s * k[i];
            }
            const float d = b * (v[j] - sk);
            float acc = 0.0f;
            for (uint i = 0; i < dS; i++) {
                const float s = Sh[i * dS + j] + k[i] * d;
                Sh[i * dS + j] = s;
                acc += s * q[i] * a.qScale;
            }
            oN[n * a.valueDim + hv * dS + j] = acc;
        }
    }
}

// The delta-rule scan with the state RESIDENT in registers over the whole
// N-token sequence; KS = dS / 32 slices per lane, the host routes other dS.
template <ushort KS>
__attribute__((always_inline))
inline void gdn_scan_slice(device const float * k, device const float * q,
                           thread float * kk, thread float * qq) {
    if (KS == 4) {
        const float4 k4 = *(device const float4 *) k;
        const float4 q4 = *(device const float4 *) q;
        #pragma clang loop unroll(full)
        for (ushort m = 0; m < 4; m++) { kk[m] = k4[m]; qq[m] = q4[m]; }
    } else {
        #pragma clang loop unroll(full)
        for (ushort m = 0; m < KS; m++) { kk[m] = k[m]; qq[m] = q[m]; }
    }
}

template <ushort KS>
inline void gdn_scan_impl(
        device const float * convOutN,
        device const float * gN,
        device const float * betaN,
        device       float * S,
        device       float * oN,
        constant ScanBatchArgs & a,
        uint3  tgpig,
        ushort sgitg,
        ushort tiisg) {
    const uint dS = 32 * KS;
    const ushort COLS = 4; // columns owned per threadgroup
    const uint hv = tgpig.y;
    const uint j  = tgpig.x * COLS + sgitg;
    const uint hk = hv % a.nK;
    const uint i0 = KS * tiisg; // lane owns keys i0 .. i0 + KS - 1
    device float * Sh = S + (ulong) a.slot0 * a.stateElems + hv * dS * dS;
    float ls[KS];
    #pragma clang loop unroll(full)
    for (ushort m = 0; m < KS; m++) { ls[m] = Sh[(i0 + m) * dS + j]; }
    device const float * q = convOutN + hk * dS + i0;
    device const float * k = convOutN + a.keyDim + hk * dS + i0;
    device const float * v = convOutN + 2 * a.keyDim + hv * dS + j;
    device const float * g = gN + hv;
    device const float * bt = betaN + hv;
    float kk[KS], qq[KS];
    gdn_scan_slice<KS>(k, q, kk, qq);
    float vv = *v;
    float gg = *g;
    float bb = *bt;
    for (uint n = 0; n < a.N; n++) {
        const uint step = n + 1 < a.N ? 1 : 0;
        q += step * a.convDim;
        k += step * a.convDim;
        v += step * a.convDim;
        g += step * a.nV;
        bt += step * a.nV;
        float kn[KS], qn[KS];
        gdn_scan_slice<KS>(k, q, kn, qn);
        const float vn = *v;
        const float gn = *g;
        const float bn = *bt;
        const float gamma = exp(gg);
        float ksum = 0.0f;
        #pragma clang loop unroll(full)
        for (ushort m = 0; m < KS; m++) {
            ls[m] *= gamma;
            ksum += ls[m] * kk[m];
        }
        const float sk = simd_sum(ksum);
        const float d = bb * (vv - sk);
        float osum = 0.0f;
        #pragma clang loop unroll(full)
        for (ushort m = 0; m < KS; m++) {
            ls[m] += kk[m] * d;
            osum += ls[m] * qq[m];
        }
        const float o = simd_sum(osum) * a.qScale;
        if (tiisg == 0) { oN[n * a.valueDim + hv * dS + j] = o; }
        if (a.slots > 1 || n + 1 == a.N) {
            device float * So = S
                + (ulong) ((a.slot0 + 1 + n) % a.slots) * a.stateElems
                + hv * dS * dS;
            #pragma clang loop unroll(full)
            for (ushort m = 0; m < KS; m++) {
                So[(i0 + m) * dS + j] = ls[m];
            }
        }
        #pragma clang loop unroll(full)
        for (ushort m = 0; m < KS; m++) { kk[m] = kn[m]; qq[m] = qn[m]; }
        vv = vn;
        gg = gn;
        bb = bn;
    }
}

#define GDN_SCAN_KERNEL(NAME, KS)                                       \
kernel void NAME(                                                       \
        device const float * convOutN [[buffer(0)]],                    \
        device const float * gN       [[buffer(1)]],                    \
        device const float * betaN    [[buffer(2)]],                    \
        device       float * S        [[buffer(3)]],                    \
        device       float * oN       [[buffer(4)]],                    \
        constant ScanBatchArgs & a    [[buffer(5)]],                    \
        uint3  tgpig [[threadgroup_position_in_grid]],                  \
        ushort sgitg [[simdgroup_index_in_threadgroup]],                \
        ushort tiisg [[thread_index_in_simdgroup]]) {                   \
    gdn_scan_impl<KS>(convOutN, gN, betaN, S, oN, a, tgpig, sgitg, tiisg); \
}

GDN_SCAN_KERNEL(gdn_scan_batch_k2, 2)
GDN_SCAN_KERNEL(gdn_scan_batch_k4, 4)
GDN_SCAN_KERNEL(gdn_scan_batch_k8, 8)

kernel void split_qgate_batch(
        device const float * qFullN [[buffer(0)]],
        device       float * qN     [[buffer(1)]],
        device       float * gateN  [[buffer(2)]],
        constant SplitArgs & a      [[buffer(3)]],
        constant uint      & N      [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    const uint per = a.hd * a.nH;
    if (gid < N * per) {
        const uint n = gid / per;
        const uint r = gid % per;
        const uint h = r / a.hd;
        const uint i = r % a.hd;
        const uint src = n * a.hd * 2 * a.nH + h * a.hd * 2;
        qN[n * per + r]    = qFullN[src + i];
        gateN[n * per + r] = qFullN[src + a.hd + i];
    }
}

// Token n is at absolute position basePos + n.
struct RopeBatchArgs {
    uint headDim; uint nHead; uint nRot; float base; uint basePos; uint N;
};

kernel void rope_batch(
        device float          * x [[buffer(0)]],
        constant RopeBatchArgs & a [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const uint hf = a.nRot / 2;
    const uint per = a.nHead * hf;
    if (gid < a.N * per) {
        const uint n = gid / per;
        const uint r = gid % per;
        const uint h = r / hf;
        const uint i = r % hf;
        const uint hoff = n * a.nHead * a.headDim + h * a.headDim;
        const float freq = pow(a.base, -2.0f * (float) i / (float) a.nRot);
        const float ang = (float) (a.basePos + n) * freq;
        rope_pair(x, hoff + i, hoff + i + hf, cos(ang), sin(ang));
    }
}

// TRAP: same pairing as rope_gemma, NOT rope_batch's.
struct RopeGemmaBatchArgs {
    uint headDim; uint nHead; uint rotated; float base; uint basePos; uint N;
};

kernel void rope_gemma_batch(
        device float                 * x [[buffer(0)]],
        constant RopeGemmaBatchArgs  & a [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const uint per = a.nHead * a.rotated;
    if (gid < a.N * per) {
        const uint n = gid / per;
        const uint r = gid % per;
        const uint hf = a.headDim / 2;
        const uint h = r / a.rotated;
        const uint j = r % a.rotated;
        const uint hoff = n * a.nHead * a.headDim + h * a.headDim;
        const float freq = pow(a.base, -2.0f * (float) j / (float) a.headDim);
        const float ang = (float) (a.basePos + n) * freq;
        rope_pair(x, hoff + j, hoff + j + hf, cos(ang), sin(ang));
    }
}

// Per-token 3D positions (t,h,w); text has t==h==w and degenerates to
// rope_batch. Frequency i takes component i % 3 (HF apply_interleaved_mrope).
struct RopeMBatchArgs {
    uint headDim; uint nHead; uint nRot; float base; uint N;
};

kernel void rope_mrope_batch(
        device float           * x    [[buffer(0)]],
        device const int       * pos3 [[buffer(1)]],   // [N][3] t,h,w
        constant RopeMBatchArgs & a   [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const uint hf = a.nRot / 2;
    const uint per = a.nHead * hf;
    if (gid < a.N * per) {
        const uint n = gid / per;
        const uint r = gid % per;
        const uint h = r / hf;
        const uint i = r % hf;
        const uint hoff = n * a.nHead * a.headDim + h * a.headDim;
        const float freq = pow(a.base, -2.0f * (float) i / (float) a.nRot);
        const float ang = (float) pos3[n * 3 + (i % 3)] * freq;
        rope_pair(x, hoff + i, hoff + i + hf, cos(ang), sin(ang));
    }
}

// token n -> position basePos+n -> page (basePos+n)/P, slot (basePos+n)%P.
struct KVBatchArgs { uint kvDim; uint basePos; uint P; uint N; };

kernel void kv_append_batch(
        device const float   * kCurN [[buffer(0)]],
        device const float   * vCurN [[buffer(1)]],
        device const KVTable & kT    [[buffer(2)]],
        device const KVTable & vT    [[buffer(3)]],
        constant KVBatchArgs & a     [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.N * a.kvDim) {
        const uint n = gid / a.kvDim;
        const uint c = gid % a.kvDim;
        const uint pos = a.basePos + n;
        device half * K = (device half *) kT.pages[pos / a.P];
        device half * V = (device half *) vT.pages[pos / a.P];
        const uint slot = pos % a.P;
        K[slot * a.kvDim + c] = (half) kCurN[n * a.kvDim + c];
        V[slot * a.kvDim + c] = (half) vCurN[n * a.kvDim + c];
    }
}

// `blk` holds each chunk ROW's vision block as absolute (lo, hi), 0/0 for
// none, read up to FA_Q rows past N, so the host sizes it past the batch.
struct AttnBatchArgs {
    uint hd; uint nH; uint nKV; uint kvDim; uint P; float scale;
    uint basePos; uint N; uint gated; uint window;
};

// A query inside a vision block reads to the block's end; the union stays
// CONTIGUOUS because the block contains q.
inline uint attn_end(uint q, uint T, uint blockLo, uint blockHi) {
    return (q >= blockLo && q < blockHi) ? max(T, blockHi) : T;
}

kernel void attn_batch(
        device const float   * qN    [[buffer(0)]],
        device const KVTable & kT    [[buffer(1)]],
        device const KVTable & vT    [[buffer(2)]],
        device const float   * gateN [[buffer(3)]],
        device       float   * outN  [[buffer(4)]],
        constant AttnBatchArgs & a   [[buffer(5)]],
        device const uint    * blk   [[buffer(6)]],
        threadgroup float    * shmem [[threadgroup(0)]],
        uint   tgid  [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint n = tgid / a.nH;           // query token in the batch
    const uint h = tgid % a.nH;
    const uint T = a.basePos + n + 1;     // causal length for this token
    // Sliding layers read a WINDOW, each query row its own start; 0 =
    // unwindowed.
    const uint lo = a.window == 0 || T <= a.window ? 0 : T - a.window;
    // attn_flash reads [lo, hi) with no further mask: the RANGE is the mask.
    const uint hi = attn_end(a.basePos + n, T, blk[2 * n], blk[2 * n + 1]);
    const uint group = a.nH / a.nKV;
    const uint off = n * a.nH * a.hd + h * a.hd;
    attn_flash(qN + off, kT, vT, gateN + off, outN + off,
               a.hd, h / group, a.kvDim, a.P, a.scale, a.gated, lo, hi,
               shmem, sgitg, tiisg);
}

// The four simdgroups split the KEY range for the scores and the HEAD DIM for
// the context; all that is not a matmul is the scalar kernel's arithmetic.
#define FA_Q 8
#define FA_K 32

// K is loaded TRANSPOSED, turning a [key][dim] tile into the [dim][key]
// operand.
inline simdgroup_float8x8 fa_scores(threadgroup const half * qs, uint hd,
                                    device const half * kBase, uint kvDim) {
    simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint d = 0; d < hd; d += 8) {
        simdgroup_half8x8 qa, kb;
        simdgroup_load(qa, qs + d, hd, ulong2(0, 0), false);
        simdgroup_load(kb, kBase + d, kvDim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s, qa, kb, s);
    }
    return s;
}

// A simdgroup matrix carries no scalar multiply, so the correction rides a
// DIAGONAL matrix: one extra product per dim tile.
inline void fa_rescale(thread simdgroup_float8x8 * o, ushort tiles,
                       threadgroup const half * diag) {
    simdgroup_half8x8 d;
    simdgroup_load(d, diag, 8, ulong2(0, 0), false);
    for (ushort i = 0; i < tiles; i++) {
        simdgroup_float8x8 t = make_filled_simdgroup_matrix<float, 8>(0.f);
        simdgroup_multiply_accumulate(t, d, o[i], t);
        o[i] = t;
    }
}

// A short tail is zero-filled so the product reads a whole 8-row block.
inline void fa_stage_q(threadgroup half * qs, device const float * qN,
                       uint base, uint hd, uint row, ushort nq, ushort tiitg) {
    for (uint i = tiitg; i < FA_Q * hd; i += 128) {
        const uint q = i / hd;
        qs[i] = q < nq ? (half) qN[base + q * row + i % hd] : 0.0h;
    }
}

// MASKED per ROW: rows of one tile can differ in causal end, window start
// and vision block. Rows past nq still read their `blk` pair.
inline void fa_mask(threadgroup float * s, uint t0, uint T0, uint window,
                    device const uint * blk, uint n0,
                    float scale, ushort nq, ushort tiitg) {
    for (uint i = tiitg; i < FA_Q * FA_K; i += 128) {
        const uint q = i / FA_K;
        const uint t = t0 + i % FA_K;
        const uint Tq = T0 + q;
        const uint loQ = (window == 0 || Tq <= window) ? 0 : Tq - window;
        const uint r = 2 * (n0 + q);
        const uint hiQ = attn_end(Tq - 1, Tq, blk[r], blk[r + 1]);
        s[i] = (q < nq && t < hiQ && t >= loQ) ? s[i] * scale : -INFINITY;
    }
}

inline void fa_softmax(threadgroup float * s, threadgroup half * p,
                       threadgroup float * mRow, threadgroup float * lRow,
                       threadgroup half * cDiag, ushort tiitg) {
    for (uint q = tiitg; q < FA_Q; q += 128) {
        float mx = mRow[q];
        for (ushort k = 0; k < FA_K; k++) {
            mx = max(mx, s[q * FA_K + k]);
        }
        // A wholly masked tile leaves the row untouched; -INF minus -INF is a
        // NaN that would poison the context for the rest of the sweep.
        const float corr = mx > -INFINITY ? exp(mRow[q] - mx) : 1.0f;
        float sum = 0.0f;
        for (ushort k = 0; k < FA_K; k++) {
            const float e =
                s[q * FA_K + k] > -INFINITY ? exp(s[q * FA_K + k] - mx) : 0.0f;
            p[q * FA_K + k] = (half) e;
            sum += e;
        }
        mRow[q] = mx;
        lRow[q] = lRow[q] * corr + sum;
        cDiag[q * 8 + q] = (half) corr;
    }
}

inline void fa_context(thread simdgroup_float8x8 * o, ushort oTiles,
                       threadgroup const half * p,
                       device const KVTable & vT, uint t0, uint tEnd,
                       uint P, uint kvDim, uint kvh, uint hd, uint slice) {
    for (ushort kb = 0; kb < FA_K / 8; kb++) {
        const uint t = t0 + kb * 8;
        if (t < tEnd) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, p + kb * 8, FA_K, ulong2(0, 0), false);
            device const half * vb = kv_row(vT, t, P, kvDim, kvh, hd) + slice;
            for (ushort i = 0; i < oTiles; i++) {
                simdgroup_half8x8 vv;
                simdgroup_load(vv, vb + i * 8, kvDim, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(o[i], pm, vv, o[i]);
            }
        }
    }
}

// Through threadgroup memory, the only place with the [row][dim] layout; each
// simdgroup lands in its OWN 64-float window, since all four store at once.
inline void fa_store(thread simdgroup_float8x8 * o, ushort oTiles,
                     threadgroup float * mine, threadgroup const float * lRow,
                     device float * outN, device const float * gateN,
                     uint base, uint row, uint hd, uint slice, uint gated,
                     ushort nq, ushort tiisg) {
    for (ushort i = 0; i < oTiles; i++) {
        simdgroup_store(o[i], mine, 8, ulong2(0, 0), false);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort j = tiisg; j < FA_Q * 8; j += 32) {
            const uint q = j / 8;
            if (q < nq) {
                const uint at = base + q * row + slice + i * 8 + j % 8;
                const float v = mine[j] / lRow[q];
                outN[at] = gated ? v * sigmoidf(gateN[at]) : v;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void attn_batch_mm(
        device const float   * qN    [[buffer(0)]],
        device const KVTable & kT    [[buffer(1)]],
        device const KVTable & vT    [[buffer(2)]],
        device const float   * gateN [[buffer(3)]],
        device       float   * outN  [[buffer(4)]],
        constant AttnBatchArgs & a   [[buffer(5)]],
        device const uint    * blk   [[buffer(6)]],
        threadgroup uchar    * shmem [[threadgroup(0)]],
        uint   tgid  [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint hd = a.hd;
    const uint slice = sgitg * (hd / 4);       // this simdgroup's dim slice
    const ushort oTiles = (ushort) (hd / 4 / 8);
    const uint n0 = (tgid / a.nH) * FA_Q;
    const uint h = tgid % a.nH;
    const ushort nq = (ushort) min((uint) FA_Q, a.N - n0);
    const uint kvh = h / (a.nH / a.nKV);
    const uint row = a.nH * hd;                // one token's q / out stride
    const uint base = n0 * row + h * hd;
    // shmem = qs[FA_Q*hd half] | s[FA_Q*FA_K float] | p[FA_Q*FA_K half]
    //       | m[FA_Q] | l[FA_Q] | corr[64 half, a diagonal]
    threadgroup half  * qs    = (threadgroup half *) shmem;
    threadgroup float * sBuf  = (threadgroup float *) (qs + FA_Q * hd);
    threadgroup half  * pBuf  = (threadgroup half *) (sBuf + FA_Q * FA_K);
    threadgroup float * mRow  = (threadgroup float *) (pBuf + FA_Q * FA_K);
    threadgroup float * lRow  = mRow + FA_Q;
    threadgroup half  * cDiag = (threadgroup half *) (lRow + FA_Q);
    fa_stage_q(qs, qN, base, hd, row, nq, tiitg);
    for (uint i = tiitg; i < FA_Q; i += 128) {
        mRow[i] = -INFINITY;
        lRow[i] = 0.0f;
    }
    // The correction matrix is DIAGONAL, so its off-diagonal must be zero and
    // stay zero, which is why each pass clears only the cell it wrote.
    for (uint i = tiitg; i < FA_Q * 8; i += 128) { cDiag[i] = 0.0h; }
    simdgroup_float8x8 o[16];
    for (ushort i = 0; i < oTiles; i++) {
        o[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    // The union of the LIVE rows' ranges; rows past nq are excluded, since
    // their `blk` pair is padding and would sweep keys no live row can reach.
    const uint tLast = a.basePos + n0 + nq;
    uint tEnd = tLast;
    for (ushort q = 0; q < nq; q++) {
        const uint r = 2 * (n0 + q);
        const uint qp = a.basePos + n0 + q;
        tEnd = max(tEnd, attn_end(qp, qp + 1, blk[r], blk[r + 1]));
    }
    const uint T0 = a.basePos + n0 + 1;
    const uint loF = (a.window == 0 || T0 <= a.window) ? 0 : T0 - a.window;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t0 = loF / FA_K * FA_K; t0 < tEnd; t0 += FA_K) {
        // Scores: simdgroup s owns keys t0 + 8s .. +8.
        const uint kt = t0 + sgitg * 8;
        const simdgroup_float8x8 sBlk = kt < tEnd
            ? fa_scores(qs, hd, kv_row(kT, kt, a.P, a.kvDim, kvh, hd), a.kvDim)
            : make_filled_simdgroup_matrix<float, 8>(0.f);
        simdgroup_store(sBlk, sBuf + sgitg * 8, FA_K, ulong2(0, 0), false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fa_mask(sBuf, t0, T0, a.window, blk, n0, a.scale, nq, tiitg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fa_softmax(sBuf, pBuf, mRow, lRow, cDiag, tiitg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fa_rescale(o, oTiles, cDiag);
        fa_context(o, oTiles, pBuf, vT, t0, tEnd, a.P, a.kvDim, kvh, hd,
                   slice);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tiitg; i < FA_Q * 8; i += 128) { cDiag[i] = 0.0h; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    fa_store(o, oTiles, sBuf + sgitg * 64, lRow, outN, gateN, base, row, hd,
             slice, a.gated, nq, tiisg);
}

// Each mirrors the CPU ViT (Qwen/QwenViT.swift) op for op; that is the oracle.

// W a plain row-major [M][K] half buffer. Unlike the quantized GEMMs it needs
// a K-tail guard: ViT K values are NOT all multiples of 32.
struct F16WArgs { uint K; uint M; };

kernel void f16w_gemm_mm(
        device const half  * A       [[buffer(0)]],
        device const float * X       [[buffer(1)]],
        device       float * dst     [[buffer(2)]],
        constant F16WArgs  & a       [[buffer(3)]],
        constant uint      & N       [[buffer(4)]],
        threadgroup uchar  * shmem   [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int K = (int) a.K, M = (int) a.M;
    const int NR0 = 64, NR1 = 32, NK = 32, NL0 = NK / 16, NL1 = NK / 8;
    threadgroup half * sa = (threadgroup half *) (shmem);
    threadgroup half * sb = (threadgroup half *) (shmem + 4096);
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;
    const short nr0 = (M - r0 < NR0) ? (short) (M - r0) : NR0;
    const short nr1 = ((int) N - r1 < NR1) ? (short) ((int) N - r1) : NR1;
    const short lr0 = ((short) (tiitg / NL0) < nr0) ? (short) (tiitg / NL0)
                                                    : nr0 - 1;
    const short lr1 = ((short) (tiitg / NL1) < nr1) ? (short) (tiitg / NL1)
                                                    : nr1 - 1;
    const short il0 = tiitg % NL0;
    device const half  * arow = A + (ulong) (r0 + lr0) * K;
    device const float * yrow = X + (ulong) (r1 + lr1) * K;
    simdgroup_float8x8 mc[8];
    #pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const int k = loop_k + il0 * 16 + i;
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            sa[64 * (8 * sx + sy) + 8 * ly + lx] =
                k < K ? arow[k] : (half) 0.0h;
        }
        {
            const short sx = tiitg % NL1;
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short iy = 8 * sx;
            threadgroup half * bp = sb + 64 * (4 * sx + sy) + 8 * ly;
            for (short i = 0; i < 8; i++) {
                const int k = loop_k + iy + i;
                bp[i] = k < K ? (half) yrow[k] : (half) 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simd_mm_slice(sa, sb, mc, sgitg);
    }
    store_mm_tile(mc, dst, shmem, r0, r1, M, (int) N, nr0, nr1,
                  tiitg, sgitg);
}

// Pre-LN with a learned bias, unlike the LM's rmsnorm; x -> y so x survives
// for the residual. TWO reductions, hence the disjoint shmem halves.
struct LNArgs { uint n; float eps; };

kernel void vit_layernorm(
        device const float * x     [[buffer(0)]],
        device const float * w     [[buffer(1)]],
        device const float * b     [[buffer(2)]],
        device       float * y     [[buffer(3)]],
        constant LNArgs    & a     [[buffer(4)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint  row  [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]],
        uint  sgi  [[simdgroup_index_in_threadgroup]],
        uint  tii  [[thread_index_in_simdgroup]]) {
    device const float * xr = x + (ulong) row * a.n;
    device       float * yr = y + (ulong) row * a.n;
    float s1 = 0.0f, s2 = 0.0f;
    for (uint i = tid; i < a.n; i += ntg) {
        const float v = xr[i];
        s1 += v;
        s2 += v * v;
    }
    // Disjoint halves, because tg_reduce_sum clobbers what it is handed; two
    // calls cost two extra barriers, the arithmetic is untouched.
    const float sum = tg_reduce_sum(s1, shmem, ntg, sgi, tii);
    const float sumsq = tg_reduce_sum(s2, shmem + 32, ntg, sgi, tii);
    const float mean = sum / (float) a.n;
    const float inv =
        1.0f / sqrt(sumsq / (float) a.n - mean * mean + a.eps);
    for (uint i = tid; i < a.n; i += ntg) {
        yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
    }
}

kernel void add_bias_rows(device float * x [[buffer(0)]],
                          device const float * bias [[buffer(1)]],
                          constant uint & m [[buffer(2)]],
                          constant uint & total [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid < total) { x[gid] += bias[gid % m]; }
}

// ggml's tanh-approximation GELU (matches ViT.gelu / vvtanhf numerics).
kernel void gelu_tanh(device float * x [[buffer(0)]],
                      constant uint & n [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < n) {
        const float v = x[gid];
        const float t =
            precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v));
        x[gid] = 0.5f * v * (1.0f + t);
    }
}

// cos/sin precomputed per (slot, pair); `off` selects q (0) or k (e).
struct VRopeArgs {
    uint rowStride; uint off; uint headDim; uint nHead; uint N;
};

kernel void vit_rope(
        device float       * x    [[buffer(0)]],
        device const float * cosT [[buffer(1)]],
        device const float * sinT [[buffer(2)]],
        constant VRopeArgs & a    [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const uint hf = a.headDim / 2;
    const uint per = a.nHead * hf;
    if (gid < a.N * per) {
        const uint s = gid / per;
        const uint r = gid % per;
        const uint h = r / hf;
        const uint j = r % hf;
        const uint base = s * a.rowStride + a.off + h * a.headDim;
        rope_pair(x, base + j, base + j + hf, cosT[s * hf + j],
                  sinT[s * hf + j]);
    }
}

// No causal mask, no GQA. The 4 x 32 lane slice caps hd at 128; the host
// asserts.
struct VAttnArgs { uint n; uint e; uint hd; uint nHead; float scale; };

inline void vit_stage_kv(device const float * qkv,
                         threadgroup float * kt, threadgroup float * vt,
                         uint t0, uint tk, uint h, uint row,
                         uint e, uint hd, uint tid, uint ntg) {
    for (uint i = tid; i < tk * hd; i += ntg) {
        const uint t = i / hd, j = i % hd;
        kt[i] = qkv[(ulong) (t0 + t) * row + e + h * hd + j];
        vt[i] = qkv[(ulong) (t0 + t) * row + 2 * e + h * hd + j];
    }
}

// TRAP: vectorize only on the hd % 4 == 0 prefix, which keeps the float4
// loads 16B-aligned. The scalar tail covers any other geometry.
inline float vit_score(threadgroup const float * qrow,
                       threadgroup const float * krow, uint hd) {
    const uint hd4 = hd & ~3u;
    threadgroup const float4 * q4 = (threadgroup const float4 *) qrow;
    threadgroup const float4 * k4 = (threadgroup const float4 *) krow;
    float4 p4 = 0.0f;
    for (uint i = 0; i < hd4 / 4; i++) {
        p4 += q4[i] * k4[i];
    }
    float p = p4.x + p4.y + p4.z + p4.w;
    for (uint i = hd4; i < hd; i++) {
        p += qrow[i] * krow[i];
    }
    return p;
}

kernel void vit_attn(
        device const float * qkv   [[buffer(0)]],
        device       float * out   [[buffer(1)]],
        constant VAttnArgs & a     [[buffer(2)]],
        threadgroup float  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  tpitg [[thread_position_in_threadgroup]],
        uint3  ntg3  [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint tid = tpitg.x, ntg = ntg3.x;
    const uint TK = 32;                       // keys per streamed tile
    const uint h = tgpig.y;
    const uint s = tgpig.x * (ntg / 32) + sgitg;
    const uint row = 3 * a.e;
    threadgroup float * kt = shmem;           // [TK][hd]
    threadgroup float * vt = shmem + TK * a.hd;
    threadgroup float * qs = shmem + 2 * TK * a.hd;   // [ntg/32][hd]
    float acc[4] = { 0, 0, 0, 0 };            // lane owns dims tiisg + 32u
    if (s < a.n) {
        device const float * qh = qkv + (ulong) s * row + h * a.hd;
        for (uint j = tiisg; j < a.hd; j += 32) {
            qs[sgitg * a.hd + j] = qh[j];
        }
    }
    float m = -INFINITY, l = 0.0f;
    for (uint t0 = 0; t0 < a.n; t0 += TK) {
        const uint tk = min(TK, a.n - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        vit_stage_kv(qkv, kt, vt, t0, tk, h, row, a.e, a.hd, tid, ntg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sc = -INFINITY;
        if (s < a.n && tiisg < tk) {
            const float p = vit_score(qs + sgitg * a.hd,
                                     kt + tiisg * a.hd, a.hd);
            sc = p * a.scale;
        }
        const float mNew = max(m, simd_max(sc));
        const float corr = exp(m - mNew);     // 0 on the first tile
        const float w = sc > -INFINITY ? exp(sc - mNew) : 0.0f;
        l = l * corr + simd_sum(w);
        for (ushort u = 0; u < 4; u++) { acc[u] *= corr; }
        for (uint t = 0; t < tk; t++) {
            const float wt = simd_broadcast(w, (ushort) t);
            // Fixed 4-slice guarded loop, NOT a hoisted per-lane bound: the
            // compile-time count unrolls; a runtime bound measured 25% slower.
            for (ushort u = 0; u < 4; u++) {
                const uint j = tiisg + 32 * u;
                if (j < a.hd) { acc[u] += wt * vt[t * a.hd + j]; }
            }
        }
        m = mNew;
    }
    if (s < a.n) {
        device float * o = out + (ulong) s * a.e + h * a.hd;
        const float inv = 1.0f / l;
        for (ushort u = 0; u < 4; u++) {
            const uint j = tiisg + 32 * u;
            if (j < a.hd) { o[j] = acc[u] * inv; }
        }
    }
}

kernel void silu_inplace(device float * x [[buffer(0)]],
                         constant uint & n [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < n) { x[gid] = siluf(x[gid]); }
}

// x[i] += y[i] * s -- the macaron halves add at HALF weight, not full.
struct FmaArgs { uint n; float s; };

kernel void fma_inplace(device float * x [[buffer(0)]],
                        device const float * y [[buffer(1)]],
                        constant FmaArgs & a [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < a.n) { x[gid] += y[gid] * a.s; }
}

struct GluArgs { uint n; uint e; };

kernel void glu_rows(device const float * src [[buffer(0)]],
                     device       float * dst [[buffer(1)]],
                     constant GluArgs   & a   [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < a.n * a.e) {
        const uint t = gid / a.e;
        const uint c = gid % a.e;
        const uint row = t * 2 * a.e;
        dst[gid] = src[row + c] * sigmoidf(src[row + a.e + c]);
    }
}

// Causal depthwise conv: each channel convolves only its own history.
struct DwArgs { uint n; uint e; uint k; };

kernel void depthwise_causal(device const float * x   [[buffer(0)]],
                             device const float * w   [[buffer(1)]],
                             device       float * out [[buffer(2)]],
                             constant DwArgs    & a   [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid < a.n * a.e) {
        const uint t = gid / a.e;
        const uint c = gid % a.e;
        float acc = 0.0f;
        for (uint j = 0; j < a.k; j++) {
            const int src = (int) t - (int) (a.k - 1) + (int) j;
            if (src >= 0) { acc += x[(uint) src * a.e + c] * w[c * a.k + j]; }
        }
        out[gid] = acc;
    }
}

// The audio query is scaled per head_dim by a learned, softplus'd factor.
struct HeadScaleArgs { uint n; uint e; uint hd; };

kernel void scale_head_dims(device       float * x [[buffer(0)]],
                            device const float * s [[buffer(1)]],
                            constant HeadScaleArgs & a [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid < a.n * a.e) { x[gid] *= s[(gid % a.e) % a.hd]; }
}

// Transformer-XL relative attention over blocked local windows. TRAP: the
// window holds `chunk` positions INCLUDING self, so `rel` runs 1..past.
struct AAttnArgs {
    uint n; uint e; uint heads; uint hd; uint chunk; uint ctx; uint past;
    float cap;
};

// Both loops need it; the mask rule was written twice.
inline int audio_key_at(uint b, uint o, uint chunk, uint past) {
    return (int) (b * chunk + o) - (int) past;
}

inline bool audio_key_live(int kAbs, uint n) {
    return kAbs >= 0 && kAbs < (int) n;
}

inline float audio_scores(device const float * q, device const float * k,
                          device const float * relk,
                          thread float (&sc)[32],
                          uint qo, uint qi, uint b, uint h,
                          constant AAttnArgs & a) {
    float mx = -INFINITY;
    for (uint o = 0; o < a.ctx; o++) {
        const int kAbs = audio_key_at(b, o, a.chunk, a.past);
        const int rel = (int) o - (int) qi;
        const bool ok = audio_key_live(kAbs, a.n)
            && rel >= 1 && rel <= (int) a.past;
        float s = -INFINITY;
        if (ok) {
            const uint ko = (uint) kAbs * a.e + h * a.hd;
            const uint ro = (uint) rel * a.e + h * a.hd;
            float ac = 0.0f, bd = 0.0f;
            for (uint c = 0; c < a.hd; c++) {
                ac += q[qo + c] * k[ko + c];
                bd += q[qo + c] * relk[ro + c];
            }
            s = precise::tanh((ac + bd) / a.cap) * a.cap;
        }
        sc[o] = s;
        mx = max(mx, s);
    }
    return mx;
}

inline void audio_context(device const float * v, device float * out,
                          thread const float (&sc)[32], float inv,
                          uint qo, uint b, uint h,
                          constant AAttnArgs & a) {
    for (uint c = 0; c < a.hd; c++) {
        out[qo + c] = 0.0f;
    }
    for (uint o = 0; o < a.ctx; o++) {
        const int kAbs = audio_key_at(b, o, a.chunk, a.past);
        if (sc[o] > 0.0f && audio_key_live(kAbs, a.n)) {
            const uint vo = (uint) kAbs * a.e + h * a.hd;
            const float wgt = sc[o] * inv;
            for (uint c = 0; c < a.hd; c++) {
                out[qo + c] += wgt * v[vo + c];
            }
        }
    }
}

kernel void gemma_audio_attn(
        device const float * q    [[buffer(0)]],
        device const float * k    [[buffer(1)]],
        device const float * v    [[buffer(2)]],
        device const float * relk [[buffer(3)]],
        device       float * out  [[buffer(4)]],
        constant AAttnArgs  & a   [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < a.n * a.heads) {
        const uint t = gid / a.heads;
        const uint h = gid % a.heads;
        const uint b = t / a.chunk;          // which blocked window
        const uint qi = t % a.chunk;         // this query's slot in it
        const uint qo = t * a.e + h * a.hd;
        float sc[32];
        const float mx = audio_scores(q, k, relk, sc, qo, qi, b, h, a);
        float sum = 0.0f;
        for (uint o = 0; o < a.ctx; o++) {
            sc[o] = sc[o] > -INFINITY ? exp(sc[o] - mx) : 0.0f;
            sum += sc[o];
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        audio_context(v, out, sc, inv, qo, b, h, a);
    }
}

// Part of the ARITHMETIC the QAT trained with. `rint` rounds half to EVEN to
// match torch.round; anything else drifts on the exact .5 the scales hit.
struct SrqArgs { uint n; float s; float lo; float hi; };

// A nonzero scale is the mobile lineage's fake-quant; otherwise the export's
// literal two-sided bound, which does NOT round.
inline float srq_bound(float v, constant SrqArgs & a) {
    return a.s != 0.0f ? clamp(rint(v / a.s), -128.0f, 127.0f) * a.s
                       : clamp(v, a.lo, a.hi);
}

kernel void srq_inplace(device float * x [[buffer(0)]],
                        constant SrqArgs & a [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < a.n) {
        x[gid] = srq_bound(x[gid], a);
    }
}

kernel void srq_to(device const float * src [[buffer(0)]],
                   device       float * dst [[buffer(1)]],
                   constant SrqArgs   & a   [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid < a.n) {
        dst[gid] = srq_bound(src[gid], a);
    }
}
