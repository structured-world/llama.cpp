#include "set-rows.cuh"
#include "cpy-utils.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows<<<grid_size, block_size, 0, stream>>>(src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
                                                         s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
                                                         ne11_fd, ne12_fd);
    }
}


// ── TurboQuant SET_ROWS device quantizers ────────────────────────────────
// K/V values are stored in FWHT-rotated domain so inv-FWHT dequant during FA
// recovers the original vectors, and graph-level ggml_turbo_wht(inv) un-rotates
// the attention output correctly.  Forward FWHT is applied here before quantizing.

// FWHT sign arrays (same values as d_turbo_wht_signs1/2 in turbo-quant-cuda.cuh)
static __constant__ float d_tsr_s1[128] = {
    -1.0f, 1.0f, 1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
     1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f,
     1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f, 1.0f, 1.0f, 1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f,
     1.0f,-1.0f,-1.0f, 1.0f, 1.0f, 1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f, 1.0f, 1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f, 1.0f,
     1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f};
static __constant__ float d_tsr_s2[128] = {
     1.0f, 1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f,-1.0f,-1.0f,
     1.0f, 1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f, 1.0f, 1.0f,
     1.0f, 1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f, 1.0f,-1.0f,
     1.0f,-1.0f, 1.0f, 1.0f, 1.0f,-1.0f,-1.0f, 1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f,
     1.0f,-1.0f, 1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f,-1.0f,
     1.0f,-1.0f, 1.0f, 1.0f, 1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f};
// Midpoints for 3-bit and 2-bit nearest-centroid search
static __constant__ float d_tsr_mid_3bit[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f };
static __constant__ float d_tsr_mid_2bit[3] = {
    -0.086728f, 0.0f, 0.086728f };

// Forward FWHT: s2 ⊙ H(s1 ⊙ x) / sqrt(128), sequential register-only butterfly
static __device__ __forceinline__ void tsr_fwht_forward(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= d_tsr_s1[i];
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; i += h*2)
            for (int j = i; j < i+h; j++) {
                float a = x[j], b = x[j+h];
                x[j] = a+b; x[j+h] = a-b;
            }
    constexpr float inv_sqrt128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] = x[i] * inv_sqrt128 * d_tsr_s2[i];
}

static __device__ __forceinline__ uint8_t tsr_nearest_3bit(float v) {
    if      (v < d_tsr_mid_3bit[0]) return 0;
    else if (v < d_tsr_mid_3bit[1]) return 1;
    else if (v < d_tsr_mid_3bit[2]) return 2;
    else if (v < d_tsr_mid_3bit[3]) return 3;
    else if (v < d_tsr_mid_3bit[4]) return 4;
    else if (v < d_tsr_mid_3bit[5]) return 5;
    else if (v < d_tsr_mid_3bit[6]) return 6;
    else                             return 7;
}
static __device__ __forceinline__ uint8_t tsr_nearest_2bit(float v) {
    if      (v < d_tsr_mid_2bit[0]) return 0;
    else if (v < d_tsr_mid_2bit[1]) return 1;
    else if (v < d_tsr_mid_2bit[2]) return 2;
    else                             return 3;
}

// ── Turbo3 SET_ROWS kernel with FWHT (one thread per 128-element group) ───
template<typename idx_t>
static __global__ void k_turbo3_sr(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo3_0 * __restrict__ dst, const int64_t ne_total_groups,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= ne_total_groups) return;
    const int64_t i_base = i * QK_TURBO3_GROUP;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo3_0 * dst_row_ptr = (block_turbo3_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    const int grp_idx = i00 / QK_TURBO3_GROUP;
    constexpr int BPGRP = QK_TURBO3_GROUP / QK_TURBO3;  // 4 blocks per group
    float x[128]; float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) { x[j] = grp_src[j]; norm_sq += x[j]*x[j]; }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) x[j] *= inv_norm;
    tsr_fwht_forward(x);
    float recon_sq = 0.0f;
    for (int b = 0; b < BPGRP; b++) {
        block_turbo3_0 & blk = dst_row_ptr[grp_idx * BPGRP + b];
        const int off = b * QK_TURBO3;
        for (int j = 0; j < QK_TURBO3/4; j++) blk.qs[j] = 0;
        for (int j = 0; j < QK_TURBO3/8; j++) blk.signs[j] = 0;
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t idx = tsr_nearest_3bit(x[off+j]);
            blk.qs[j/4] |= (idx & 0x3) << ((j%4)*2);
            if (idx & 0x4) blk.signs[j/8] |= (1 << (j%8));
            recon_sq += d_turbo_centroids_3bit_sr[idx] * d_turbo_centroids_3bit_sr[idx];
        }
    }
    float recon_norm = sqrtf(recon_sq);
    float corr = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    for (int b = 0; b < BPGRP; b++)
        dst_row_ptr[grp_idx * BPGRP + b].norm = __float2half(corr);
    GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13);
}

// ── Turbo2 SET_ROWS kernel with FWHT (one thread per 128-element group) ───
template<typename idx_t>
static __global__ void k_turbo2_sr(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo2_0 * __restrict__ dst, const int64_t ne_total_groups,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= ne_total_groups) return;
    const int64_t i_base = i * QK_TURBO2_GROUP;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo2_0 * dst_row_ptr = (block_turbo2_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    const int grp_idx = i00 / QK_TURBO2_GROUP;
    constexpr int BPGRP = QK_TURBO2_GROUP / QK_TURBO2;
    float x[128]; float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) { x[j] = grp_src[j]; norm_sq += x[j]*x[j]; }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) x[j] *= inv_norm;
    tsr_fwht_forward(x);
    float recon_sq = 0.0f;
    for (int b = 0; b < BPGRP; b++) {
        block_turbo2_0 & blk = dst_row_ptr[grp_idx * BPGRP + b];
        const int off = b * QK_TURBO2;
        for (int j = 0; j < QK_TURBO2/4; j++) blk.qs[j] = 0;
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx = tsr_nearest_2bit(x[off+j]);
            blk.qs[j/4] |= (idx & 0x3) << ((j%4)*2);
            recon_sq += d_turbo_centroids_2bit_sr[idx] * d_turbo_centroids_2bit_sr[idx];
        }
    }
    float recon_norm = sqrtf(recon_sq);
    float corr = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    for (int b = 0; b < BPGRP; b++)
        dst_row_ptr[grp_idx * BPGRP + b].norm = __float2half(corr);
    GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13);
}

// Dispatch helpers for FWHT-based turbo set_rows
template<typename idx_t, typename block_t, int GROUP_SIZE, typename KernelFunc>
static void turbo_sr_fwht_dispatch(
        const float * src0_d, const idx_t * src1_d, block_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream, KernelFunc kernel) {
    GGML_ASSERT(ne00 % GROUP_SIZE == 0);
    const int64_t ne_total_groups = (ne00 * ne01 * ne02 * ne03) / GROUP_SIZE;
    if (ne_total_groups <= 0) return;
    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;
    const uint3 ne00_fd = init_fastdiv_values((uint32_t)ne00);
    const uint3 ne01_fd = init_fastdiv_values((uint32_t)ne01);
    const uint3 ne02_fd = init_fastdiv_values((uint32_t)ne02);
    const uint3 ne11_fd = init_fastdiv_values((uint32_t)ne11);
    const uint3 ne12_fd = init_fastdiv_values((uint32_t)ne12);
    const int block_sz = CUDA_SET_ROWS_BLOCK_SIZE;
    const int n_blocks = ((int)ne_total_groups + block_sz - 1) / block_sz;
    kernel<<<n_blocks, block_sz, 0, stream>>>(
        src0_d, src1_d, dst_d, ne_total_groups,
        ne00, ne01, ne02, ne10, ne11, ne12, ne13,
        s01, s02, s03, s10, s11, s12, s1, s2, s3,
        ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
}

static __constant__ float d_turbo_centroids_2bit_sr[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};
static __constant__ float d_turbo_centroids_3bit_sr[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
static __constant__ float d_turbo_centroids_4bit_sr[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};

// turbo2_0: 32 elements/block, 4-centroid 2-bit, 4 indices/byte
static __device__ void quantize_f32_turbo2_0_block(const float * src, block_turbo2_0 * dst) {
    constexpr int QK = QK_TURBO2;  // 32
    float norm_sq = 0.0f;
    for (int i = 0; i < QK; i++) norm_sq += src[i] * src[i];
    float norm = sqrtf(norm_sq);
    float inv  = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    uint8_t indices[QK];
    for (int i = 0; i < QK; i++) {
        float val = src[i] * inv;
        int best = 0;
        float best_d = fabsf(val - d_turbo_centroids_2bit_sr[0]);
        for (int c = 1; c < 4; c++) {
            float d = fabsf(val - d_turbo_centroids_2bit_sr[c]);
            if (d < best_d) { best_d = d; best = c; }
        }
        indices[i] = (uint8_t)best;
    }

    float recon_sq = 0.0f;
    for (int i = 0; i < QK; i++) {
        float r = d_turbo_centroids_2bit_sr[indices[i]];
        recon_sq += r * r;
    }
    float recon_norm = sqrtf(recon_sq);
    dst->norm = __float2half((recon_norm > 1e-10f) ? (norm / recon_norm) : norm);

    // Pack 4 × 2-bit per byte
    for (int i = 0; i < QK; i += 4) {
        dst->qs[i / 4] = (uint8_t)(indices[i] | (indices[i+1]<<2) | (indices[i+2]<<4) | (indices[i+3]<<6));
    }
}

// turbo3_0: 32 elements/block, 8-centroid 3-bit, lower 2 bits in qs, upper 1 in signs
static __device__ void quantize_f32_turbo3_0_block(const float * src, block_turbo3_0 * dst) {
    constexpr int QK = QK_TURBO3;  // 32
    float norm_sq = 0.0f;
    for (int i = 0; i < QK; i++) norm_sq += src[i] * src[i];
    float norm = sqrtf(norm_sq);
    float inv  = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    uint8_t indices[QK];
    for (int i = 0; i < QK; i++) {
        float val = src[i] * inv;
        int best = 0;
        float best_d = fabsf(val - d_turbo_centroids_3bit_sr[0]);
        for (int c = 1; c < 8; c++) {
            float d = fabsf(val - d_turbo_centroids_3bit_sr[c]);
            if (d < best_d) { best_d = d; best = c; }
        }
        indices[i] = (uint8_t)best;
    }

    float recon_sq = 0.0f;
    for (int i = 0; i < QK; i++) {
        float r = d_turbo_centroids_3bit_sr[indices[i]];
        recon_sq += r * r;
    }
    float recon_norm = sqrtf(recon_sq);
    dst->norm = __float2half((recon_norm > 1e-10f) ? (norm / recon_norm) : norm);

    // Lower 2 bits into qs (4 per byte), upper 1 bit into signs (8 per byte)
    for (int i = 0; i < QK; i += 4) {
        dst->qs[i / 4] = (uint8_t)((indices[i]&3) | ((indices[i+1]&3)<<2) | ((indices[i+2]&3)<<4) | ((indices[i+3]&3)<<6));
    }
    for (int i = 0; i < QK; i += 8) {
        uint8_t s = 0;
        for (int j = 0; j < 8; j++) s |= ((indices[i+j] >> 2) & 1) << j;
        dst->signs[i / 8] = s;
    }
}

// turbo4_0: 128 elements/block, 16-centroid 4-bit, 2 per byte (low nibble first)
// QK_TURBO4=128 == group size, so one block = one FWHT group.
static __device__ void quantize_f32_turbo4_0_block(const float * src, block_turbo4_0 * dst) {
    constexpr int QK = QK_TURBO4;  // 128
    float norm_sq = 0.0f;
    for (int i = 0; i < QK; i++) norm_sq += src[i] * src[i];
    float norm = sqrtf(norm_sq);
    float inv  = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    float x[QK];
    for (int i = 0; i < QK; i++) x[i] = src[i] * inv;
    // Forward FWHT rotation before quantization
    tsr_fwht_forward(x);

    uint8_t indices[QK];
    for (int i = 0; i < QK; i++) {
        float val = x[i];
        int best = 0;
        float best_d = fabsf(val - d_turbo_centroids_4bit_sr[0]);
        for (int c = 1; c < 16; c++) {
            float d = fabsf(val - d_turbo_centroids_4bit_sr[c]);
            if (d < best_d) { best_d = d; best = c; }
        }
        indices[i] = (uint8_t)best;
    }

    float recon_sq = 0.0f;
    for (int i = 0; i < QK; i++) {
        float r = d_turbo_centroids_4bit_sr[indices[i]];
        recon_sq += r * r;
    }
    float recon_norm = sqrtf(recon_sq);
    dst->norm = __float2half((recon_norm > 1e-10f) ? (norm / recon_norm) : norm);

    // Pack: low nibble = even index, high nibble = odd index
    for (int i = 0; i < QK; i += 2) {
        dst->qs[i / 2] = (uint8_t)((indices[i+1] << 4) | (indices[i] & 0xF));
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO2_0) {
        // 128-element group kernel with FWHT — cannot use set_rows_cuda_quant (32-elem granularity)
        turbo_sr_fwht_dispatch<idx_t, block_turbo2_0, QK_TURBO2_GROUP>(
            src0_d, src1_d, (block_turbo2_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream, k_turbo2_sr<idx_t>
        );
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        // 128-element group kernel with FWHT — cannot use set_rows_cuda_quant (32-elem granularity)
        turbo_sr_fwht_dispatch<idx_t, block_turbo3_0, QK_TURBO3_GROUP>(
            src0_d, src1_d, (block_turbo3_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream, k_turbo3_sr<idx_t>
        );
    } else if (dst->type == GGML_TYPE_TURBO4_0) {
        // QK_TURBO4=128 == group size; quantize_f32_turbo4_0_block applies FWHT internally
        set_rows_cuda_quant<idx_t, block_turbo4_0, QK_TURBO4, quantize_f32_turbo4_0_block>(
            src0_d, src1_d, (block_turbo4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}
