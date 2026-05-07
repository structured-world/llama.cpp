#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

// Initialize turbo KV decode constants to identity (all 1.0).
// d_innerq_channel_scale_inv_fattn is a __device__ array — zero-initialized
// by default. Must be set to 1.0 before the first FA call or turbo inv-FWHT
// decode will multiply every K element by 0 → attention collapse ("of of of").
void turbo_innerq_init_fattn();
