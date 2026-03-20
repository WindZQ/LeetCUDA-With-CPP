#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cstdint>

#include "merge_attn_states.cuh"

inline __device__ float to_float(float u) { return u; }
inline __device__ float to_float(half u) { return __half2float(u); }
inline __device__ float to_float(__nv_bfloat16 u) 
{
	return __bfloat162float(u);
}
inline __device__ void from_float(float& d, float s) { d = s; }
inline __device__ void from_float(half& d, float s) { d = __float2half(s); }
inline __device__ void from_float(__nv_bfloat16& d, float s) 
{
	d = __float2bfloat16(s);
}

// Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
// can be used to combine partial attention results (in the split-KV case)
template <typename scalar_t, unsigned int num_threads = 256>
__global__ void merge_attn_states_kernel(scalar_t* output, float* output_lse,
    const scalar_t* prefix_output, const float* prefix_lse,
    const scalar_t* suffix_output, const float* suffix_lse,
    const unsigned int num_tokens, const unsigned int num_heads,
    const unsigned int head_size)
{
    using pack_128b_t = uint4;
    const unsigned int pack_size = 16 / sizeof(scalar_t);
    const unsigned int thread_per_head = head_size / pack_size;

    const unsigned int global_idx = threadIdx.x + num_threads * blockIdx.x;
    const unsigned int token_head_threads = num_tokens * num_heads * thread_per_head;

    if (global_idx >= token_head_threads) return;

    // global_idx -> token_idx + head_idx + pack_idx
    const unsigned int token_head_idx = global_idx / thread_per_head;
    const unsigned int pack_idx = global_idx % thread_per_head;

    const unsigned int token_idx = token_head_idx / num_heads;
    const unsigned int head_idx = token_head_idx % num_heads;

    const unsigned int pack_offset = pack_idx * pack_size; // (0~15)*8, etc
    const unsigned int head_offset = token_idx * num_heads * head_size + head_idx * head_size;
    const scalar_t* prefix_head_ptr = prefix_output + head_offset;
    const scalar_t* suffix_head_ptr = suffix_output + head_offset;
    scalar_t* output_head_ptr = output + head_offset;

    float p_lse = prefix_lse[head_idx * num_tokens + token_idx];
    float s_lse = suffix_lse[head_idx * num_tokens + token_idx];
    if (isinf(p_lse)) p_lse = -CUDART_INF_F;
    if (isinf(s_lse)) s_lse = -CUDART_INF_F;

    const float max_lse = fmaxf(p_lse, s_lse);
    p_lse = p_lse - max_lse;
    s_lse = s_lse - max_lse;
    const float p_se = expf(p_lse);
    const float s_se = expf(s_lse);
    const float out_se = p_se + s_se;
    const float p_scale = p_se / out_se;
    const float s_scale = s_se / out_se;

    if (pack_offset < head_size) {
        // Pack 128b load
        pack_128b_t p_out_pack = reinterpret_cast<const pack_128b_t*>(
            prefix_head_ptr)[pack_offset / pack_size];
        pack_128b_t s_out_pack = reinterpret_cast<const pack_128b_t*>(
            suffix_head_ptr)[pack_offset / pack_size];
        pack_128b_t o_out_pack;

#pragma unroll 
        for (int i = 0; i < pack_size; ++i)
        {
            // Always use float for FMA to keep high precision
            // half(uint16_t), bfloat16, float -> float
            const float p_out_f =
                to_float(reinterpret_cast<const scalar_t*>(&p_out_pack)[i]);
            const float s_out_f =
                to_float(reinterpret_cast<const scalar_t*>(&s_out_pack)[i]);
            // fma: a * b + c = p_out_f * p_scale + (s_out_f * s_scale)
            const float o_out_f = p_out_f * p_scale + (s_out_f * s_scale);
            // float -> half(uint16_t), bfloat16, float
            from_float(reinterpret_cast<scalar_t*>(&o_out_pack)[i], o_out_f);
        }

        // Pack 128b storage
        reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_offset / pack_size] =
            o_out_pack;
    }

}

template <typename scalar_t>
void merge_attn_states_launcher_typed(
    scalar_t* output,
    float* output_lse,
    const scalar_t* prefix_output,
    const float* prefix_lse,
    const scalar_t* suffix_output,
    const float* suffix_lse,
    int num_tokens,
    int num_heads,
    int head_size,
    cudaStream_t stream)
{
    constexpr int NUM_THREADS = 128;
    constexpr int pack_size = 16 / sizeof(scalar_t);

    if (head_size % pack_size != 0) {
        return ;
    }

    const unsigned int threads_per_head = head_size / pack_size;
    const unsigned int total_threads = num_tokens * num_heads * threads_per_head;

    dim3 block(NUM_THREADS);
    dim3 grid((total_threads + NUM_THREADS - 1) / NUM_THREADS);

    merge_attn_states_kernel<scalar_t, NUM_THREADS> << <grid, block, 0, stream >> > (
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        num_tokens,
        num_heads,
        head_size
        );
}

void merge_attn_states(
    void* output,
    float* output_lse,
    const void* prefix_output,
    const float* prefix_lse,
    const void* suffix_output,
    const float* suffix_lse,
    unsigned int num_tokens,
    unsigned int num_heads,
    unsigned int head_size,
    int scalar_type,
    cudaStream_t stream)
{
    switch (scalar_type) {
    case SCALAR_FLOAT:
        merge_attn_states_launcher_typed<float>(
            reinterpret_cast<float*>(output),
            output_lse,
            reinterpret_cast<const float*>(prefix_output),
            prefix_lse,
            reinterpret_cast<const float*>(suffix_output),
            suffix_lse,
            num_tokens,
            num_heads,
            head_size,
            stream
            );
    case SCALAR_HALF:
        merge_attn_states_launcher_typed<__half>(
            reinterpret_cast<__half*>(output),
            output_lse,
            reinterpret_cast<const __half*>(prefix_output),
            prefix_lse,
            reinterpret_cast<const __half*>(suffix_output),
            suffix_lse,
            num_tokens,
            num_heads,
            head_size,
            stream
            );
    case SCALAR_BFLOAT16:
        merge_attn_states_launcher_typed<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16*>(output),
            output_lse,
            reinterpret_cast<const __nv_bfloat16*>(prefix_output),
            prefix_lse,
            reinterpret_cast<const __nv_bfloat16*>(suffix_output),
            suffix_lse,
            num_tokens,
            num_heads,
            head_size,
            stream
            );
    default:
        return ;
    }
}