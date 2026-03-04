#ifndef BLOCK_ALL_REDUCE_CUH
#define BLOCK_ALL_REDUCE_CUH


#ifdef __cplusplus
extern "C"
{
#endif

void block_all_reduce_sum_f32_f32(const float* x, float* y, int64_t S, int64_t K,  cudaStream_t stream);
void block_all_reduce_sum_f32x4_f32(const float* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_f16_f16(const void* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_f16_f32(const void* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_f16x2_f32(const void* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_f16x2_f16(const void* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_f16x8_pack_f16(const void* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_f16x8_pack_f32(const void* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_bf16_bf16(const void* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_bf16_f32(const void* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_bf16x2_f32(const void* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_bf16x2_bf16(const void* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_bf16x8_pack_f32(const void* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_bf16x8_pack_bf16(const void* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_fp8_e4m3_f16(const uint8_t* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_fp8_e4m3x16_pack_f16(const uint8_t* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_fp8_e5m2_f16(const uint8_t* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_fp8_e5m2x16_pack_f16(const uint8_t* x, float* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_i8_i32(const int8_t* x, int32_t* y, int64_t S, int64_t K, cudaStream_t stream);
void block_all_reduce_sum_i8x16_pack_i32(const int8_t* x, int32_t* y, int64_t S, int64_t K, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif 