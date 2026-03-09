#ifndef SOFTMAT_CUH
#define SOFTMAT_CUH

#ifdef __cplusplus
extern "C"
{
#endif

void softmax_f32_per_token(float* x, float* y, int S, int H, cudaStream_t stream);
void softmax_f32x4_per_token(float* x, float* y, int S, int H, cudaStream_t stream);
void safe_softmax_f32_per_token(float* x, float* y, int S, int H, cudaStream_t stream);
void safe_softmax_f32x4_pre_token(float* x, float* y, int S, int H, cudaStream_t stream);
void safe_softmax_f16_f32_per_token(half* x, half* y, int S, int H, cudaStream_t stream);
void safe_softmax_f16x2_f32_per_token(half* x, half* y, int S, int H, cudaStream_t stream);
void safe_softmax_f16x8_pack_f32_per_token(half* x, half* y, int S, int H, cudaStream_t stream);
void online_safe_softmax_f32_per_token(float* x, float* y, int S, int H, cudaStream_t stream);
void online_safe_softmax_f32x4_pack_per_token(float* x, float* y, int S, int H, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif 