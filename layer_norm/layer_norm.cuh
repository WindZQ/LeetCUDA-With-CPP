#ifndef LAYER_NORM_CUH
#define LAYER_NORM_CUH

#ifdef __cplusplus
extern "C"
{
#endif

void layer_norm_f32(float* x, float* y, float g, float b, int N, int K, cudaStream_t stream);
void layer_norm_f32x4(float* x, float* y, float g, float b, int N, int K, cudaStream_t stream);
void layer_norm_f16_f16(half* x, half* y, float g, float b, int N, int K, cudaStream_t stream);
void layer_norm_f16_f32(half* x, half* y, float g, float b, int N, int K, cudaStream_t stream);
void layer_norm_f16x2_f16(half* x, half* y, float g, float b, int N, int K, cudaStream_t stream);
void layer_norm_f16x8_f16(half* x, half* y, float g, float b, int N, int K, cudaStream_t stream);
void layer_norm_f16x8_pack_f16(half* x, half* y, float g, float b, int N, int K, cudaStream_t stream);
void layer_norm_f16x8_pack_f32(half* x, half* y, float g, float b, int N, int K, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif