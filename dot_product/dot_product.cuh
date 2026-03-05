#ifndef DOT_PRODUCT_CUH
#define DOT_PRODUCT_CUH

#ifdef __cplusplus
extern "C"
{
#endif

void dot_prod_f32_f32(const float* a, const float* b, int n, float* out, cudaStream_t stream);
void dot_prod_f32x4_f32(const float* a, const float* b, int n, float* out, cudaStream_t stream);
void dot_prod_f16_f32(const half* a, const half* b, int n, float* out, cudaStream_t stream);
void dot_prod_f16x2_f32(const half* a, const half* b, int n, float* out, cudaStream_t stream);
void dot_prod_f16x8_pack_f32(const half* a, const half* b, int n, float* out, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif 