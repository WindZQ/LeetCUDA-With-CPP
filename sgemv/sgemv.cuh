#ifndef SGEMV_CUH
#define SGEMV_CUH

#ifdef __cplusplus
extern "C"
{
#endif

void sgemv_k32_f32(float* a, float* x, float* y, int M, int K);
void sgemv_k128_f32x4(float* a, float* x, float* y, int M, int K);
void sgemv_k16_f32(float* a, float* x, float* y, int M, int K);

#ifdef __cplusplus
}
#endif

#endif 