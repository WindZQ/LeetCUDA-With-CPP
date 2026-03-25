#ifndef HGEMV_CUH
#define HGEMV_CUH

#ifdef __cplusplus
extern "C"
{
#endif

void hgemv_k32_f16(half* a, half* x, half* y, int M, int K);
void hgemv_k128_f16x4(half* a, half* x, half* y, int M, int K);
void hgemv_k16_f16(half* a, half* x, half* y, int M, int K);

#ifdef __cplusplus
}
#endif

#endif 