#ifndef HGEMV_CUTE_CUH
#define HGEMV_CUTE_CUH

#ifdef __cplusplus
extern "C"
{
#endif

void hgemv_f16_cute(half* A, half* B, half* C, int M, int K);
void hgemv_f16x8_cute(half* A, half* B, half* C, int M, int K);
void hgemv_tensor_core_cute(half* A, half* B, half* C, int M, int K);

#ifdef __cplusplus
}
#endif

#endif 