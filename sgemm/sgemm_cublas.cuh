#ifndef SGEMM_CUBLAS_CUH
#define SGEMM_CUBLAS_CUH

#ifdef __cplusplus
extern "C"
{
#endif

void sgemm_cublas(float* a, float* b, float* c, int M, int N, int K);
void sgemm_cublas_tf32(float* a, float* b, float* c, int M, int N, int K);

#ifdef __cplusplus
}
#endif


#endif 