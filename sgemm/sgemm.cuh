#ifndef SGEMM_CUH
#define SGEMM_CUH

#ifdef __cplusplus
extern "C"
{
#endif

void sgemm_naive_f32(float* a, float* b, float* c, int M, int N, int K);
void sgemm_sliced_k_f32(float* a, float* b, float* c, int M, int N, int K);
void sgemm_t_8x8_sliced_k_f32x4(float* a, float* b, float* c, int M, int N, int K);
void sgemm_t_8x8_sliced_k_f32x4_bcf(float* a, float* b, float* c, int M, int N, int K);
void sgemm_t_8x8_sliced_k_f32x4_bcf_offset(float* a, float* b, float* c, int M, int N, int K);
void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf(float* a, float* b, float* c, int M, int N, int K);
void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset(float* a, float* b, float* c, int M, int N, int K);

#ifdef __cplusplus
}
#endif


#endif 