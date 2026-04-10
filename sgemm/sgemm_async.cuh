#ifndef SGEMM_ASYNC_CUH
#define SGEMM_ASYNC_CUH

#ifdef __cplusplus
extern "C"
{
#endif

void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf(float* a, float* b, float* c, int M, int N, int K);
void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async(float* a, float* b, float* c, int M, int N, int K);
void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf(float* a, float* b, float* c, int M, int N, int K);
void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async(float* a, float* b, float* c, int M, int N, int K);
void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf(float* a, float* b, float* c, int M, int N, int K);
void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async(float* a, float* b, float* c, int M, int N, int K);

#ifdef __cplusplus
}
#endif


#endif 