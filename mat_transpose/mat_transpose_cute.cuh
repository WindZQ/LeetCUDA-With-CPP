#ifndef MAT_TRANSPOSE_CUTE_CUH
#define MAT_TRANSPOSE_CUTE_CUH


#ifdef __cplusplus
extern "C"
{
#endif

void mat_transpose_cute_row2col_reg(const float* dA, float* dB, int M, int N, cudaStream_t stream);
void mat_transpose_cute_col2row_reg(const float* dA, float* dB, int M, int N, cudaStream_t stream);
void mat_transpose_cute_col_smem(const float* dA, float* dB, int M, int N, cudaStream_t stream);
void mat_transpose_cute_row_smem(const float* dA, float* dB, int M, int N, cudaStream_t stream);
void mat_transpose_cute_col_smem_swizzled(const float* dA, float* dB, int M, int N, cudaStream_t stream);
void mat_transpose_cute_row_smem_swizzled(const float* dA, float* dB, int M, int N, cudaStream_t stream);
void mat_transpose_cute_row_cvectorized(const float* dA, float* dB, int M, int N, cudaStream_t stream);
void mat_transpose_cute_row_cvectorized_swizzled(const float* dA, float* dB, int M, int N, cudaStream_t stream);
void mat_transpose_cute_row_rvectorized(const float* dA, float* dB, int M, int N, cudaStream_t stream);
void mat_transpose_cute_row_rvectorized_swizzled(const float* dA, float* dB, int M, int N, cudaStream_t stream);
void mat_transpose_cute_row_rvectorized_swizzled_optimized(const float* dA, float* dB, int M, int N, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif 