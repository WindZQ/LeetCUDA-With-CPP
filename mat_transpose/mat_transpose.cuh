#ifndef MAT_TRANSPOSE_CUH
#define MAT_TRANSPOSE_CUH

#ifdef __cplusplus
extern "C"
{
#endif

void mat_transpose_f32_col2row(float* x, float* y, int row, int col, cudaStream_t stream);
void mat_transpose_f32_row2col(float* x, float* y, int row, int col, cudaStream_t stream);
void mat_transpose_f32x4_col2row(float* x, float* y, int row, int col, cudaStream_t stream);
void mat_transpose_f32x4_row2col(float* x, float* y, int row, int col, cudaStream_t stream);
void mat_transpose_f32_col2row2d(float* x, float* y, int row, int col, cudaStream_t stream);
void mat_transpose_f32_row2col2d(float* x, float* y, int row, int col, cudaStream_t stream);
void mat_transpose_f32x4_col2row2d(float* x, float* y, int row, int col, cudaStream_t stream);
void mat_transpose_f32x4_row2col2d(float* x, float* y, int row, int col, cudaStream_t stream);
void mat_transpose_f32_diagonal2d(float* x, float* y, int row, int col, cudaStream_t stream);
void mat_transpose_f32x4_shared_col2row2d(float* x, float* y, int row, int col, cudaStream_t stream);
void mat_transpose_f32x4_shared_row2col2d(float* x, float* y, int row, int col, cudaStream_t stream);
void mat_transpose_f32x4_shared_bcf_col2row2d(float* x, float* y, int row, int col, cudaStream_t stream);
void mat_transpose_f32x4_shared_bcf_row2col2d(float* x, float* y, int row, int col, cudaStream_t stream);
void mat_transpose_f32x4_shared_bcf_merge_write_row2col2d(float* x, float* y, int row, int col, cudaStream_t stream);

#ifdef __cplusplus
}
#endif


#endif 