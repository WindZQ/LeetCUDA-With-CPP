#ifndef GELU_CUH
#define GELU_CUH

#ifdef __cplusplus
extern "C"
{
#endif

void gelu_f32(float* x, float* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream);
void gelu_f32x4(float* x, float* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream);
void gelu_f16(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream);
void gelu_f16x2(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream);
void gelu_f16x8(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream);
void gelu_f16x8_pack(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif