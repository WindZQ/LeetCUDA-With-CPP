#ifndef SIGMOID_CUH
#define SIGMOID_CUH

#ifdef __cplusplus
extern "C" 
{
#endif

void sigmoid_f32(float* x, float* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream);
void sigmoid_f32x4(float* x, float* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream);
void sigmoid_f16(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream);
void sigmoid_f16x2(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream);
void sigmoid_f16x8(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream);
void sigmoid_f16x8_pack(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif