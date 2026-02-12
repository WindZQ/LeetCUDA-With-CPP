#ifndef EMBEDDING_CUH
#define EMBEDDING_CUH

#ifdef __cplusplus
extern "C"
{
#endif

void embedding_f32(const int* idx, float* weight, float* out,
	int N, int K, cudaStream_t stream);
void embedding_f32x4(const int* idx, float* weight, float* out,
	int N, int K, cudaStream_t stream);
void embedding_f32x4_pack(const int* idx, float* weight, float* out,
	int N, int K, cudaStream_t stream);
void embedding_f16(const int* idx, half* weight, half* out,
	int N, int K, cudaStream_t stream);
void embedding_f16x8(const int* idx, half* weight, half* out,
	int N, int K, cudaStream_t stream);
void embedding_f16x8_pack(const int* idx, half* weight, half* out,
	int N, int K, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif