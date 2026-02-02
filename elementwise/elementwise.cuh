#ifndef ELEMENTWIST_CUH
#define ELEMENTWIST_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <device_launch_parameters.h>

#ifdef __cplusplus
extern "C" 
{
#endif

void elementwise_add_f32(float* a, float* b, float* c, int64_t N, int64_t S, int64_t K, cudaStream_t stream);
void elementwise_add_f32x4(float* a, float* b, float* c, int64_t N, int64_t S, int64_t K, cudaStream_t stream);
void elementwise_add_f16(__half* a, __half* b, __half* c, int64_t N, int64_t S, int64_t K, cudaStream_t stream);
void elementwise_add_f16x2(__half* a, __half* b, __half* c, int64_t N, int64_t S, int64_t K, cudaStream_t stream);
void elementwise_add_f16x8(__half* a, __half* b, __half* c, int64_t N, int64_t S, int64_t K, cudaStream_t stream);
void elementwise_add_f16x8_pack(__half* a, __half* b, __half* c, int64_t N, int64_t S, int64_t K, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif