#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <device_launch_parameters.h>
#include <tuple>
#include <vector>

#ifdef __cplusplus
extern "C" 
{
#endif

void histogram_i32(int* d_a, int* d_y, int N, cudaStream_t stream);
void histogram_i32x4(int* d_a, int* d_y, int N, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif