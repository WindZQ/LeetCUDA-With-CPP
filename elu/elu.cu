#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "elu.cuh"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define ALPHA 1.0f 

__device__ __forceinline__ half hexp_compat(half a)
{
	return __float2half(__expf(__half2float(a)));
}

// ELU ¼ÆËãº¯Êý
// FP32
__device__ __forceinline__ float elu(float x)
{
	return x > 0.f ? x : ALPHA * (expf(x) - 1.f);
}

// FP16
__device__ __forceinline__ half elu_half(half x)
{
	return __hgt(x, __float2half(0.0f)) ? x : __hmul(__float2half(ALPHA), __hsub(hexp_compat(x), __float2half(1.0f)));
}

// CUDA ºËº¯Êý
// FP32 
__global__ void elu_f32_kernel(float* x, float* y, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < n) {
		y[idx] = elu(x[idx]);
	}
}

__global__ void elu_f32x4_kernel(float* x, float* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;

	if (idx < n) {
		float4 reg_x = FLOAT4(x[idx]);
		float4 reg_y;

		reg_y.x = elu(reg_x.x);
		reg_y.y = elu(reg_x.y);
		reg_y.z = elu(reg_x.z);
		reg_y.w = elu(reg_x.w);

		FLOAT4(y[idx]) = reg_y;
	}
}

// FP16
__global__ void elu_f16_kernel(half* x, half* y, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < n) {
		y[idx] = elu_half(x[idx]);
	}
}

__global__ void elu_f16x2_kernel(half* x, half* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;

	if (idx < n) {
		half2 reg_x = HALF2(x[idx]);
		half2 reg_y;

		reg_y.x = elu_half(reg_x.x);
		reg_y.y = elu_half(reg_x.y);

		HALF2(y[idx]) = reg_y;
	}
}

__global__ void elu_f16x8_kernel(half* x, half* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 8;

	half2 reg_x_0 = HALF2(x[idx + 0]);
	half2 reg_x_1 = HALF2(x[idx + 2]);
	half2 reg_x_2 = HALF2(x[idx + 4]);
	half2 reg_x_3 = HALF2(x[idx + 6]);
	half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;

	reg_y_0.x = elu_half(reg_x_0.x);
	reg_y_0.y = elu_half(reg_x_0.y);
	reg_y_1.x = elu_half(reg_x_1.x);
	reg_y_1.y = elu_half(reg_x_1.y);
	reg_y_2.x = elu_half(reg_x_2.x);
	reg_y_2.y = elu_half(reg_x_2.y);
	reg_y_3.x = elu_half(reg_x_3.x);
	reg_y_3.y = elu_half(reg_x_3.y);

	if ((idx + 0) < n) {
		HALF2(y[idx + 0]) = reg_y_0;
	}
	if ((idx + 2) < n) {
		HALF2(y[idx + 2]) = reg_y_1;
	}
	if ((idx + 4) < n) {
		HALF2(y[idx + 4]) = reg_y_2;
	}
	if ((idx + 6) < n) {
		HALF2(y[idx + 6]) = reg_y_3;
	}
}

__global__ void elu_f16x8_pack_kernel(half* x, half* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
	half pack_x[8], pack_y[8];
	LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
	for (int i = 0; i < 8; i++) 
	{
		pack_y[i] = elu_half(pack_x[i]);
	}

	if ((idx + 7) < n) {
		LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
	}
}

static inline void launch_1d_or_2d(int64_t ndim, int64_t S, int64_t K, int n_elements,
	dim3& grid, dim3& block, int64_t N)
{
	if (ndim == 2 && (K / n_elements) <= 1024) {
		block = dim3((unsigned)(K / n_elements));
		grid = dim3((unsigned)S);
	}
	else {
		block = dim3((unsigned)(256 / n_elements));
		grid = dim3((unsigned)((N + 256 - 1) / 256));
	}
}

void elu_f32(float* x, float* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 1, grid, block, N);
	elu_f32_kernel << <grid, block, 0, stream >> > (x, y, (int)N);
}

void elu_f32x4(float* x, float* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	if ((N % 4) != 0 || (ndim == 2 && (K % 4) != 0)) {
		elu_f32(x, y, ndim, S, K, N, stream);
	}
	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 4, grid, block, N);
	elu_f32x4_kernel << <grid, block, 0, stream >> > (x, y, (int)N);
}

void elu_f16(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 1, grid, block, N);
	elu_f16_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}

void elu_f16x2(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	if ((N % 2) != 0 || (ndim == 2 && (K % 2) != 0)) {
		elu_f16(x, y, ndim, S, K, N, stream);
	}
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 2, grid, block, N);
	elu_f16x2_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}

void elu_f16x8(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	if ((N % 8) != 0 || (ndim == 2 && (K % 8) != 0)) {
		elu_f16(x, y, ndim, S, K, N, stream);
	}
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 8, grid, block, N);
	elu_f16x8_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}

void elu_f16x8_pack(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	if ((N % 8) != 0 || (ndim == 2 && (K % 8) != 0)) {
		elu_f16(x, y, ndim, S, K, N, stream);
	}
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 8, grid, block, N);
	elu_f16x8_pack_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}