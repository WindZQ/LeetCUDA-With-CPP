#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "relu.cuh"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32
// Relu x: N, y: N y=max(0,x)
// grid(N/256), block(K=256)
__global__ void relu_f32_kernel(float* x, float* y, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < n) {
		y[idx] = fmaxf(0.0f, x[idx]);
	}
}

// Relu x: N, y: N y=max(0,x) Vec4
// grid(N/256/4), block(256/4)
__global__ void relu_f32x4_kernel(float* x, float* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;

	if (idx < n) {
		float4 reg_x = FLOAT4(x[idx]);
		float4 reg_y;

		reg_x.x = fmaxf(0.0f, reg_x.x);
		reg_x.y = fmaxf(0.0f, reg_x.y);
		reg_x.z = fmaxf(0.0f, reg_x.z);
		reg_x.w = fmaxf(0.0f, reg_x.w);

		FLOAT4(y[idx]) = reg_y;
	}
}

// FP16
__global__ void relu_f16_kernel(half* x, half* y, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < n) {
		y[idx] = __hmax(__float2half(0.0f), x[idx]);
	}
}

__global__ void relu_f16x2_kernel(half* x, half* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;

	if (idx < n) {
		half2 reg_x = HALF2(x[idx]);
		half2 reg_y = HALF2(y[idx]);

		reg_y.x = __hmax(__float2half(0.0f), reg_x.x);
		reg_y.y = __hmax(__float2half(0.0f), reg_x.y);
		
		HALF2(y[idx]) = reg_y;
	}
}

__global__ void relu_f16x8_kernel(half* x, half* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 8;

	half2 reg_x_0 = HALF2(x[idx + 0]);
	half2 reg_x_1 = HALF2(x[idx + 2]);
	half2 reg_x_2 = HALF2(x[idx + 4]);
	half2 reg_x_3 = HALF2(x[idx + 6]);
	half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;

	reg_y_0.x = __hmax(__float2half(0.0f), reg_x_0.x);
	reg_y_0.y = __hmax(__float2half(0.0f), reg_x_0.y);
	reg_y_1.x = __hmax(__float2half(0.0f), reg_x_1.x);
	reg_y_1.y = __hmax(__float2half(0.0f), reg_x_1.y);
	reg_y_2.x = __hmax(__float2half(0.0f), reg_x_2.x);
	reg_y_2.y = __hmax(__float2half(0.0f), reg_x_2.y);
	reg_y_3.x = __hmax(__float2half(0.0f), reg_x_3.x);
	reg_y_3.y = __hmax(__float2half(0.0f), reg_x_3.y);

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

__global__ void relu_f16x8_pack_kernel(half* x, half* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
	const half2 z2 = { __float2half(0.0f), __float2half(0.0f) };
	// temporary register(memory), .local space in ptx, addressable
	half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
	// reinterpret as float4 and load 128 bits in 1 memory issue.
	LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

#pragma unroll
	for (int i = 0; i < 8; i += 2) 
	{
		// __hmax2 for half2 x 4
		HALF2(pack_y[i]) = __hmax2(HALF2(pack_x[i]), z2);
	}
	// reinterpret as float4 and store 128 bits in 1 memory issue.
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
	} else {
		block = dim3((unsigned)(256 / n_elements));
		grid = dim3((unsigned)((N + 256 - 1) / 256));
	}
}

void relu_f32(float* x, float* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 1, grid, block, N);
	relu_f32_kernel << <grid, block, 0, stream >> > (x, y, (int)N);
}

void relu_f32x4(float* x, float* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	if ((N % 4) != 0 || (ndim == 2 && (K % 4) != 0)) {
		relu_f32(x, y, ndim, S, K, N, stream);
	}
	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 4, grid, block, N);
	relu_f32x4_kernel << <grid, block, 0, stream >> > (x, y, (int)N);
}

void relu_f16(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 1, grid, block, N);
	relu_f16_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}

void relu_f16x2(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	if ((N % 2) != 0 || (ndim == 2 && (K % 2) != 0)) {
		relu_f16(x, y, ndim, S, K, N, stream);
	}
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 2, grid, block, N);
	relu_f16x2_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}

void relu_f16x8(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	if ((N % 8) != 0 || (ndim == 2 && (K % 8) != 0)) {
		relu_f16(x, y, ndim, S, K, N, stream);
	}
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 8, grid, block, N);
	relu_f16x8_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}

void relu_f16x8_pack(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	if ((N % 8) != 0 || (ndim == 2 && (K % 8) != 0)) {
		relu_f16(x, y, ndim, S, K, N, stream);
	}
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 8, grid, block, N);
	relu_f16x8_pack_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}