#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>

#include "sgemv.cuh"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32
// Warp Reduce Sum
template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val)
{
#pragma unroll
	for (int mask = warp_size >> 1; mask >= 1; mask >>= 1)
	{
		val += __shfl_xor_sync(0xffffffff, val, mask);
	}

	return val;
}

// SGEMV: Warp SGEMV K32
// 假设K为32的倍数，每个warp负责一行
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k32_f32_kernel(float* a, float* x, float *y, int M, int K)
{
	int tx = threadIdx.x;         // 0~31
	int ty = threadIdx.y;         // 0~4
	int bx = blockIdx.x;          // 0~M/4
	int lane = tx % WARP_SIZE;    // 0~31
	int m = bx * blockDim.y + ty; // (0~M/4) * 4 + (0~3)

	if (m < M) {
		float sum = 0.0f;
		int num_warps = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
		for (int w = 0; w < num_warps; ++w)
		{
			int k = w * WARP_SIZE + lane;
			sum += a[m * K + k] * x[k];
		}

		sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
		if (lane == 0) y[m] = sum;
	}
}

// SGEMV: Warp SGEMV K128 + Vec4
// 假设K为128的倍数 float4
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k128_f32x4_kernel(float* a, float* x, float* y, int M, int K)
{
	// 每个线程负责4个元素，一个warp覆盖128个元素
	int tx = threadIdx.x;         // 0~31
	int ty = threadIdx.y;         // 0~3
	int bx = blockIdx.x;          // 0~M/4
	int lane = tx % WARP_SIZE;    // 0~31
	int m = blockDim.y * bx + ty; // (0~M/4) * 4 + (0~3)

	if (m < M) {
		float sum = 0.0f;
		int num_warps = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
		for (int w = 0; w < num_warps; ++w)
		{
			int k = (w * WARP_SIZE + lane) * 4;
			float4 reg_x = FLOAT4(x[k]);
			float4 reg_a = FLOAT4(a[k + m * K]);
			sum += (reg_a.x * reg_x.x + reg_a.y * reg_x.y + reg_a.z * reg_x.z +
				reg_a.w * reg_x.w);
		}

		sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
		if (lane == 0) y[m] = sum;
	}
}

// SGEMV: Warp SGEMV K16
// 假设K为16 < 32,每个warp负责2行，每行有16个元素
// NUM_THREADS=128, NUM_WARPS=NUM_THREADS/WARP_SIZE;
// NUM_ROWS=NUM_WARPS * ROW_PER_WARP, grid(M/NUM_ROWS), block(32,NUM_WARPS)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
template <const int row_per_warp = 2>
__global__ void sgemv_k16_f32_kernel(float* a, float* x, float* y, int M, int K)
{
	constexpr int k_warp_size = (WARP_SIZE + row_per_warp - 1) / row_per_warp;
	int tx = threadIdx.x;      // 0~31
	int ty = threadIdx.y;      // 0~NUM_WARPS
	int bx = blockIdx.x;       // 0~M/NUM_ROWS (NUM_ROWS=NUM_WARPS * ROW_PER_WARP)
	int lane = tx % WARP_SIZE; // 0~31
	int k = lane % k_warp_size; // 0~15
	// gloabl row of a: MxK and y:Mx1, blockDim.y=NUM_WARPS
	int m = (blockDim.y * bx + ty) * row_per_warp + lane / k_warp_size;

	if (m < M) {
		float sum = a[k + m * K] * x[k];
		sum = warp_reduce_sum_f32<k_warp_size>(sum);
		if (k == 0) y[m] = sum;
	}
}

void sgemv_k32_f32(float* a, float* x, float* y, int M, int K) 
{
	if (K % 32 != 0) {
		fprintf(stderr, "Error: K must be multiple of 32\n");
		exit(EXIT_FAILURE);
	}

	dim3 block(32, 4);
	dim3 grid((M + 4 - 1) / 4);

	sgemv_k32_f32_kernel << <grid, block >> > (a, x, y, M, K);
}

void sgemv_k128_f32x4(float* a, float* x, float* y, int M, int K) 
{
	if (K % 128 != 0) {
		fprintf(stderr, "Error: K must be multiple of 128\n");
		exit(EXIT_FAILURE);
	}

	dim3 block(32, 4);
	dim3 grid((M + 4 - 1) / 4);

	sgemv_k128_f32x4_kernel << <grid, block >> > (a, x, y, M, K);
}

void sgemv_k16_f32(float* a, float* x, float* y, int M, int K) 
{
	if (K != 16) {
		fprintf(stderr, "Error: K must be 16\n");
		exit(EXIT_FAILURE);
	}

	constexpr int NUM_THREADS = 128;
	constexpr int ROW_PER_WARP = 2;
	constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
	constexpr int NUM_ROWS = NUM_WARPS * ROW_PER_WARP;

	dim3 block(32, NUM_WARPS);
	dim3 grid((M + NUM_ROWS - 1) / NUM_ROWS);

	sgemv_k16_f32_kernel<ROW_PER_WARP> << <grid, block >> > (a, x, y, M, K);
}