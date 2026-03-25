#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>

#include "hgemv.cuh"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])

// FP16
// Warp Reduce Sum
template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16(half val)
{
#pragma unroll
	for (int mask = warp_size >> 1; mask >= 1; mask >>= 1)
	{
		val += __shfl_xor_sync(0xffffffff, val, mask);
	}

	return val;
}

// HGEMV: Warp HGEMV K32
// 假设K为32的倍数，每个warp负责一行
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void hgemv_k32_f16_kernel(half* a, half* x, half* y, int M, int K)
{
	int tx = threadIdx.x;         // 0~31
	int ty = threadIdx.y;         // 0~4
	int bx = blockIdx.x;          // 0~M/4
	int lane = tx % WARP_SIZE;    // 0~31
	int m = ty + bx * blockDim.y; // (0~M/4) * 4 + (0~3)
	
	if (m < M) {
		half sum = 0.0f;
		int num_warps = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
		for (int w = 0; w < num_warps; ++w)
		{
			int k = w * WARP_SIZE + lane;
			sum += a[k + m * K] * x[k];
		}

		sum = warp_reduce_sum_f16<WARP_SIZE>(sum);
		if (lane == 0) y[m] = sum;
	}
}

// HGEMV: Warp HGEMV K128 + half2x2
// 假设K为128的倍数 float4
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void hgemv_k128_f16x4_kernel(half* a, half* x, half* y, int M, int K)
{
	// 每个线程负责4个元素，一个warp覆盖128个元素
	int tx = threadIdx.x;         // 0~31
	int ty = threadIdx.y;         // 0~4
	int bx = blockIdx.x;          // 0~M/4
	int lane = tx % WARP_SIZE;    // 0~31
	int m = ty + bx * blockDim.y; // (0~M/4) * 4 + (0~3)

	if (m < M) {
		half sum = 0.0f;
		// process 4 * WARP_SIZE elements per warp
		int num_warps = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
		for (int w = 0; w < num_warps; ++w)
		{
			int k = (w * WARP_SIZE + lane) * 4;
			half2 reg_x_0 = HALF2(x[k + 0]);
			half2 reg_x_1 = HALF2(x[k + 2]);
			half2 reg_a_0 = HALF2(a[m * K + k + 0]);
			half2 reg_a_1 = HALF2(a[m * K + k + 2]);
			sum += (reg_x_0.x * reg_a_0.x + reg_x_0.y * reg_a_0.y +
				reg_x_1.x * reg_a_1.x + reg_x_1.y * reg_a_1.y);
		}

		sum = warp_reduce_sum_f16<WARP_SIZE>(sum);
		if (lane == 0) y[m] = sum;
	}
}

// HGEMV: Warp HGEMV K16
// 假设K为16 < 32,每个warp负责2行，每行有16个元素
// NUM_THREADS=128, NUM_WARPS=NUM_THREADS/WARP_SIZE;
// NUM_ROWS=NUM_WARPS * ROW_PER_WARP, grid(M/NUM_ROWS), block(32,NUM_WARPS)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
template <const int row_pre_warp = 2>
__global__ void hgemv_k16_f16_kernel(half* a, half* x, half* y, int M, int K)
{
	constexpr int k_warp_size = (WARP_SIZE + row_pre_warp - 1) / row_pre_warp;
	int tx = threadIdx.x;         // 0~31
	int ty = threadIdx.y;         // 0~num_warps
	int bx = blockIdx.x;          // 0~M/num_warps
	int lane = tx % WARP_SIZE;    // 0~31
	int k = lane % k_warp_size;   // 0~15
	// gloabl row of a: M x K and y:M x 1, blockDim.y= num_warps
	int m = (ty + blockDim.y * bx) * row_pre_warp + lane / k_warp_size;

	if (m < M) {
		half sum = a[k + m * K] * x[k];
		sum = warp_reduce_sum_f16<k_warp_size>(sum);
		if (k == 0) y[m] = sum;
	}
}

void hgemv_k32_f16(half* a, half* x, half* y, int M, int K) 
{
	if (a == nullptr || x == nullptr || y == nullptr) {
		fprintf(stderr, "hgemv_k32_f16: null pointer.\n");
		return;
	}

	if (K % 32 != 0) {
		fprintf(stderr, "hgemv_k32_f16: K must be multiple of 32.\n");
		return;
	}

	dim3 block(32, 4);
	dim3 grid((M + 4 - 1) / 4);

	hgemv_k32_f16_kernel << <grid, block >> > (a, x, y, M, K);
}

void hgemv_k128_f16x4(half * a, half * x, half * y, int M, int K) 
{
	if (a == nullptr || x == nullptr || y == nullptr) {
		fprintf(stderr, "hgemv_k128_f16x4: null pointer.\n");
		return;
	}
	if (K % 128 != 0) {
		fprintf(stderr, "hgemv_k128_f16x4: K must be multiple of 128.\n");
		return;
	}

	dim3 block(32, 4);
	dim3 grid((M + 4 - 1) / 4);

	hgemv_k128_f16x4_kernel << <grid, block >> > (a, x, y, M, K);
}

void hgemv_k16_f16(half * a, half * x, half * y, int M, int K) 
{
	if (a == nullptr || x == nullptr || y == nullptr) {
		fprintf(stderr, "hgemv_k16_f16: null pointer.\n");
		return;
	}

	if (K != 16) {
		fprintf(stderr, "hgemv_k16_f16: K must be 16.\n");
		return;
	}

	constexpr int NUM_THREADS = 128;
	constexpr int ROW_PER_WARP = 2;
	constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
	constexpr int NUM_ROWS = NUM_WARPS * ROW_PER_WARP;

	dim3 block(32, NUM_WARPS);
	dim3 grid((M + NUM_ROWS - 1) / NUM_ROWS);

	hgemv_k16_f16_kernel<ROW_PER_WARP> << <grid, block >> > (a, x, y, M, K);
}