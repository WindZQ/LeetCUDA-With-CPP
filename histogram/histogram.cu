#include "histogram.cuh"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// Histogram
// grid(N/256), block(256)
// a: Nx1, y: count histogram, a >= 1
__global__ void histogram_i32_kernel(int* a, int* y, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < n) {
		atomicAdd(&(y[a[idx]]), 1);
	}
}

// Histogram + Vec4
// grid(N/256), block(256/4)
// a: Nx1, y: count histogram, a >= 1
__global__ void histogram_i32x4_kernel(int* a, int* y, int n)
{
	int idx = 4 * (threadIdx.x + blockIdx.x * blockDim.x);

	if (idx < n) {
		int4 reg_a = INT4(a[idx]);
		atomicAdd(&(y[reg_a.x]), 1);
		atomicAdd(&(y[reg_a.y]), 1);
		atomicAdd(&(y[reg_a.z]), 1);
		atomicAdd(&(y[reg_a.w]), 1);
	}
}

void histogram_i32(int* d_a, int* d_y, int N, cudaStream_t stream)
{
	dim3 block(256);
	dim3 grid((N + block.x - 1) / block.x);
	histogram_i32_kernel << <grid, block, 0, stream >> > (d_a, d_y, N);
}

void histogram_i32x4(int* d_a, int* d_y, int N, cudaStream_t stream)
{
	dim3 block(256 / 4);
	dim3 grid((N + block.x - 1) / block.x);
	histogram_i32_kernel << <grid, block, 0, stream >> > (d_a, d_y, N);
}