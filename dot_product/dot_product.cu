#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "dot_product.cuh"

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32
// warp reduce sum
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

// dot product
// grid(N/256), block(256)
// a: Nx1, b: Nx1, y = sum(elementwise_mul(a,b))
template <const int num_threads = 256>
__global__ void dot_prod_f32_f32_kernel(float* a, float* b, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = blockIdx.x * num_threads + tid;
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float reduce_smem[num_warps];

	// keep the data in register is enough for warp operaion
	float prod = (idx < n) ? a[idx] * b[idx] : 0.0f;
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
	// warp leaders store the data to shared memory
	if (lane == 0) reduce_smem[warp] = prod;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	prod = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) prod = warp_reduce_sum_f32<num_warps>(prod);
	if (tid == 0) atomicAdd(y, prod);
}

// dot product + vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template <const int num_threads = 256 / 4>
__global__ void dot_prod_f32x4_f32_kernel(float* a, float* b, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = (blockIdx.x * num_threads + tid) * 4;
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float reduce_smem[num_warps];

	float4 reg_a = FLOAT4(a[idx]);
	float4 reg_b = FLOAT4(b[idx]);
	float prod = (idx < n) ? (reg_a.x * reg_b.x + reg_a.y * reg_b.y +
		reg_a.z * reg_b.z + reg_a.w * reg_b.w)
		: 0.0f;
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
	// warp leaders store the data to shared memory
	if (lane == 0) reduce_smem[warp] = prod;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	prod = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) prod = warp_reduce_sum_f32<num_warps>(prod);
	if (tid == 0) atomicAdd(y, prod);
}

// FP16
// warp reduce sum: half
template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val)
{
#pragma unroll
	for (int mask = warp_size; mask >= 1; mask >>= 1)
	{
		val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
	}

	return val;
}

template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val)
{
	float val_f32 = __half2float(val);
#pragma unroll
	for (int mask = warp_size; mask >= 1; mask >>= 1)
	{
		val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
	}

	return val_f32;
}

template <const int num_threads = 256>
__global__ void dot_prod_f16_f32_kernel(half* a, half* b, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = blockIdx.x * num_threads + tid;
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float reduce_smem[num_warps];
	// keep the data in register is enough for warp operaion
	half prod_f16 = (idx < n) ? __hmul(a[idx], b[idx]) : __float2half(0.0f);
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
	// warp leaders store the data to shared memory
	if (lane == 0) reduce_smem[warp] = prod;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum.
	prod = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) prod = warp_reduce_sum_f32<num_warps>(prod);
	if (tid == 0) atomicAdd(y, prod);
}

template <const int num_threads = 256 / 2>
__global__ void dot_prod_f16x2_f32_kernel(half* a, half* b, float* y, int n) 
{
	int tid = threadIdx.x;
	int idx = (blockIdx.x * num_threads + tid) * 2; // 2 half elements per thread
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float reduce_smem[num_warps];
	// keep the data in register is enough for warp operaion
	half2 reg_a = HALF2(a[idx]);
	half2 reg_b = HALF2(b[idx]);
	half prod_f16 =
		(idx < n) ? __hadd(__hmul(reg_a.x, reg_b.x), __hmul(reg_a.y, reg_b.y))
		: __float2half(0.0f);
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce.
	float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
	// warp leaders store the data to shared memory
	if (lane == 0) reduce_smem[warp] = prod;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	prod = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) prod = warp_reduce_sum_f32<num_warps>(prod);
	if (tid == 0) atomicAdd(y, prod);
}

template <const int num_threads = 256 / 8>
__global__ void dot_prod_f16x8_pack_f32_kernel(half* a, half* b, float* y, int n) 
{
	int tid = threadIdx.x;
	int idx = (blockIdx.x * num_threads + tid) * 8; // 8 half elements per thread
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float reduce_smem[num_warps];
	// temporary register(memory), .local space in ptx, addressable
	half pack_a[8], pack_b[8];                    // 8x16 bits=128 bits
	LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
	LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]); // load 128 bits
	const half z = __float2half(0.0f);
	half prod_f16 = z;
#pragma unroll
	for (int i = 0; i < 8; i += 2) 
	{
		half2 v = __hmul2(HALF2(pack_a[i]), HALF2(pack_b[i]));
		prod_f16 += (((idx + i) < n) ? (v.x + v.y) : z);
	}
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
	// warp leaders store the data to shared memory
	if (lane == 0) reduce_smem[warp] = prod;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	prod = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) prod = warp_reduce_sum_f32<num_warps>(prod);
	if (tid == 0) atomicAdd(y, prod);
}

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static inline int pick_grid(int N, int elems_per_block) 
{
	int g = ceil_div(N, elems_per_block);
	if (g < 1) g = 1;
	if (g > 4096) g = 4096;  

	return g;
}

void dot_prod_f32_f32(const float* a, const float* b, int n, float* out, cudaStream_t stream)
{
	if (!a || !b || !out || n < 0) return;
	cudaError err = cudaMemsetAsync(out, 0, sizeof(float), stream);
	if (err != cudaSuccess || n == 0) return;
	int grid = pick_grid(n, BLOCK_SIZE);
	dot_prod_f32_f32_kernel<BLOCK_SIZE> << <grid, BLOCK_SIZE, 0, stream >> > (const_cast<float*>(a), const_cast<float*>(b), out, n);
}

void dot_prod_f32x4_f32(const float* a, const float* b, int n, float* out, cudaStream_t stream)
{
	if (!a || !b || !out || n < 0) return;
	cudaError err = cudaMemsetAsync(out, 0, sizeof(float), stream);
	if (err != cudaSuccess || n == 0) return;
	int grid = pick_grid(n, BLOCK_SIZE);
	dot_prod_f32x4_f32_kernel<BLOCK_SIZE / 4> << <grid, BLOCK_SIZE, 0, stream >> > (const_cast<float*>(a), const_cast<float*>(b), out, n);
}

void dot_prod_f16_f32(const half* a, const half* b, int n, float* out, cudaStream_t stream)
{
	if (!a || !b || !out || n < 0) return;
	cudaError err = cudaMemsetAsync(out, 0, sizeof(float), stream);
	if (err != cudaSuccess || n == 0) return;
	int grid = pick_grid(n, BLOCK_SIZE);
	dot_prod_f16_f32_kernel<BLOCK_SIZE> << <grid, BLOCK_SIZE, 0, stream >> > (const_cast<half*>(a), const_cast<half*>(b), out, n);
}

void dot_prod_f16x2_f32(const half* a, const half* b, int n, float* out, cudaStream_t stream)
{
	if (!a || !b || !out || n < 0) return;
	cudaError err = cudaMemsetAsync(out, 0, sizeof(float), stream);
	if (err != cudaSuccess || n == 0) return;
	int grid = pick_grid(n, BLOCK_SIZE);
	dot_prod_f16x2_f32_kernel<BLOCK_SIZE / 2> << <grid, BLOCK_SIZE, 0, stream >> > (const_cast<half*>(a), const_cast<half*>(b), out, n);
}

void dot_prod_f16x8_pack_f32(const half* a, const half* b, int n, float* out, cudaStream_t stream)
{
	if (!a || !b || !out || n < 0) return;
	cudaError err = cudaMemsetAsync(out, 0, sizeof(float), stream);
	if (err != cudaSuccess || n == 0) return;
	int grid = pick_grid(n, BLOCK_SIZE);
	dot_prod_f16x8_pack_f32_kernel<BLOCK_SIZE / 8> << <grid, BLOCK_SIZE, 0, stream >> > (const_cast<half*>(a), const_cast<half*>(b), out, n);
}