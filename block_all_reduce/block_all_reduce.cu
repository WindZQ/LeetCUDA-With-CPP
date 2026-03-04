#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "block_all_reduce.cuh"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

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

// Block All Reduce Sum
// grid(N/256), block(256)
// a: Nx1, y = sum(a)
template <const int num_threads = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float* a, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = blockIdx.x * num_threads + tid;
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float reduce_smem[num_warps];
	// keep the data in register is enough for warp operaion
	float sum = (idx < n) ? a[idx] : 0.0f;
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
	// warp leaders store the data to shared memory
	if (lane == 0) reduce_smem[warp] = sum;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	sum = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) sum = warp_reduce_sum_f32<num_warps>(sum);
	if (tid == 0) atomicAdd(y, sum);
}

// Block All Reduce Sum + float4
// grid(N/256), block(256/4)
// a: Nx1, y = sum(a)
template <const int num_threads = 256 / 4>
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float* a, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = (blockIdx.x * num_threads + tid) * 4;
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float reduce_smem[num_warps];

	float4 reg_a = FLOAT4(a[idx]);
	// keep the data in register is enough for warp operaion
	float sum = (idx < n) ? (reg_a.x + reg_a.y + reg_a.z + reg_a.w) : 0.0f;
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
	// warp leaders store the data to shared memory
	if (lane == 0) reduce_smem[warp] = sum;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	sum = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) sum = warp_reduce_sum_f32<num_warps>(sum);
	if (tid == 0) atomicAdd(y, sum);
}

//  FP16
//  Warp Reduce Sum: Half
template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val)
{
#pragma unroll
	for (int mask = warp_size >> 1; mask >= 1; mask >>= 1)
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
	for (int mask = warp_size >> 1; mask >= 1; mask >>= 1)
	{
		val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
	}

	return val_f32;
}

// Block All Reduce Sum: Half
// grid(N/256), block(256)
// a: Nx1, y = sum(a)
template <const int num_threads = 256>
__global__ void block_all_reduce_sum_f16_f16_kernel(half* a, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = blockIdx.x * num_threads + tid;
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float reduce_smem[num_warps];
	// keep the data in register is enough for warp operaion
	half sum_f16 = (idx < n) ? a[idx] : __float2half(0.0f);
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp32 inter warps
	if (lane == 0) reduce_smem[warp] = __half2float(sum_f16);
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	float sum = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) sum = warp_reduce_sum_f32<num_warps>(sum);
	if (tid == 0) atomicAdd(y, sum);
}

template <const int num_threads = 256>
__global__ void block_all_reduce_sum_f16_f32_kernel(half* a, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = blockIdx.x * num_threads + tid;
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float reduce_smem[num_warps];
	// keep the data in register is enough for warp operaion
	half sum_f16 = (idx < n) ? a[idx] : __float2half(0.0f);
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	float sum_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(sum_f16);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp32 inter warps
	if (lane == 0) reduce_smem[warp] = sum_f32;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	float sum = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) sum = warp_reduce_sum_f32<num_warps>(sum);
	if (tid == 0) atomicAdd(y, sum);
}

template <const int num_threads = 256 / 2>
__global__ void block_all_reduce_sum_f16x2_f32_kernel(half* a, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = (blockIdx.x * num_threads + tid) * 2; // 2 half elements per thread
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float reduce_smem[num_warps];
	// keep the data in register is enough for warp operaion
	half2 reg_a = HALF2(a[idx]);
	half sum_f16 = (idx < n) ? __hadd(reg_a.x, reg_a.y) : __float2half(0.0f);
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	float sum_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(sum_f16);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp32 inter warps
	if (lane == 0) reduce_smem[warp] = sum_f32;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	float sum = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) sum = warp_reduce_sum_f32<num_warps>(sum);
	if (tid == 0) atomicAdd(y, sum);
}

template <const int num_threads = 256 / 2>
__global__ void block_all_reduce_sum_f16x2_f16_kernel(half* a, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = (blockIdx.x * num_threads + tid) * 2; // 2 half elements per thread
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float reduce_smem[num_warps];
	// keep the data in register is enough for warp operaion
	half2 reg_a = HALF2(a[idx]);
	half sum_f16 = (idx < n) ? __hadd(reg_a.x, reg_a.y) : __float2half(0.0f);
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp32 inter warps
	if (lane == 0) reduce_smem[warp] = __half2float(sum_f16);
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	float sum = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) sum = warp_reduce_sum_f32<num_warps>(sum);
	if (tid == 0) atomicAdd(y, sum);
}

template <const int num_threads = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f16_kernel(half* a, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = (blockIdx.x * num_threads + tid) * 8; // 8 half elements per thread
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float reduce_smem[num_warps];
	// keep the data in register is enough for warp operaion
	half pack_a[8]; // 8x16 bits=128 bits
	// reinterpret as float4 and load 128 bits in 1 memory issue
	LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
	const half z = __float2half(0.0f);
	half sum_f16 = z;

#pragma unroll
	for (int i = 0; i < 8; ++i) 
	{
		sum_f16 += (((idx + i) < n) ? pack_a[i] : z);
	}
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp32 inter warps
	if (lane == 0) reduce_smem[warp] = __half2float(sum_f16);
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	float sum = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) sum = warp_reduce_sum_f32<num_warps>(sum);
	if (tid == 0) atomicAdd(y, sum);
}

template <const int num_threads = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f32_kernel(half* a, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = (blockIdx.x * num_threads + tid) * 8; // 8 half elements per thread
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float reduce_smem[num_warps];
	// keep the data in register is enough for warp operaion
	half pack_a[8]; // 8x16 bits=128 bits
	// reinterpret as float4 and load 128 bits in 1 memory issue
	LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
	float sum_f32 = 0.0f;
#pragma unroll
	for (int i = 0; i < 8; ++i)
	{
		sum_f32 += (((idx + i) < n) ? __half2float(pack_a[i]) : 0.0f);
	}
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	sum_f32 = warp_reduce_sum_f32<WARP_SIZE>(sum_f32);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp32 inter warps
	if (lane == 0) reduce_smem[warp] = sum_f32;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	float sum = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) sum = warp_reduce_sum_f32<num_warps>(sum);
	if (tid == 0) atomicAdd(y, sum);
}

//  BF16
//  Warp Reduce Sum: Half
template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ __nv_bfloat16 warp_reduce_sum_bf16_bf16(__nv_bfloat16 val)
{
#pragma unroll
	for (int mask = warp_size >> 1; mask >= 1; mask >>= 1)
	{
		val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
	}

	return val;
}

template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_bf16_f32(__nv_bfloat16 val)
{
	float val_f32 = __bfloat162float(val);
#pragma unroll
	for (int mask = warp_size >> 1; mask >= 1; mask >>= 1)
	{
		val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
	}

	return val_f32;
}

// Block All Reduce Sum: BF16
// grid(N/256), block(256)
// a: Nx1, y = sum(a)
template<const int num_threads = 256>
__global__ void block_all_reduce_sum_bf16_bf16_kernel(__nv_bfloat16* a, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = blockIdx.x * num_threads + tid;
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ __nv_bfloat16  reduce_smem[num_warps];

	// keep the data in register is enough for warp operaion
	__nv_bfloat16 sum_bf16 = (idx < n) ? a[idx] : __float2bfloat16(0.0f);
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp32 inter warps
	if (lane == 0) reduce_smem[warp] = sum_bf16;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	__nv_bfloat16 sum = (lane < num_warps) ? reduce_smem[lane] : __float2bfloat16(0.0f);
	if (warp == 0) sum = warp_reduce_sum_bf16_bf16<num_warps>(sum);
	if (tid == 0) atomicAdd(y, __bfloat162float(sum));
}

template<const int num_threads = 256>
__global__ void block_all_reduce_sum_bf16_f32_kernel(__nv_bfloat16* a, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = blockIdx.x * num_threads + tid;
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float  reduce_smem[num_warps];

	// keep the data in register is enough for warp operaion
	__nv_bfloat16 sum_bf16 = (idx < n) ? a[idx] : __float2bfloat16(0.0f);
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp32 inter warps
	if (lane == 0) reduce_smem[warp] = sum_f32;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	float sum = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) sum = warp_reduce_sum_f32<num_warps>(sum);
	if (tid == 0) atomicAdd(y, sum);
}

template<const int num_threads = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_bf16_kernel(__nv_bfloat16* a, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = (blockIdx.x * num_threads + tid) * 2; // 2 bf16 elements per thread
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ __nv_bfloat16  reduce_smem[num_warps];

	// keep the data in register is enough for warp operaion
	__nv_bfloat162 reg_a = BFLOAT2(a[idx]);
	__nv_bfloat16 sum_bf16 = (idx < n) ? __hadd(reg_a.x, reg_a.y) : __float2bfloat16(0.0f);
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp32 inter warps
	if (lane == 0) reduce_smem[warp] = sum_bf16;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	__nv_bfloat16 sum = (lane < num_warps) ? reduce_smem[lane] : __float2bfloat16(0.0f);
	if (warp == 0) sum = warp_reduce_sum_bf16_bf16<num_warps>(sum);
	if (tid == 0) atomicAdd(y, __bfloat162float(sum));
}

template<const int num_threads = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_f32_kernel(__nv_bfloat16* a, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = (blockIdx.x * num_threads + tid) * 2; // 2 bf16 elements per thread
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float  reduce_smem[num_warps];

	// keep the data in register is enough for warp operaion
	__nv_bfloat162 reg_a = BFLOAT2(a[idx]);
	__nv_bfloat16 sum_bf16 = (idx < n) ? __hadd(reg_a.x, reg_a.y) : __float2bfloat16(0.0f);
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp32 inter warps
	if (lane == 0) reduce_smem[warp] = sum_f32;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	float sum = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) sum = warp_reduce_sum_f32<num_warps>(sum);
	if (tid == 0) atomicAdd(y, sum);
}

template<const int num_threads = 256 / 8>
__global__ void block_all_reduce_sum_bf16x8_pack_bf16_kernel(__nv_bfloat16* a, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = (blockIdx.x * num_threads + tid) * 8; // 8 bf16 elements per thread
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ __nv_bfloat16  reduce_smem[num_warps];
	// temporary register(memory), .local space in ptx, addressable
	__nv_bfloat16 pack_a[8]; // 8x16 bits=128 bits
	// reinterpret as float4 and load 128 bits in 1 memory issue
	LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
	const __nv_bfloat16 z = __float2bfloat16(0.0f);

	__nv_bfloat16 sum_bf16 = z;
#pragma unroll
	for (int i = 0; i < 8; ++i) 
	{
		sum_bf16 += (((idx + i) < n) ? pack_a[i] : z);
	}

	// keep the data in register is enough for warp operaion
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp32 inter warps
	if (lane == 0) reduce_smem[warp] = sum_bf16;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	__nv_bfloat16 sum = (lane < num_warps) ? reduce_smem[lane] : z;
	if (warp == 0) sum = warp_reduce_sum_bf16_bf16<num_warps>(sum);
	if (tid == 0) atomicAdd(y, __bfloat162float(sum));
}

template<const int num_threads = 256 / 8>
__global__ void block_all_reduce_sum_bf16x8_pack_f32_kernel(__nv_bfloat16* a, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = (blockIdx.x * num_threads + tid) * 8; // 8 bf16 elements per thread
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float  reduce_smem[num_warps];
	// temporary register(memory), .local space in ptx, addressable
	__nv_bfloat16 pack_a[8]; // 8x16 bits=128 bits.
	// reinterpret as float4 and load 128 bits in 1 memory issue
	LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
	const __nv_bfloat16 z = __float2bfloat16(0.0f);

	__nv_bfloat16 sum_bf16 = z;
#pragma unroll
	for (int i = 0; i < 8; ++i) 
	{
		sum_bf16 += (((idx + i) < n) ? pack_a[i] : z);
	}

	// keep the data in register is enough for warp operaion
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp32 inter warps
	if (lane == 0) reduce_smem[warp] = sum_f32;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	float sum = (lane < num_warps) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) sum = warp_reduce_sum_f32<num_warps>(sum);
	if (tid == 0) atomicAdd(y, sum);
}

// FP8
template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_fp8_e4m3_f16(__nv_fp8_storage_t val) 
{
	// typedef unsigned char __nv_fp8_storage_t
	// __half &operator=(const __half_raw &hr)
	half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E4M3);
#pragma unroll
	for (int mask = warp_size >> 1; mask >= 1; mask >>= 1) 
	{
		val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
	}

	return val_f16;
}

template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_fp8_e5m2_f16(__nv_fp8_storage_t val) 
{
	// typedef unsigned char __nv_fp8_storage_t
	// __half &operator=(const __half_raw &hr)
	half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E5M2);
#pragma unroll
	for (int mask = warp_size >> 1; mask >= 1; mask >>= 1)
	{
		val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
	}

	return val_f16;
}

template <const int num_threads = 256>
__global__ void block_all_reduce_sum_fp8_e4m3_f16_kernel(__nv_fp8_storage_t* a, float* y, int n) 
{
	int tid = threadIdx.x;
	int idx = blockIdx.x * num_threads + tid;
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ half reduce_smem[num_warps];

	// keep the data in register is enough for warp operaion
	__nv_fp8_storage_t sum_f8 = (idx < n) ? a[idx]
		: __nv_cvt_float_to_fp8(0.0f, __NV_SATFINITE, __NV_E4M3);
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	half sum_f16 = warp_reduce_sum_fp8_e4m3_f16<WARP_SIZE>(sum_f8);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp16 inter warps
	if (lane == 0) reduce_smem[warp] = sum_f16;
	__syncthreads(); // make sure the data is in shared memory.
	// the first warp compute the final sum.
	half sum = (lane < num_warps) ? reduce_smem[lane] : __float2half(0.0f);
	if (warp == 0) sum = warp_reduce_sum_f16_f16<num_warps>(sum);
	if (tid == 0) atomicAdd(y, __half2float(sum));
}

template <const int num_threads = 256>
__global__ void block_all_reduce_sum_fp8_e5m2_f16_kernel(__nv_fp8_storage_t* a, float* y, int n)
{
	int tid = threadIdx.x;
	int idx = blockIdx.x * num_threads + tid;
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ half reduce_smem[num_warps];

	// keep the data in register is enough for warp operaion
	__nv_fp8_storage_t sum_f8 = (idx < n) ? a[idx]
		: __nv_cvt_float_to_fp8(0.0f, __NV_SATFINITE, __NV_E5M2);
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	half sum_f16 = warp_reduce_sum_fp8_e5m2_f16<WARP_SIZE>(sum_f8);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp16 inter warps
	if (lane == 0) reduce_smem[warp] = sum_f16;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	half sum = (lane < num_warps) ? reduce_smem[lane] : __float2half(0.0f);
	if (warp == 0) sum = warp_reduce_sum_f16_f16<num_warps>(sum);
	if (tid == 0) atomicAdd(y, __half2float(sum));
}

template <const int num_threads = 256 / 16>
__global__ void block_all_reduce_sum_fp8_e4m3x16_pack_f16_kernel(__nv_fp8_storage_t* a, float* y, int n) 
{
	int tid = threadIdx.x;
	int idx = (blockIdx.x * num_threads + tid) * 16;
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ half reduce_smem[num_warps];
	__nv_fp8_storage_t pack_a[16]; // 16x8 bits=128 bits
	// reinterpret as float4 and load 128 bits in 1 memory issue
	LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
	half sum_f16 = __float2half(0.0f);

#pragma unroll
	for (int i = 0; i < 16; ++i) 
	{
		sum_f16 += __nv_cvt_fp8_to_halfraw(pack_a[i], __NV_E4M3);
	}

	// keep the data in register is enough for warp operaion
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce.
	sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp16 inter warps
	if (lane == 0) reduce_smem[warp] = sum_f16;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	half sum = (lane < num_warps) ? reduce_smem[lane] : __float2half(0.0f);
	if (warp == 0) sum = warp_reduce_sum_f16_f16<num_warps>(sum);
	if (tid == 0) atomicAdd(y, __half2float(sum));
}

template <const int num_threads = 256 / 16>
__global__ void block_all_reduce_sum_fp8_e5m2x16_pack_f16_kernel(__nv_fp8_storage_t* a, float* y, int n) 
{
	int tid = threadIdx.x;
	int idx = (blockIdx.x * num_threads + tid) * 16;
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ half reduce_smem[num_warps];
	__nv_fp8_storage_t pack_a[16]; // 16x8 bits=128 bits
	// reinterpret as float4 and load 128 bits in 1 memory issue
	LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
	half sum_f16 = __float2half(0.0f);

#pragma unroll
	for (int i = 0; i < 16; ++i) 
	{
		sum_f16 += __nv_cvt_fp8_to_halfraw(pack_a[i], __NV_E5M2);
	}

	// keep the data in register is enough for warp operaion
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
	// warp leaders store the data to shared memory
	// use float to keep sum from each block and reduce
	// with fp16 inter warps
	if (lane == 0) reduce_smem[warp] = sum_f16;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	half sum = (lane < num_warps) ? reduce_smem[lane] : __float2half(0.0f);
	if (warp == 0) sum = warp_reduce_sum_f16_f16<num_warps>(sum);
	if (tid == 0) atomicAdd(y, __half2float(sum));
}

// INT8
template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i8_i32(int8_t val) 
{
	int32_t val_i32 = static_cast<int32_t>(val);
#pragma unroll
	for (int mask = warp_size >> 1; mask >= 1; mask >>= 1) 
	{
		val_i32 += __shfl_xor_sync(0xffffffff, val_i32, mask);
	}

	return val_i32;
}

template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i32_i32(int32_t val) 
{
#pragma unroll
	for (int mask = warp_size >> 1; mask >= 1; mask >>= 1) 
	{
		val += __shfl_xor_sync(0xffffffff, val, mask);
	}

	return val;
}

template <const int num_threads = 256>
__global__ void block_all_reduce_sum_i8_i32_kernel(int8_t* a, int32_t* y, int n) 
{
	int tid = threadIdx.x;
	int idx = blockIdx.x * num_threads + tid;
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ int32_t reduce_smem[num_warps];

	// keep the data in register is enough for warp operaion
	int8_t sum_i8 = (idx < n) ? a[idx] : 0;
	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce.
	int32_t sum_i32 = warp_reduce_sum_i8_i32<WARP_SIZE>(sum_i8);
	if (lane == 0) reduce_smem[warp] = sum_i32;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	int32_t sum = (lane < num_warps) ? reduce_smem[lane] : 0;
	if (warp == 0) sum = warp_reduce_sum_i32_i32<num_warps>(sum);
	if (tid == 0) atomicAdd(y, sum);
}

template <const int num_threads = 256 / 16>
__global__ void block_all_reduce_sum_i8x16_pack_i32_kernel(int8_t* a, int32_t* y, int n) 
{
	int tid = threadIdx.x;
	int idx = (blockIdx.x * num_threads + tid) * 16;
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ int32_t reduce_smem[num_warps];
	int8_t pack_a[16]; // 16x8=128 bits
	// reinterpret as float4 and load 128 bits in 1 memory issue
	LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits

	// keep the data in register is enough for warp operaion
	int32_t sum_i32 = 0;
#pragma unroll
	for (int i = 0; i < 16; ++i) 
	{
		sum_i32 += (static_cast<int32_t>(pack_a[i]));
	}

	int warp = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;
	// perform warp sync reduce
	sum_i32 = warp_reduce_sum_i32_i32<WARP_SIZE>(sum_i32);
	if (lane == 0) reduce_smem[warp] = sum_i32;
	__syncthreads(); // make sure the data is in shared memory
	// the first warp compute the final sum
	int32_t sum = (lane < num_warps) ? reduce_smem[lane] : 0;
	if (warp == 0) sum = warp_reduce_sum_i32_i32<num_warps>(sum);
	if (tid == 0) atomicAdd(y, sum);
}
static inline bool is_aligned_n(const void* p, size_t n) { return ((uintptr_t)p % n) == 0u; }
static inline bool ok_intN_i64(int64_t v) { return (v > 0 && v <= (int64_t)INT_MAX); }

static inline bool device_supports_fp8() 
{
	int dev = 0;
	cudaGetDevice(&dev);
	cudaDeviceProp prop{};
	cudaGetDeviceProperties(&prop, dev);
	return (prop.major >= 9);
}

#define LAUNCH(NT_, packed_type, acc_type, InT, OutT)                                             \
  do {                                                                                            \
    dim3 grid((unsigned)S);                                                                       \
    dim3 block((unsigned)(NT_));                                                                  \
    InT* x_mut = const_cast<InT*>(reinterpret_cast<const InT*>(x_nc));                            \
    block_all_reduce_sum_##packed_type##_##acc_type##_kernel<(NT_)><<<grid, block, 0, stream>>>(  \
        x_mut, (OutT*)y, (int)N);                                                                 \
  } while (0)

#define LAUNCH_FLATTEN(packed_type, acc_type, InT, OutT, n_elems, pack_unsafe)                     \
  do {                                                                                             \
    const int NT_flat = 1024 / (n_elems);                                                          \
    if ((pack_unsafe) && (N % 1024 != 0)) return;                                                  \
    dim3 block((unsigned)NT_flat);                                                                 \
    dim3 grid((unsigned)((pack_unsafe) ? (N / 1024) : ((N + 1023) / 1024)));                       \
    InT* x_mut = const_cast<InT*>(reinterpret_cast<const InT*>(x));                                \
    block_all_reduce_sum_##packed_type##_##acc_type##_kernel<NT_flat>                              \
      <<<grid, block, 0, stream>>>(x_mut, (OutT*)y, (int)N);                                       \
  } while (0)


#define DISPATCH_NT(packed_type, acc_type, InT, OutT)                        \
  do {                                                                       \
    switch ((int)NT) {                                                       \
      case 32:   LAUNCH(32,   packed_type, acc_type, InT, OutT); break;      \
      case 64:   LAUNCH(64,   packed_type, acc_type, InT, OutT); break;      \
      case 128:  LAUNCH(128,  packed_type, acc_type, InT, OutT); break;      \
      case 256:  LAUNCH(256,  packed_type, acc_type, InT, OutT); break;      \
      case 512:  LAUNCH(512,  packed_type, acc_type, InT, OutT); break;      \
      case 1024: LAUNCH(1024, packed_type, acc_type, InT, OutT); break;      \
      default: return;                                                       \
    }                                                                        \
  } while (0)

#define DEFINE_REDUCE_FLOAT(fn_name, packed_type, acc_type, InParamT, InCastT, n_elems, align_bytes, pack_unsafe)           \
extern "C" void fn_name(InParamT x, float* y, int64_t S, int64_t K, cudaStream_t stream) {                                  \
  if (!x || !y) return;                                                                                                     \
  if (!ok_intN_i64(S) || !ok_intN_i64(K)) return;                                                                           \
  if ((K % (n_elems)) != 0) return;                                                                                         \
  const int64_t N = S * K;                                                                                                  \
  if (!ok_intN_i64(N)) return;                                                                                              \
  if ((align_bytes) && !is_aligned_n((const void*)x, (align_bytes))) return;                                                \
  const int NT = (int)(K / (n_elems));                                                                                      \
  (void)cudaMemsetAsync(y, 0, sizeof(float), stream);                                                                       \
  const void* x_nc = (const void*)x;                                                                                        \
  if (NT <= 1024) {                                                                                                         \
    DISPATCH_NT(packed_type, acc_type, InCastT, float);                                                                     \
  } else {                                                                                                                  \
    LAUNCH_FLATTEN(packed_type, acc_type, InCastT, float, (n_elems), (pack_unsafe));                                        \
  }                                                                                                                         \
}

#define DEFINE_REDUCE_I32(fn_name, packed_type, acc_type, InParamT, InCastT, n_elems, align_bytes, pack_unsafe)             \
extern "C" void fn_name(InParamT x, int32_t* y, int64_t S, int64_t K, cudaStream_t stream) {                                \
  if (!x || !y) return;                                                                                                     \
  if (!ok_intN_i64(S) || !ok_intN_i64(K)) return;                                                                           \
  if ((K % (n_elems)) != 0) return;                                                                                         \
  const int64_t N = S * K;                                                                                                  \
  if (!ok_intN_i64(N)) return;                                                                                              \
  if ((align_bytes) && !is_aligned_n((const void*)x, (align_bytes))) return;                                                \
  const int NT = (int)(K / (n_elems));                                                                                      \
  (void)cudaMemsetAsync(y, 0, sizeof(int32_t), stream);                                                                     \
  const void* x_nc = (const void*)x;                                                                                        \
  if (NT <= 1024) {                                                                                                         \
    DISPATCH_NT(packed_type, acc_type, InCastT, int32_t);                                                                   \
  } else {                                                                                                                  \
    LAUNCH_FLATTEN(packed_type, acc_type, InCastT, int32_t, (n_elems), (pack_unsafe));                                      \
  }                                                                                                                         \
}


DEFINE_REDUCE_FLOAT(block_all_reduce_sum_f32_f32, f32, f32, const float*, float, 1, 0, 0)
DEFINE_REDUCE_FLOAT(block_all_reduce_sum_f32x4_f32, f32x4, f32, const float*, float, 4, 16, 1)

DEFINE_REDUCE_FLOAT(block_all_reduce_sum_f16_f16, f16, f16, const void*, half, 1, 0, 0)
DEFINE_REDUCE_FLOAT(block_all_reduce_sum_f16_f32, f16, f32, const void*, half, 1, 0, 0)
DEFINE_REDUCE_FLOAT(block_all_reduce_sum_f16x2_f16, f16x2, f16, const void*, half, 2, 4, 0)
DEFINE_REDUCE_FLOAT(block_all_reduce_sum_f16x2_f32, f16x2, f32, const void*, half, 2, 4, 0)
DEFINE_REDUCE_FLOAT(block_all_reduce_sum_f16x8_pack_f16, f16x8_pack, f16, const void*, half, 8, 16, 1)
DEFINE_REDUCE_FLOAT(block_all_reduce_sum_f16x8_pack_f32, f16x8_pack, f32, const void*, half, 8, 16, 1)

DEFINE_REDUCE_FLOAT(block_all_reduce_sum_bf16_bf16, bf16, bf16, const void*, __nv_bfloat16, 1, 0, 0)
DEFINE_REDUCE_FLOAT(block_all_reduce_sum_bf16_f32, bf16, f32, const void*, __nv_bfloat16, 1, 0, 0)
DEFINE_REDUCE_FLOAT(block_all_reduce_sum_bf16x2_bf16, bf16x2, bf16, const void*, __nv_bfloat16, 2, 4, 0)
DEFINE_REDUCE_FLOAT(block_all_reduce_sum_bf16x2_f32, bf16x2, f32, const void*, __nv_bfloat16, 2, 4, 0)
DEFINE_REDUCE_FLOAT(block_all_reduce_sum_bf16x8_pack_bf16, bf16x8_pack, bf16, const void*, __nv_bfloat16, 8, 16, 1)
DEFINE_REDUCE_FLOAT(block_all_reduce_sum_bf16x8_pack_f32, bf16x8_pack, f32, const void*, __nv_bfloat16, 8, 16, 1)
DEFINE_REDUCE_FLOAT(block_all_reduce_sum_fp8_e4m3_f16, fp8_e4m3, f16, const uint8_t*, __nv_fp8_storage_t, 1, 0, 0)
DEFINE_REDUCE_FLOAT(block_all_reduce_sum_fp8_e4m3x16_pack_f16, fp8_e4m3x16_pack, f16, const uint8_t*, __nv_fp8_storage_t, 16, 16, 1)
DEFINE_REDUCE_FLOAT(block_all_reduce_sum_fp8_e5m2_f16, fp8_e5m2, f16, const uint8_t*, __nv_fp8_storage_t, 1, 0, 0)
DEFINE_REDUCE_FLOAT(block_all_reduce_sum_fp8_e5m2x16_pack_f16, fp8_e5m2x16_pack, f16, const uint8_t*, __nv_fp8_storage_t, 16, 16, 1)

DEFINE_REDUCE_I32(block_all_reduce_sum_i8_i32, i8, i32, const int8_t*, int8_t, 1, 0, 0)
DEFINE_REDUCE_I32(block_all_reduce_sum_i8x16_pack_i32, i8x16_pack, i32, const int8_t*, int8_t, 16, 16, 1)
