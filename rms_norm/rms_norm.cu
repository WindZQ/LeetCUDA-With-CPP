#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#include "rms_norm.cuh"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

//  FP32
//  Warp Reduce Sum
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

// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc
// grid 1D block 1D, grid(N/256), block(256)
template <const int num_threads = 256>
__device__ __forceinline__ float block_reduce_sum_f32(float val) 
{
    // always <= 32 warps per block (limited by 1024 threads per block)
    constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[num_warps];

    val = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    val = (lane < num_warps) ? shared[lane] : 0.0f;
    val = warp_reduce_sum_f32<num_warps>(val);

    return val;
}

// RMS Norm: x: NxK(K=256<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template <const int num_threads = 256>
__global__ void rms_norm_f32_kernel(float* x, float* y, float g, int n, int k) 
{
    int tid = threadIdx.x; // 0..k-1
    int bid = blockIdx.x;  // 0..n-1
    int idx = tid + bid * blockDim.x;
    const float epsilon = 1e-5f;

    __shared__ float s_variance;                 // shared within block
    float value = (idx < n * k) ? x[idx] : 0.0f; // load once only
    float variance = value * value;
    variance = block_reduce_sum_f32<num_threads>(variance);
    if (tid == 0) s_variance = rsqrtf(variance / (float)k + epsilon);
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads();
    if (idx < n * k)
        y[idx] = (value * s_variance) * g;
}

// RMS Norm Vec4: x: NxK(K=256<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K/4<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template <const int num_threads = 256 / 4>
__global__ void rms_norm_f32x4_kernel(float* x, float* y, float g, int n, int k) 
{
    int tid = threadIdx.x; // 0..K-1
    int bid = blockIdx.x;  // 0..N-1
    int idx = (tid + bid * blockDim.x) * 4;
    const float epsilon = 1e-5f;

    __shared__ float s_variance; // shared within block
    float4 reg_x = FLOAT4(x[idx]);
    float variance = (idx < n * k) ? (reg_x.x * reg_x.x + reg_x.y * reg_x.y +
        reg_x.z * reg_x.z + reg_x.w * reg_x.w)
        : 0.0f;
    variance = block_reduce_sum_f32<num_threads>(variance);
    if (tid == 0) s_variance = rsqrtf(variance / (float)k + epsilon);
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads();
    float4 reg_y;
    reg_y.x = reg_x.x * s_variance * g;
    reg_y.y = reg_x.y * s_variance * g;
    reg_y.z = reg_x.z * s_variance * g;
    reg_y.w = reg_x.w * s_variance * g;
    if (idx < n * k)
        FLOAT4(y[idx]) = reg_y;
}

//  FP16
//  Warp Reduce Sum: Half
template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) 
{
#pragma unroll
    for (int mask = warp_size >> 1; mask >= 1; mask >>= 1) 
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
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

template <const int num_threads = 256>
__device__ half block_reduce_sum_f16_f16(half val) 
{
    // always <= 32 warps per block (limited by 1024 threads per block)
    constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ half shared[num_warps];
    // reduce using half dtype within warps
    val = warp_reduce_sum_f16_f16<WARP_SIZE>(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    val = (lane < num_warps) ? shared[lane] : __float2half(0.0f);
    val = warp_reduce_sum_f16_f16<num_warps>(val);

    return val; 
}

template <const int num_threads = 256>
__device__ float block_reduce_sum_f16_f32(half val) 
{
    // always <= 32 warps per block (limited by 1024 threads per block)
    constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[num_warps];
    // reduce using float dtype within warps
    float val_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(val);
    if (lane == 0) shared[warp] = val_f32;
    __syncthreads();
    val_f32 = (lane < num_warps) ? shared[lane] : 0.0f;
    val_f32 = warp_reduce_sum_f32<num_warps>(val_f32);

    return val_f32; 
}

template <const int num_threads = 256>
__global__ void rms_norm_f16_f16_kernel(half* x, half* y, float g, int n, int k) 
{
    int tid = threadIdx.x; // 0..k-1
    int bid = blockIdx.x;  // 0..n-1
    int idx = tid + bid * blockDim.x;
    const half epsilon = __float2half(1e-5f);
    const half g_ = __float2half(g);
    const half k_ = __int2half_rn(k);

    __shared__ half s_variance; // shared within block
    half value = (idx < n * k) ? x[idx] : __float2half(0.0f); // load once only
    half variance = value * value;
    variance = block_reduce_sum_f16_f16<num_threads>(variance);
    if (tid == 0) s_variance = hrsqrt(variance / k_ + epsilon);
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads();
    if (idx < n * k)
        y[idx] = (value * s_variance) * g_;
}

template <const int num_threads = 256>
__global__ void rms_norm_f16x2_f16_kernel(half* x, half* y, float g, int n, int k) 
{
    int tid = threadIdx.x; // 0..k-1
    int bid = blockIdx.x;  // 0..n-1
    int idx = (tid + bid * blockDim.x) * 2;
    const half epsilon = __float2half(1e-5f);
    const half g_ = __float2half(g);
    const half k_ = __int2half_rn(k);

    __shared__ half s_variance; // shared within block
    half2 reg_x = HALF2(x[idx]);
    half variance = (idx < n * k) ? (reg_x.x * reg_x.x + reg_x.y * reg_x.y)
        : __float2half(0.0f);
    variance = block_reduce_sum_f16_f16<num_threads>(variance);
    if (tid == 0) s_variance = hrsqrt(variance / k_ + epsilon);
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads();
    half2 reg_y;
    reg_y.x = reg_x.x * s_variance * g_;
    reg_y.y = reg_x.y * s_variance * g_;
    if (idx < n * k)
        HALF2(y[idx]) = reg_y;
}

#define HALF2_VARIANCE(reg, i)                                                 \
  (((idx + (i)) < n * k) ? ((reg).x * (reg).x + (reg).y * (reg).y)             \
                         : __float2half(0.0f))

#define FLOAT2_VARIANCE(reg, i)                                                \
  (((idx + (i)) < n * k) ? ((reg).x * (reg).x + (reg).y * (reg).y) : 0.0f)

#define HALF2_RMS_NORM(reg_y, reg_x, g)                                        \
  (reg_y).x = (reg_x).x * s_variance * (g);                                    \
  (reg_y).y = (reg_x).y * s_variance * (g);

#define FLOAT2_RMS_NORM(reg_y, reg_x, g)                                       \
  (reg_y).x = (reg_x).x * s_variance * (g);                                    \
  (reg_y).y = (reg_x).y * s_variance * (g);

template <const int num_threads = 256>
__global__ void rms_norm_f16x8_f16_kernel(half* x, half* y, float g, int n, int k)
{
    int tid = threadIdx.x; // 0..k-1
    int bid = blockIdx.x;  // 0..n-1
    int idx = (tid + bid * blockDim.x) * 8;
    const half epsilon = __float2half(1e-5f);
    const half g_ = __float2half(g);
    const half k_ = __int2half_rn(k);

    __shared__ half s_variance; // shared within block
    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);
    half variance = HALF2_VARIANCE(reg_x_0, 0);
    variance += HALF2_VARIANCE(reg_x_1, 2);
    variance += HALF2_VARIANCE(reg_x_2, 4);
    variance += HALF2_VARIANCE(reg_x_3, 6);
    variance = block_reduce_sum_f16_f16<num_threads>(variance);
    if (tid == 0) s_variance = hrsqrt(variance / k_ + epsilon);
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads();
    // manual unroll
    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    HALF2_RMS_NORM(reg_y_0, reg_x_0, g_);
    HALF2_RMS_NORM(reg_y_1, reg_x_1, g_);
    HALF2_RMS_NORM(reg_y_2, reg_x_2, g_);
    HALF2_RMS_NORM(reg_y_3, reg_x_3, g_);
    if ((idx + 0) < n * k) {
        HALF2(y[idx + 0]) = reg_y_0;
    }
    if ((idx + 2) < n * k) {
        HALF2(y[idx + 2]) = reg_y_1;
    }
    if ((idx + 4) < n * k) {
        HALF2(y[idx + 4]) = reg_y_2;
    }
    if ((idx + 6) < n * k) {
        HALF2(y[idx + 6]) = reg_y_3;
    }
}

template <const int num_threads = 256>
__global__ void rms_norm_f16x8_f32_kernel(half* x, half* y, float g, int n, int k) 
{
    int tid = threadIdx.x; // 0..k-1
    int bid = blockIdx.x;  // 0..n-1
    int idx = (tid + bid * blockDim.x) * 8;
    const float epsilon = 1e-5f;

    __shared__ float s_variance; // shared within block
    // manual unroll and improve L2 cache hit rate.
    // Only   L2 cache: load 32  bytes in 1 memory issue (default)
    // Enable L1 cache: load 128 bytes in 1 memory issue (-Xptxas -dlcm=ca)
    // why try fp16x8 within 1 threads? ref:
    // https://zhuanlan.zhihu.com/p/641639133 0. first, tid_0 load 32 bytes in 1
    // memory issue and cache data into L2 cache.
    // 1. then, tid_1,...,tid_3 hit L2 cache and load data from L2 cache directly
    float2 reg_x_0 = __half22float2(HALF2(x[idx + 0]));
    float2 reg_x_1 = __half22float2(HALF2(x[idx + 2]));
    float2 reg_x_2 = __half22float2(HALF2(x[idx + 4]));
    float2 reg_x_3 = __half22float2(HALF2(x[idx + 6]));

    float variance = FLOAT2_VARIANCE(reg_x_0, 0);
    variance += FLOAT2_VARIANCE(reg_x_1, 2);
    variance += FLOAT2_VARIANCE(reg_x_2, 4);
    variance += FLOAT2_VARIANCE(reg_x_3, 6);

    variance = block_reduce_sum_f32<num_threads>(variance);
    if (tid == 0) s_variance = rsqrtf(variance / (float)k + epsilon);
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads();
    // manual unroll
    float2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    FLOAT2_RMS_NORM(reg_y_0, reg_x_0, g);
    FLOAT2_RMS_NORM(reg_y_1, reg_x_1, g);
    FLOAT2_RMS_NORM(reg_y_2, reg_x_2, g);
    FLOAT2_RMS_NORM(reg_y_3, reg_x_3, g);
    if ((idx + 0) < n * k) {
        HALF2(y[idx + 0]) = __float22half2_rn(reg_y_0);
    }
    if ((idx + 2) < n * k) {
        HALF2(y[idx + 2]) = __float22half2_rn(reg_y_1);
    }
    if ((idx + 4) < n * k) {
        HALF2(y[idx + 4]) = __float22half2_rn(reg_y_2);
    }
    if ((idx + 6) < n * k) {
        HALF2(y[idx + 6]) = __float22half2_rn(reg_y_3);
    }
}

template <const int num_threads = 256>
__global__ void rms_norm_f16_f32_kernel(half* x, half* y, float g, int n, int k) 
{
    int tid = threadIdx.x; // 0..k-1
    int bid = blockIdx.x;  // 0..n-1
    int idx = tid + bid * blockDim.x;
    const float epsilon = 1e-5f;

    __shared__ float s_variance; // shared within block
    float value = (idx < n * k) ? __half2float(x[idx]) : 0.0f; // load once only
    float variance = value * value;
    variance = block_reduce_sum_f32<num_threads>(variance);
    if (tid == 0) s_variance = rsqrtf(variance / (float)k + epsilon);
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads();
    if (idx < n * k) {
        y[idx] = __float2half((value * s_variance) * g);
    }
}

template <const int num_threads = 256>
__global__ void rms_norm_f16x8_pack_f16_kernel(half* x, half* y, float g, int n, int k) 
{
    int tid = threadIdx.x; // 0..k-1
    int bid = blockIdx.x;  // 0..n-1
    int idx = (tid + bid * blockDim.x) * 8;
    const half epsilon = __float2half(1e-5f);
    const half g_ = __float2half(g);
    const half k_ = __int2half_rn(k);
    const half z_ = __float2half(0.0f);

    __shared__ half s_variance; // shared within block
    // temporary register(memory), .local space in ptx, addressable
    half pack_x[8], pack_y[8]; // 8x16 bits=128 bits
    // reinterpret as float4 and load 128 bits in 1 memory issue
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

    half variance = z_;
#pragma unroll
    for (int i = 0; i < 8; ++i) 
    {
        variance += ((idx + i) < n * k ? pack_x[i] * pack_x[i] : z_);
    }

    variance = block_reduce_sum_f16_f16<num_threads>(variance);
    if (tid == 0) s_variance = hrsqrt(variance / k_ + epsilon);
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 8; ++i) 
    {
        pack_y[i] = pack_x[i] * s_variance * g_;
    }
    // reinterpret as float4 and store 128 bits in 1 memory issue.
    if ((idx + 7) < n * k) {
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
    // TODO: support non 8-multiple k here
}

template <const int num_threads = 256>
__global__ void rms_norm_f16x8_pack_f32_kernel(half* x, half* y, float g, int n, int k) 
{
    int tid = threadIdx.x; // 0..k-1
    int bid = blockIdx.x;  // 0..n-1
    int idx = (tid + bid * blockDim.x) * 8;
    const float epsilon = 1e-5f;
    __shared__ float s_variance; // shared within block
    // temporary register(memory), .local space in ptx, addressable
    half pack_x[8], pack_y[8]; // 8x16 bits=128 bits
    // reinterpret as float4 and load 128 bits in 1 memory issue
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

    float variance = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) 
    {
        float v = __half2float(pack_x[i]);
        variance += ((idx + i) < n * k ? v * v : 0.0f);
    }
    variance = block_reduce_sum_f32<num_threads>(variance);
    if (tid == 0) s_variance = rsqrtf(variance / (float)k + epsilon);
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 8; i += 2) 
    {
        float2 v2 = __half22float2(HALF2(pack_x[i]));
        float2 y2 = { v2.x * s_variance * g, v2.y * s_variance * g };
        HALF2(pack_y[i]) = __float22half2_rn(y2);
    }
    // reinterpret as float4 and store 128 bits in 1 memory issue.
    if ((idx + 7) < n * k) {
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
    // TODO: support non 8-multiple k here
}

static inline void check_cuda_launch(const char* name)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "%s launch failed: %s\n", name, cudaGetErrorString(err));
	}
}

template <int BLOCK_SIZE, typename KernelT>
static inline void launch_rms_norm_kernel_f32(
	KernelT kernel,
	const char* kernel_name,
	float* x,
	float* y,
	float g,
	int N,
	int K,
	cudaStream_t stream)
{
	dim3 grid(N);
	dim3 block(BLOCK_SIZE);
	kernel << <grid, block, 0, stream >> > (x, y, g, N, K);
	check_cuda_launch(kernel_name);
}

template <int BLOCK_SIZE, typename KernelT>
static inline void launch_rms_norm_kernel_f16(
	KernelT kernel,
	const char* kernel_name,
	half* x,
	half* y,
	float g,
	int N,
	int K,
	cudaStream_t stream)
{
	dim3 grid(N);
	dim3 block(BLOCK_SIZE);
	kernel << <grid, block, 0, stream >> > (x, y, g, N, K);
	check_cuda_launch(kernel_name);
}

void rms_norm_f32(float* x, float* y, float g, int N, int K, cudaStream_t stream)
{
	switch (K)
	{
	case 64:
		launch_rms_norm_kernel_f32<64>(
			rms_norm_f32_kernel<64>, "rms_norm_f32", x, y, g, N, K, stream);
		break;
	case 128:
		launch_rms_norm_kernel_f32<128>(
			rms_norm_f32_kernel<128>, "rms_norm_f32", x, y, g, N, K, stream);
		break;
	case 256:
		launch_rms_norm_kernel_f32<256>(
			rms_norm_f32_kernel<256>, "rms_norm_f32", x, y, g, N, K, stream);
		break;
	case 512:
		launch_rms_norm_kernel_f32<512>(
			rms_norm_f32_kernel<512>, "rms_norm_f32", x, y, g, N, K, stream);
		break;
	case 1024:
		launch_rms_norm_kernel_f32<1024>(
			rms_norm_f32_kernel<1024>, "rms_norm_f32", x, y, g, N, K, stream);
		break;
	default:
		fprintf(stderr, "rsm_norm_f32 only supports K: 64/128/256/512/1024\n");
		return;
	}
}

void rms_norm_f32x4(float* x, float* y, float g, int N, int K, cudaStream_t stream)
{
	switch (K)
	{
	case 64:
		launch_rms_norm_kernel_f32<64 / 4>(
			rms_norm_f32x4_kernel<64 / 4>, "rms_norm_f32x4", x, y, g, N, K, stream);
		break;
	case 128:
		launch_rms_norm_kernel_f32<128 / 4>(
			rms_norm_f32x4_kernel<128 / 4>, "rms_norm_f32x4", x, y, g, N, K, stream);
		break;
	case 256:
		launch_rms_norm_kernel_f32<256 / 4>(
			rms_norm_f32x4_kernel<256 / 4>, "rms_norm_f32x4", x, y, g, N, K, stream);
		break;
	case 512:
		launch_rms_norm_kernel_f32<512 / 4>(
			rms_norm_f32x4_kernel<512 / 4>, "rms_norm_f32x4", x, y, g, N, K, stream);
		break;
	case 1024:
		launch_rms_norm_kernel_f32<1024 / 4>(
			rms_norm_f32x4_kernel<1024 / 4>, "rms_norm_f32x4", x, y, g, N, K, stream);
		break;
	case 2048:
		launch_rms_norm_kernel_f32<2048 / 4>(
			rms_norm_f32x4_kernel<2048 / 4>, "rms_norm_f32x4", x, y, g, N, K, stream);
		break;
	case 4096:
		launch_rms_norm_kernel_f32<4096 / 4>(
			rms_norm_f32x4_kernel<4096 / 4>, "rms_norm_f32x4", x, y, g, N, K, stream);
		break;
	default:
		fprintf(stderr, "rms_norm_f32x4 only supports K: 64/128/.../4096\n");
		return;
	}
}

void rms_norm_f16_f16(half* x, half* y, float g, int N, int K, cudaStream_t stream)
{
	switch (K)
	{
	case 64:
		launch_rms_norm_kernel_f16<64>(
			rms_norm_f16_f16_kernel<64>, "rms_norm_f16_f16", x, y, g, N, K, stream);
		break;
	case 128:
		launch_rms_norm_kernel_f16<128>(
			rms_norm_f16_f16_kernel<128>, "rms_norm_f16_f16", x, y, g, N, K, stream);
		break;
	case 256:
		launch_rms_norm_kernel_f16<256>(
			rms_norm_f16_f16_kernel<256>, "rms_norm_f16_f16", x, y, g, N, K, stream);
		break;
	case 512:
		launch_rms_norm_kernel_f16<512>(
			rms_norm_f16_f16_kernel<512>, "rms_norm_f16_f16", x, y, g, N, K, stream);
		break;
	case 1024:
		launch_rms_norm_kernel_f16<1024>(
			rms_norm_f16_f16_kernel<1024>, "rms_norm_f16_f16", x, y, g, N, K, stream);
		break;
	default:
		fprintf(stderr, "rms_norm_f16_f16 only supports K: 64/128/256/512/1024\n");
		return;
	}
}

void rms_norm_f16_f32(half* x, half* y, float g, int N, int K, cudaStream_t stream)
{
	switch (K)
	{
	case 64:
		launch_rms_norm_kernel_f16<64>(
			rms_norm_f16_f32_kernel<64>, "rms_norm_f16_f32", x, y, g, N, K, stream);
		break;
	case 128:
		launch_rms_norm_kernel_f16<128>(
			rms_norm_f16_f32_kernel<128>, "rms_norm_f16_f32", x, y, g, N, K, stream);
		break;
	case 256:
		launch_rms_norm_kernel_f16<256>(
			rms_norm_f16_f32_kernel<256>, "rms_norm_f16_f32", x, y, g, N, K, stream);
		break;
	case 512:
		launch_rms_norm_kernel_f16<512>(
			rms_norm_f16_f32_kernel<512>, "rms_norm_f16_f32", x, y, g, N, K, stream);
		break;
	case 1024:
		launch_rms_norm_kernel_f16<1024>(
			rms_norm_f16_f32_kernel<1024>, "rms_norm_f16_f32", x, y, g, N, K, stream);
		break;
	default:
		fprintf(stderr, "rms_norm_f16_f32 only supports K: 64/128/256/512/1024\n");
		return;
	}
}

void rms_norm_f16x2_f16(half* x, half* y, float g, int N, int K, cudaStream_t stream)
{
	switch (K)
	{
	case 64:
		launch_rms_norm_kernel_f16<64 / 2>(
			rms_norm_f16x2_f16_kernel<64 / 2>, "rms_norm_f16x2_f16", x, y, g, N, K, stream);
		break;
	case 128:
		launch_rms_norm_kernel_f16<128 / 2>(
			rms_norm_f16x2_f16_kernel<128 / 2>, "rms_norm_f16x2_f16", x, y, g, N, K, stream);
		break;
	case 256:
		launch_rms_norm_kernel_f16<256 / 2>(
			rms_norm_f16x2_f16_kernel<256 / 2>, "rms_norm_f16x2_f16", x, y, g, N, K, stream);
		break;
	case 512:
		launch_rms_norm_kernel_f16<512 / 2>(
			rms_norm_f16x2_f16_kernel<512 / 2>, "rms_norm_f16x2_f16", x, y, g, N, K, stream);
		break;
	case 1024:
		launch_rms_norm_kernel_f16<1024 / 2>(
			rms_norm_f16x2_f16_kernel<1024 / 2>, "rms_norm_f16x2_f16", x, y, g, N, K, stream);
		break;
	case 2048:
		launch_rms_norm_kernel_f16<2048 / 2>(
			rms_norm_f16x2_f16_kernel<2048 / 2>, "rms_norm_f16x2_f16", x, y, g, N, K, stream);
		break;
	default:
		fprintf(stderr, "rms_norm_f16x2_f16 only supports K: 64/128/.../2048\n");
		return;
	}
}

void rms_norm_f16x8_f16(half* x, half* y, float g, int N, int K, cudaStream_t stream)
{
	switch (K)
	{
	case 64:
		launch_rms_norm_kernel_f16<64 / 8>(
			rms_norm_f16x8_f16_kernel<64 / 8>, "rms_norm_f16x8_f16", x, y, g, N, K, stream);
		break;
	case 128:
		launch_rms_norm_kernel_f16<128 / 8>(
			rms_norm_f16x8_f16_kernel<128 / 8>, "rms_norm_f16x8_f16", x, y, g, N, K, stream);
		break;
	case 256:
		launch_rms_norm_kernel_f16<256 / 8>(
			rms_norm_f16x8_f16_kernel<256 / 8>, "rms_norm_f16x8_f16", x, y, g, N, K, stream);
		break;
	case 512:
		launch_rms_norm_kernel_f16<512 / 8>(
			rms_norm_f16x8_f16_kernel<512 / 8>, "rms_norm_f16x8_f16", x, y, g, N, K, stream);
		break;
	case 1024:
		launch_rms_norm_kernel_f16<1024 / 8>(
			rms_norm_f16x8_f16_kernel<1024 / 8>, "rms_norm_f16x8_f16", x, y, g, N, K, stream);
		break;
	case 2048:
		launch_rms_norm_kernel_f16<2048 / 8>(
			rms_norm_f16x8_f16_kernel<2048 / 8>, "rms_norm_f16x8_f16", x, y, g, N, K, stream);
		break;
	case 4096:
		launch_rms_norm_kernel_f16<4096 / 8>(
			rms_norm_f16x8_f16_kernel<4096 / 8>, "rms_norm_f16x8_f16", x, y, g, N, K, stream);
		break;
	case 8192:
		launch_rms_norm_kernel_f16<8192 / 8>(
			rms_norm_f16x8_f16_kernel<8192 / 8>, "rms_norm_f16x8_f16", x, y, g, N, K, stream);
		break;
	default:
		fprintf(stderr, "rms_norm_f16x8_f16 only supports K: 64/128/.../8192\n");
		return;
	}
}

void rms_norm_f16x8_f32(half* x, half* y, float g, int N, int K, cudaStream_t stream)
{
	switch (K)
	{
	case 64:
		launch_rms_norm_kernel_f16<64 / 8>(
			rms_norm_f16x8_f32_kernel<64 / 8>, "rms_norm_f16x8_f32", x, y, g, N, K, stream);
		break;
	case 128:
		launch_rms_norm_kernel_f16<128 / 8>(
			rms_norm_f16x8_f32_kernel<128 / 8>, "rms_norm_f16x8_f32", x, y, g, N, K, stream);
		break;
	case 256:
		launch_rms_norm_kernel_f16<256 / 8>(
			rms_norm_f16x8_f32_kernel<256 / 8>, "rms_norm_f16x8_f32", x, y, g, N, K, stream);
		break;
	case 512:
		launch_rms_norm_kernel_f16<512 / 8>(
			rms_norm_f16x8_f32_kernel<512 / 8>, "rms_norm_f16x8_f32", x, y, g, N, K, stream);
		break;
	case 1024:
		launch_rms_norm_kernel_f16<1024 / 8>(
			rms_norm_f16x8_f32_kernel<1024 / 8>, "rms_norm_f16x8_f32", x, y, g, N, K, stream);
		break;
	case 2048:
		launch_rms_norm_kernel_f16<2048 / 8>(
			rms_norm_f16x8_f32_kernel<2048 / 8>, "rms_norm_f16x8_f32", x, y, g, N, K, stream);
		break;
	case 4096:
		launch_rms_norm_kernel_f16<4096 / 8>(
			rms_norm_f16x8_f32_kernel<4096 / 8>, "rms_norm_f16x8_f32", x, y, g, N, K, stream);
		break;
	case 8192:
		launch_rms_norm_kernel_f16<8192 / 8>(
			rms_norm_f16x8_f32_kernel<8192 / 8>, "rms_norm_f16x8_f32", x, y, g, N, K, stream);
		break;
	default:
		fprintf(stderr, "rms_norm_f16x8_f32 only supports K: 64/128/.../8192\n");
		return;
	}
}

void rms_norm_f16x8_pack_f16(half* x, half* y, float g, int N, int K, cudaStream_t stream)
{
	switch (K)
	{
	case 64:
		launch_rms_norm_kernel_f16<64 / 8>(
			rms_norm_f16x8_pack_f16_kernel<64 / 8>, "rms_norm_f16x8_pack_f16", x, y, g, N, K, stream);
		break;
	case 128:
		launch_rms_norm_kernel_f16<128 / 8>(
			rms_norm_f16x8_pack_f16_kernel<128 / 8>, "rms_norm_f16x8_pack_f16", x, y, g, N, K, stream);
		break;
	case 256:
		launch_rms_norm_kernel_f16<256 / 8>(
			rms_norm_f16x8_pack_f16_kernel<256 / 8>, "rms_norm_f16x8_pack_f16", x, y, g, N, K, stream);
		break;
	case 512:
		launch_rms_norm_kernel_f16<512 / 8>(
			rms_norm_f16x8_pack_f16_kernel<512 / 8>, "rms_norm_f16x8_pack_f16", x, y, g, N, K, stream);
		break;
	case 1024:
		launch_rms_norm_kernel_f16<1024 / 8>(
			rms_norm_f16x8_pack_f16_kernel<1024 / 8>, "rms_norm_f16x8_pack_f16", x, y, g, N, K, stream);
		break;
	case 2048:
		launch_rms_norm_kernel_f16<2048 / 8>(
			rms_norm_f16x8_pack_f16_kernel<2048 / 8>, "rms_norm_f16x8_pack_f16", x, y, g, N, K, stream);
		break;
	case 4096:
		launch_rms_norm_kernel_f16<4096 / 8>(
			rms_norm_f16x8_pack_f16_kernel<4096 / 8>, "rms_norm_f16x8_pack_f16", x, y, g, N, K, stream);
		break;
	case 8192:
		launch_rms_norm_kernel_f16<8192 / 8>(
			rms_norm_f16x8_pack_f16_kernel<8192 / 8>, "rms_norm_f16x8_pack_f16", x, y, g, N, K, stream);
		break;
	default:
		fprintf(stderr, "rms_norm_f16x8_pack_f16 only supports K: 64/128/.../8192\n");
		return;
	}
}

void rms_norm_f16x8_pack_f32(half* x, half* y, float g, int N, int K, cudaStream_t stream)
{
	switch (K)
	{
	case 64:
		launch_rms_norm_kernel_f16<64 / 8>(
			rms_norm_f16x8_pack_f32_kernel<64 / 8>, "rms_norm_f16x8_pack_f32", x, y, g, N, K, stream);
		break;
	case 128:
		launch_rms_norm_kernel_f16<128 / 8>(
			rms_norm_f16x8_pack_f32_kernel<128 / 8>, "rms_norm_f16x8_pack_f32", x, y, g, N, K, stream);
		break;
	case 256:
		launch_rms_norm_kernel_f16<256 / 8>(
			rms_norm_f16x8_pack_f32_kernel<256 / 8>, "rms_norm_f16x8_pack_f32", x, y, g, N, K, stream);
		break;
	case 512:
		launch_rms_norm_kernel_f16<512 / 8>(
			rms_norm_f16x8_pack_f32_kernel<512 / 8>, "rms_norm_f16x8_pack_f32", x, y, g, N, K, stream);
		break;
	case 1024:
		launch_rms_norm_kernel_f16<1024 / 8>(
			rms_norm_f16x8_pack_f32_kernel<1024 / 8>, "rms_norm_f16x8_pack_f32", x, y, g, N, K, stream);
		break;
	case 2048:
		launch_rms_norm_kernel_f16<2048 / 8>(
			rms_norm_f16x8_pack_f32_kernel<2048 / 8>, "rms_norm_f16x8_pack_f32", x, y, g, N, K, stream);
		break;
	case 4096:
		launch_rms_norm_kernel_f16<4096 / 8>(
			rms_norm_f16x8_pack_f32_kernel<4096 / 8>, "rms_norm_f16x8_pack_f32", x, y, g, N, K, stream);
		break;
	case 8192:
		launch_rms_norm_kernel_f16<8192 / 8>(
			rms_norm_f16x8_pack_f32_kernel<8192 / 8>, "rms_norm_f16x8_pack_f32", x, y, g, N, K, stream);
		break;
	default:
		fprintf(stderr, "rms_norm_f16x8_pack_f32 only supports K: 64/128/.../8192\n");
		return;
	}
}