#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#include "layer_norm.cuh"

#define WARP_SIZE   256
#define WARP_SIZE_S 16
#define PAD 1
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

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

// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc.
// grid 1D block 1D, grid(N/256), block(256)
template <const int num_threads = WARP_SIZE>
__device__ float block_reduce_sum_f32(float val)
{
	// always <= 32 warps per block (limited by 1024 threads per block)
	constexpr int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
	int warp = threadIdx.x / WARP_SIZE;
	int lane = threadIdx.x % WARP_SIZE;
	static __shared__ float shared[num_warps];

	val = warp_reduce_sum_f32<WARP_SIZE>(val);
	if (lane == 0) shared[warp] = val;
	__syncthreads();
	val = (lane < num_threads) ? shared[lane] : 0.0f;
	val = warp_reduce_sum_f32<num_warps>(val);

	return val;
}

// Layer Norm: x: NxK(K=256<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template <const int num_threads = WARP_SIZE>
__global__ void layer_norm_f32_kernel(float* x, float* y, float g, float b, int n, int k)
{
	int tid = threadIdx.x; // 0...k-1
	int bid = blockIdx.x; // 0...n-1
	int idx = tid + bid * blockDim.x;
	const float epsilon = 1e-5f;

	__shared__ float s_mean;          // shared within block
	__shared__ float s_variance;      // shared within block
	float value = (idx < n* k) ? x[idx] : 0.0f; // load once only
	float sum = block_reduce_sum_f32<num_threads>(value);
	if (tid == 0) s_mean = sum / (float)k;
	// wait for s_mean in shared memory to be ready for all threads
	__syncthreads();
	float variance = (value - s_mean) * (value - s_mean);
	variance = block_reduce_sum_f32<num_threads>(variance);
	if (tid == 0) s_variance = rsqrtf(variance / (float)k + epsilon);
	// wait for s_variance in shared memory to be ready for all threads
	__syncthreads();
	
	if (idx < n * k) y[idx] = ((value - s_mean) * s_variance) * g + b;
}

// Layer Norm Vec4: x: NxK(K=256<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K/4<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template <const int num_threads = WARP_SIZE / 4>
__global__ void layer_norm_f32x4_kernel(float* x, float* y, float g, float b, int n, int k) 
{
	int tid = threadIdx.x; // 0...k-1
	int bid = blockIdx.x; // 0...n-1
	int idx = (tid + bid * blockDim.x) * 4;
	const float epsilon = 1e-5f;

	__shared__ float s_mean;     // shared within block
	__shared__ float s_variance; // shared within block
	float4 reg_x = FLOAT4(x[idx]);
	float value = (idx < n * k) ? (reg_x.x + reg_x.y + reg_x.z + reg_x.w) : 0.0f;
	float sum = block_reduce_sum_f32<num_threads>(value);
	if (tid == 0) s_mean = sum / (float)k;
	// wait for s_mean in shared memory to be ready for all threads
	__syncthreads();
	float4 reg_x_hat;
	reg_x_hat.x = reg_x.x - s_mean;
	reg_x_hat.y = reg_x.y - s_mean;
	reg_x_hat.z = reg_x.z - s_mean;
	reg_x_hat.w = reg_x.w - s_mean;
	float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y +
		reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
	variance = block_reduce_sum_f32<num_threads>(variance);
	if (tid == 0) s_variance = rsqrtf(variance / (float)k + epsilon);
	// wait for s_variance in shared memory to be ready for all threads
	__syncthreads();
	float4 reg_y;
	reg_y.x = reg_x_hat.x * s_variance * g + b;
	reg_y.y = reg_x_hat.y * s_variance * g + b;
	reg_y.z = reg_x_hat.z * s_variance * g + b;
	reg_y.w = reg_x_hat.w * s_variance * g + b;
	if (idx < n * k) FLOAT4(y[idx]) = reg_y;
}

// FP16
// Warp Reduce Sum: Half
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

template <const int num_threads = WARP_SIZE>
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

template <const int num_threads = WARP_SIZE>
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

template <const int num_threads = WARP_SIZE>
__global__ void layer_norm_f16_f16_kernel(half* x, half* y, float g, float b, int n, int k) 
{
	int tid = threadIdx.x; // 0..k-1
	int bid = blockIdx.x;  // 0..n-1
	int idx = tid + bid * blockDim.x;
	const half epsilon = __float2half(1e-5f);
	const half g_ = __float2half(g);
	const half b_ = __float2half(b);
	const half k_ = __int2half_rn(k);

	__shared__ half s_mean;     // shared within block
	__shared__ half s_variance; // shared within block
	half value = (idx < n * k) ? x[idx] : __float2half(0.0f); // load once only
	half sum = block_reduce_sum_f16_f16<num_threads>(value);
	if (tid == 0) s_mean = sum / k_;
	// wait for s_mean in shared memory to be ready for all threads
	__syncthreads();
	half variance = (value - s_mean) * (value - s_mean);
	variance = block_reduce_sum_f16_f16<num_threads>(variance);
	if (tid == 0) s_variance = hrsqrt(variance / k_ + epsilon);
	// wait for s_variance in shared memory to be ready for all threads
	__syncthreads();
	if (idx < n * k) {
		y[idx] = ((value - s_mean) * s_variance) * g_ + b_;
		// y[idx] = ((value - s_mean) * s_variance) * g_ + b_;
	}
}

template <const int num_threads = WARP_SIZE>
__global__ void layer_norm_f16x2_f16_kernel(half* x, half* y, float g, float b, int n, int k) 
{
	int tid = threadIdx.x; // 0..k-1
	int bid = blockIdx.x;  // 0..n-1
	int idx = (tid + bid * blockDim.x) * 2;
	const half epsilon = __float2half(1e-5f);
	const half g_ = __float2half(g);
	const half b_ = __float2half(b);
	const half k_ = __int2half_rn(k);

	__shared__ half s_mean;     // shared within block
	__shared__ half s_variance; // shared within block
	half2 reg_x = HALF2(x[idx]);
	half value = (idx < n * k) ? (reg_x.x + reg_x.y) : __float2half(0.0f);
	half sum = block_reduce_sum_f16_f16<num_threads>(value);
	if (tid == 0) s_mean = sum / k_;
	// wait for s_mean in shared memory to be ready for all threads
	__syncthreads();
	half2 reg_x_hat;
	reg_x_hat.x = reg_x.x - s_mean;
	reg_x_hat.y = reg_x.y - s_mean;
	half variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y;
	variance = block_reduce_sum_f16_f16<num_threads>(variance);
	if (tid == 0) s_variance = hrsqrt(variance / k_ + epsilon);
	// wait for s_variance in shared memory to be ready for all threads
	__syncthreads();
	if (idx < n * k) {
		half2 reg_y;
		reg_y.x = reg_x_hat.x * s_variance * g_ + b_;
		reg_y.y = reg_x_hat.y * s_variance * g_ + b_;
		HALF2(y[idx]) = reg_y;
	}
}

#define HALF2_SUM(reg, i)                                                      \
  (((idx + (i)) < n * k) ? ((reg).x + (reg).y) : __float2half(0.0f))

#define HALF2_SUB(reg_y, reg_x)                                                \
  (reg_y).x = (reg_x).x - s_mean;                                              \
  (reg_y).y = (reg_x).y - s_mean;

#define HALF2_VARIANCE(reg, i)                                                 \
  (((idx + (i)) < n * k) ? ((reg).x * (reg).x + (reg).y * (reg).y)             \
                         : __float2half(0.0f))

#define HALF2_LAYER_NORM(reg_y, reg_x, g_, b_)                                 \
    (reg_y).x = (reg_x).x * s_variance * (g_) + (b_);                          \
    (reg_y).y = (reg_x).y * s_variance * (g_) + (b_);

template <const int num_threads = WARP_SIZE>
__global__ void layer_norm_f16x8_f16_kernel(half* x, half* y, float g, float b, int n, int k)
{
	int tid = threadIdx.x; // 0..k-1
	int bid = blockIdx.x;  // 0..n-1
	int idx = (tid + bid * blockDim.x) * 8;
	const half epsilon = __float2half(1e-5f);
	const half g_ = __float2half(g);
	const half b_ = __float2half(b);
	const half k_ = __int2half_rn(k);

	__shared__ half s_mean;     // shared within block
	__shared__ half s_variance; // shared within block
	half2 reg_x_0 = HALF2(x[idx + 0]);
	half2 reg_x_1 = HALF2(x[idx + 2]);
	half2 reg_x_2 = HALF2(x[idx + 4]);
	half2 reg_x_3 = HALF2(x[idx + 6]);

	half value = HALF2_SUM(reg_x_0, 0);
	value += HALF2_SUM(reg_x_1, 2);
	value += HALF2_SUM(reg_x_2, 4);
	value += HALF2_SUM(reg_x_3, 6);
	
	half sum = block_reduce_sum_f16_f16<num_threads>(value);
	if (tid == 0) s_mean = sum / k_;
	// wait for s_mean in shared memory to be ready for all threads
	__syncthreads();
	// manual unroll
	half2 reg_x_hat_0, reg_x_hat_1, reg_x_hat_2, reg_x_hat_3;
	HALF2_SUB(reg_x_hat_0, reg_x_0);
	HALF2_SUB(reg_x_hat_1, reg_x_1);
	HALF2_SUB(reg_x_hat_2, reg_x_2);
	HALF2_SUB(reg_x_hat_3, reg_x_3);

	half variance = HALF2_VARIANCE(reg_x_hat_0, 0);
	variance += HALF2_VARIANCE(reg_x_hat_1, 2);
	variance += HALF2_VARIANCE(reg_x_hat_2, 4);
	variance += HALF2_VARIANCE(reg_x_hat_3, 6);

	variance = block_reduce_sum_f16_f16<num_threads>(variance);
	if (tid == 0) s_variance = hrsqrt(variance / k_ + epsilon);
	// wait for s_variance in shared memory to be ready for all threads
	__syncthreads();
	// manual unroll
	half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
	HALF2_LAYER_NORM(reg_y_0, reg_x_hat_0, g_, b_);
	HALF2_LAYER_NORM(reg_y_1, reg_x_hat_1, g_, b_);
	HALF2_LAYER_NORM(reg_y_2, reg_x_hat_2, g_, b_);
	HALF2_LAYER_NORM(reg_y_3, reg_x_hat_3, g_, b_);

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

template <const int num_threads = WARP_SIZE>
__global__ void layer_norm_f16_f32_kernel(half* x, half* y, float g, float b, int n, int k) 
{
	int tid = threadIdx.x; // 0..k-1
	int bid = blockIdx.x;  // 0..n-1
	int idx = tid + bid * blockDim.x;
	const float epsilon = 1e-5f;

	__shared__ float s_mean;     // shared within block
	__shared__ float s_variance; // shared within block
	float value = (idx < n * k) ? __half2float(x[idx]) : 0.0f; // load once only
	float sum = block_reduce_sum_f32<num_threads>(value);
	if (tid == 0) s_mean = sum / (float)k;
	// wait for s_mean in shared memory to be ready for all threads
	__syncthreads();
	float variance = (value - s_mean) * (value - s_mean);
	variance = block_reduce_sum_f32<num_threads>(variance);
	if (tid == 0) s_variance = rsqrtf(variance / (float)k + epsilon);
	// wait for s_variance in shared memory to be ready for all threads
	__syncthreads();
	if (idx < n * k) {
		// x*y + z -> x'*g + b
		y[idx] = __float2half(__fmaf_rn(((value - s_mean) * s_variance), g, b));
	}
}

template <const int num_threads = WARP_SIZE>
__global__ void layer_norm_f16x8_pack_f16_kernel(half* x, half* y, float g, float b, int n, int k) 
{
	int tid = threadIdx.x; // 0..k-1
	int bid = blockIdx.x;  // 0..n-1
	int idx = (tid + bid * blockDim.x) * 8;
	const half epsilon = __float2half(1e-5f);
	const half g_ = __float2half(g);
	const half b_ = __float2half(b);
	const half k_ = __int2half_rn(k);
	const half z_ = __float2half(0.0f);

	__shared__ half s_mean;     // shared within block
	__shared__ half s_variance; // shared within block
	// temporary register(memory), .local space in ptx, addressable
	half pack_x[8], pack_y[8]; // 8x16 bits=128 bits
	// reinterpret as float4 and load 128 bits in 1 memory issue
	LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

	half value = z_;
#pragma unroll
	for (int i = 0; i < 8; ++i) 
	{
		value += ((idx + i) < n * k ? pack_x[i] : z_);
	}
	half sum = block_reduce_sum_f16_f16<num_threads>(value);
	if (tid == 0) s_mean = sum / k_;
	// wait for s_mean in shared memory to be ready for all threads
	__syncthreads();

	half variance = z_;
#pragma unroll
	for (int i = 0; i < 8; ++i) 
	{
		half v_hat = pack_x[i] - s_mean;
		variance += ((idx + i) < n * k ? v_hat * v_hat : z_);
	}
	variance = block_reduce_sum_f16_f16<num_threads>(variance);
	if (tid == 0) s_variance = hrsqrt(variance / k_ + epsilon);
	// wait for s_variance in shared memory to be ready for all threads
	__syncthreads();

#pragma unroll
	for (int i = 0; i < 8; ++i) 
	{
		// TODO: use __hfma2, __hsub2, __hmul2 here
		pack_y[i] = ((pack_x[i] - s_mean) * s_variance) * g_ + b_;
	}
	// reinterpret as float4 and store 128 bits in 1 memory issue
	if ((idx + 7) < n * k) {
		LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
	}
	// TODO: support non 8-multiple K here
}

template <const int num_threads = WARP_SIZE>
__global__ void layer_norm_f16x8_pack_f32_kernel(half* x, half* y, float g, float b, int n, int k) 
{
	int tid = threadIdx.x; // 0..k-1
	int bid = blockIdx.x;  // 0..n-1
	int idx = (tid + bid * blockDim.x) * 8;
	const float epsilon = 1e-5f;

	__shared__ float s_mean;     // shared within block
	__shared__ float s_variance; // shared within block
	// temporary register(memory), .local space in ptx, addressable
	half pack_x[8], pack_y[8]; // 8x16 bits=128 bits
	// reinterpret as float4 and load 128 bits in 1 memory issue
	LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

	float value = 0.0f;
#pragma unroll
	for (int i = 0; i < 8; ++i) 
	{
		value += ((idx + i) < n * k ? __half2float(pack_x[i]) : 0.0f);
	}
	float sum = block_reduce_sum_f32<num_threads>(value);
	if (tid == 0) s_mean = sum / (float)k;
	// wait for s_mean in shared memory to be ready for all threads
	__syncthreads();

	float variance = 0.0f;
#pragma unroll
	for (int i = 0; i < 8; ++i) 
	{
		float v_hat = __half2float(pack_x[i]) - s_mean;
		variance += ((idx + i) < n * k ? v_hat * v_hat : 0.0f);
	}
	variance = block_reduce_sum_f32<num_threads>(variance);
	if (tid == 0) s_variance = rsqrtf(variance / (float)k + epsilon);
	// wait for s_variance in shared memory to be ready for all threads
	__syncthreads();

#pragma unroll
	for (int i = 0; i < 8; ++i)
	{
		pack_y[i] = __float2half(
			__fmaf_rn(((__half2float(pack_x[i]) - s_mean) * s_variance), g, b));
	}
	// reinterpret as float4 and store 128 bits in 1 memory issue
	if ((idx + 7) < n * k) {
		LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
	}
	// TODO: support non 8-multiple K here
}

static inline void check_cuda_launch(const char* name) 
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "%s launch failed: %s\n", name, cudaGetErrorString(err));
	}
}

template <int BLOCK_SIZE, typename KernelT>
static inline void launch_layer_norm_kernel_f32(
	KernelT kernel,
	const char* kernel_name,
	float* x,
	float* y,
	float g,
	float b,
	int N,
	int K,
	cudaStream_t stream) 
{
	dim3 grid(N);
	dim3 block(BLOCK_SIZE);
	kernel << <grid, block, 0, stream >> > (x, y, g, b, N, K);
	check_cuda_launch(kernel_name);
}

template <int BLOCK_SIZE, typename KernelT>
static inline void launch_layer_norm_kernel_f16(
	KernelT kernel,
	const char* kernel_name,
	half* x,
	half* y,
	float g,
	float b,
	int N,
	int K,
	cudaStream_t stream) 
{
	dim3 grid(N);
	dim3 block(BLOCK_SIZE);
	kernel << <grid, block, 0, stream >> > (x, y, g, b, N, K);
	check_cuda_launch(kernel_name);
}

void layer_norm_f32(float* x, float* y, float g, float b, int N, int K, cudaStream_t stream)
{
	switch (K) 
	{
	case 64:
		launch_layer_norm_kernel_f32<64>(
			layer_norm_f32_kernel<64>, "layer_norm_f32", x, y, g, b, N, K, stream);
		break;
	case 128:
		launch_layer_norm_kernel_f32<128>(
			layer_norm_f32_kernel<128>, "layer_norm_f32", x, y, g, b, N, K, stream);
		break;
	case 256:
		launch_layer_norm_kernel_f32<256>(
			layer_norm_f32_kernel<256>, "layer_norm_f32", x, y, g, b, N, K, stream);
		break;
	case 512:
		launch_layer_norm_kernel_f32<512>(
			layer_norm_f32_kernel<512>, "layer_norm_f32", x, y, g, b, N, K, stream);
		break;
	case 1024:
		launch_layer_norm_kernel_f32<1024>(
			layer_norm_f32_kernel<1024>, "layer_norm_f32", x, y, g, b, N, K, stream);
		break;
	default:
		fprintf(stderr, "layer_norm_f32 only supports K: 64/128/256/512/1024\n");
		return;
	}
}

void layer_norm_f32x4(float* x, float* y, float g, float b, int N, int K, cudaStream_t stream) 
{
	switch (K) 
	{
	case 64:
		launch_layer_norm_kernel_f32<64 / 4>(
			layer_norm_f32x4_kernel<64 / 4>, "layer_norm_f32x4", x, y, g, b, N, K, stream);
		break;
	case 128:
		launch_layer_norm_kernel_f32<128 / 4>(
			layer_norm_f32x4_kernel<128 / 4>, "layer_norm_f32x4", x, y, g, b, N, K, stream);
		break;
	case 256:
		launch_layer_norm_kernel_f32<256 / 4>(
			layer_norm_f32x4_kernel<256 / 4>, "layer_norm_f32x4", x, y, g, b, N, K, stream);
		break;
	case 512:
		launch_layer_norm_kernel_f32<512 / 4>(
			layer_norm_f32x4_kernel<512 / 4>, "layer_norm_f32x4", x, y, g, b, N, K, stream);
		break;
	case 1024:
		launch_layer_norm_kernel_f32<1024 / 4>(
			layer_norm_f32x4_kernel<1024 / 4>, "layer_norm_f32x4", x, y, g, b, N, K, stream);
		break;
	case 2048:
		launch_layer_norm_kernel_f32<2048 / 4>(
			layer_norm_f32x4_kernel<2048 / 4>, "layer_norm_f32x4", x, y, g, b, N, K, stream);
		break;
	case 4096:
		launch_layer_norm_kernel_f32<4096 / 4>(
			layer_norm_f32x4_kernel<4096 / 4>, "layer_norm_f32x4", x, y, g, b, N, K, stream);
		break;
	default:
		fprintf(stderr, "layer_norm_f32x4 only supports K: 64/128/.../4096\n");
		return;
	}
}

void layer_norm_f16_f16(half* x, half* y, float g, float b, int N, int K, cudaStream_t stream) 
{
	switch (K)
	{
	case 64:
		launch_layer_norm_kernel_f16<64>(
			layer_norm_f16_f16_kernel<64>, "layer_norm_f16_f16", x, y, g, b, N, K, stream);
		break;
	case 128:
		launch_layer_norm_kernel_f16<128>(
			layer_norm_f16_f16_kernel<128>, "layer_norm_f16_f16", x, y, g, b, N, K, stream);
		break;
	case 256:
		launch_layer_norm_kernel_f16<256>(
			layer_norm_f16_f16_kernel<256>, "layer_norm_f16_f16", x, y, g, b, N, K, stream);
		break;
	case 512:
		launch_layer_norm_kernel_f16<512>(
			layer_norm_f16_f16_kernel<512>, "layer_norm_f16_f16", x, y, g, b, N, K, stream);
		break;
	case 1024:
		launch_layer_norm_kernel_f16<1024>(
			layer_norm_f16_f16_kernel<1024>, "layer_norm_f16_f16", x, y, g, b, N, K, stream);
		break;
	default:
		fprintf(stderr, "layer_norm_f16_f16 only supports K: 64/128/256/512/1024\n");
		return;
	}
}

void layer_norm_f16_f32(half* x, half* y, float g, float b, int N, int K, cudaStream_t stream) 
{
	switch (K) 
	{
	case 64:
		launch_layer_norm_kernel_f16<64>(
			layer_norm_f16_f32_kernel<64>, "layer_norm_f16_f32", x, y, g, b, N, K, stream);
		break;
	case 128:
		launch_layer_norm_kernel_f16<128>(
			layer_norm_f16_f32_kernel<128>, "layer_norm_f16_f32", x, y, g, b, N, K, stream);
		break;
	case 256:
		launch_layer_norm_kernel_f16<256>(
			layer_norm_f16_f32_kernel<256>, "layer_norm_f16_f32", x, y, g, b, N, K, stream);
		break;
	case 512:
		launch_layer_norm_kernel_f16<512>(
			layer_norm_f16_f32_kernel<512>, "layer_norm_f16_f32", x, y, g, b, N, K, stream);
		break;
	case 1024:
		launch_layer_norm_kernel_f16<1024>(
			layer_norm_f16_f32_kernel<1024>, "layer_norm_f16_f32", x, y, g, b, N, K, stream);
		break;
	default:
		fprintf(stderr, "layer_norm_f16_f32 only supports K: 64/128/256/512/1024\n");
		return;
	}
}

void layer_norm_f16x2_f16(half* x, half* y, float g, float b, int N, int K, cudaStream_t stream) 
{
	switch (K) 
	{
	case 64:
		launch_layer_norm_kernel_f16<64 / 2>(
			layer_norm_f16x2_f16_kernel<64 / 2>, "layer_norm_f16x2_f16", x, y, g, b, N, K, stream);
		break;
	case 128:
		launch_layer_norm_kernel_f16<128 / 2>(
			layer_norm_f16x2_f16_kernel<128 / 2>, "layer_norm_f16x2_f16", x, y, g, b, N, K, stream);
		break;
	case 256:
		launch_layer_norm_kernel_f16<256 / 2>(
			layer_norm_f16x2_f16_kernel<256 / 2>, "layer_norm_f16x2_f16", x, y, g, b, N, K, stream);
		break;
	case 512:
		launch_layer_norm_kernel_f16<512 / 2>(
			layer_norm_f16x2_f16_kernel<512 / 2>, "layer_norm_f16x2_f16", x, y, g, b, N, K, stream);
		break;
	case 1024:
		launch_layer_norm_kernel_f16<1024 / 2>(
			layer_norm_f16x2_f16_kernel<1024 / 2>, "layer_norm_f16x2_f16", x, y, g, b, N, K, stream);
		break;
	case 2048:
		launch_layer_norm_kernel_f16<2048 / 2>(
			layer_norm_f16x2_f16_kernel<2048 / 2>, "layer_norm_f16x2_f16", x, y, g, b, N, K, stream);
		break;
	default:
		fprintf(stderr, "layer_norm_f16x2_f16 only supports K: 64/128/.../2048\n");
		return;
	}
}

void layer_norm_f16x8_f16(half* x, half* y, float g, float b, int N, int K, cudaStream_t stream) 
{
	switch (K) 
	{
	case 64:
		launch_layer_norm_kernel_f16<64 / 8>(
			layer_norm_f16x8_f16_kernel<64 / 8>, "layer_norm_f16x8_f16", x, y, g, b, N, K, stream);
		break;
	case 128:
		launch_layer_norm_kernel_f16<128 / 8>(
			layer_norm_f16x8_f16_kernel<128 / 8>, "layer_norm_f16x8_f16", x, y, g, b, N, K, stream);
		break;
	case 256:
		launch_layer_norm_kernel_f16<256 / 8>(
			layer_norm_f16x8_f16_kernel<256 / 8>, "layer_norm_f16x8_f16", x, y, g, b, N, K, stream);
		break;
	case 512:
		launch_layer_norm_kernel_f16<512 / 8>(
			layer_norm_f16x8_f16_kernel<512 / 8>, "layer_norm_f16x8_f16", x, y, g, b, N, K, stream);
		break;
	case 1024:
		launch_layer_norm_kernel_f16<1024 / 8>(
			layer_norm_f16x8_f16_kernel<1024 / 8>, "layer_norm_f16x8_f16", x, y, g, b, N, K, stream);
		break;
	case 2048:
		launch_layer_norm_kernel_f16<2048 / 8>(
			layer_norm_f16x8_f16_kernel<2048 / 8>, "layer_norm_f16x8_f16", x, y, g, b, N, K, stream);
		break;
	case 4096:
		launch_layer_norm_kernel_f16<4096 / 8>(
			layer_norm_f16x8_f16_kernel<4096 / 8>, "layer_norm_f16x8_f16", x, y, g, b, N, K, stream);
		break;
	case 8192:
		launch_layer_norm_kernel_f16<8192 / 8>(
			layer_norm_f16x8_f16_kernel<8192 / 8>, "layer_norm_f16x8_f16", x, y, g, b, N, K, stream);
		break;
	default:
		fprintf(stderr, "layer_norm_f16x8_f16 only supports K: 64/128/.../8192\n");
		return;
	}
}

void layer_norm_f16x8_pack_f16(half* x, half* y, float g, float b, int N, int K, cudaStream_t stream) 
{
	switch (K) 
	{
	case 64:
		launch_layer_norm_kernel_f16<64 / 8>(
			layer_norm_f16x8_pack_f16_kernel<64 / 8>, "layer_norm_f16x8_pack_f16", x, y, g, b, N, K, stream);
		break;
	case 128:
		launch_layer_norm_kernel_f16<128 / 8>(
			layer_norm_f16x8_pack_f16_kernel<128 / 8>, "layer_norm_f16x8_pack_f16", x, y, g, b, N, K, stream);
		break;
	case 256:
		launch_layer_norm_kernel_f16<256 / 8>(
			layer_norm_f16x8_pack_f16_kernel<256 / 8>, "layer_norm_f16x8_pack_f16", x, y, g, b, N, K, stream);
		break;
	case 512:
		launch_layer_norm_kernel_f16<512 / 8>(
			layer_norm_f16x8_pack_f16_kernel<512 / 8>, "layer_norm_f16x8_pack_f16", x, y, g, b, N, K, stream);
		break;
	case 1024:
		launch_layer_norm_kernel_f16<1024 / 8>(
			layer_norm_f16x8_pack_f16_kernel<1024 / 8>, "layer_norm_f16x8_pack_f16", x, y, g, b, N, K, stream);
		break;
	case 2048:
		launch_layer_norm_kernel_f16<2048 / 8>(
			layer_norm_f16x8_pack_f16_kernel<2048 / 8>, "layer_norm_f16x8_pack_f16", x, y, g, b, N, K, stream);
		break;
	case 4096:
		launch_layer_norm_kernel_f16<4096 / 8>(
			layer_norm_f16x8_pack_f16_kernel<4096 / 8>, "layer_norm_f16x8_pack_f16", x, y, g, b, N, K, stream);
		break;
	case 8192:
		launch_layer_norm_kernel_f16<8192 / 8>(
			layer_norm_f16x8_pack_f16_kernel<8192 / 8>, "layer_norm_f16x8_pack_f16", x, y, g, b, N, K, stream);
		break;
	default:
		fprintf(stderr, "layer_norm_f16x8_pack_f16 only supports K: 64/128/.../8192\n");
		return;
	}
}

void layer_norm_f16x8_pack_f32(half* x, half* y, float g, float b, int N, int K, cudaStream_t stream) 
{
	switch (K) 
	{
	case 64:
		launch_layer_norm_kernel_f16<64 / 8>(
			layer_norm_f16x8_pack_f32_kernel<64 / 8>, "layer_norm_f16x8_pack_f32", x, y, g, b, N, K, stream);
		break;
	case 128:
		launch_layer_norm_kernel_f16<128 / 8>(
			layer_norm_f16x8_pack_f32_kernel<128 / 8>, "layer_norm_f16x8_pack_f32", x, y, g, b, N, K, stream);
		break;
	case 256:
		launch_layer_norm_kernel_f16<256 / 8>(
			layer_norm_f16x8_pack_f32_kernel<256 / 8>, "layer_norm_f16x8_pack_f32", x, y, g, b, N, K, stream);
		break;
	case 512:
		launch_layer_norm_kernel_f16<512 / 8>(
			layer_norm_f16x8_pack_f32_kernel<512 / 8>, "layer_norm_f16x8_pack_f32", x, y, g, b, N, K, stream);
		break;
	case 1024:
		launch_layer_norm_kernel_f16<1024 / 8>(
			layer_norm_f16x8_pack_f32_kernel<1024 / 8>, "layer_norm_f16x8_pack_f32", x, y, g, b, N, K, stream);
		break;
	case 2048:
		launch_layer_norm_kernel_f16<2048 / 8>(
			layer_norm_f16x8_pack_f32_kernel<2048 / 8>, "layer_norm_f16x8_pack_f32", x, y, g, b, N, K, stream);
		break;
	case 4096:
		launch_layer_norm_kernel_f16<4096 / 8>(
			layer_norm_f16x8_pack_f32_kernel<4096 / 8>, "layer_norm_f16x8_pack_f32", x, y, g, b, N, K, stream);
		break;
	case 8192:
		launch_layer_norm_kernel_f16<8192 / 8>(
			layer_norm_f16x8_pack_f32_kernel<8192 / 8>, "layer_norm_f16x8_pack_f32", x, y, g, b, N, K, stream);
		break;
	default:
		fprintf(stderr, "layer_norm_f16x8_pack_f32 only supports K: 64/128/.../8192\n");
		return;
	}
}