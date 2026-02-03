#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "sigmoid.cuh"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

__device__ __forceinline__ half hexp_compat(half a) 
{
	return __float2half(__expf(__half2float(a)));
}

// FP32
// Sigmoid x: n, y: n y=1/(1+exp(-x))
// grid(N/256), block(K=256)
__global__ void sigmoid_f32_kernel(float* x, float* y, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < n) {
		float v = x[idx];
		v = fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32);
		y[idx] = 1.0f / (1.0f + expf(-v));
	}
}

// Sigmoid x: n, y: n y=1/(1+exp(-x)) Vec4
// grid(n/256), block(256/4)
__global__ void sigmoid_f32x4_kernel(float* x, float* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
	float4 reg_x = FLOAT4(x[idx]);
	float4 reg_y;

	reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
	reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
	reg_x.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
	reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);

	reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
	reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
	reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));
	reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));

	if ((idx + 0) < n) {
		FLOAT4(y[idx]) = reg_y;
	}
}

// FP16
__global__ void sigmoid_f16_kernel(half* x, half* y, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const half f = __float2half(1.0f);

	if (idx < n) {
		half v = x[idx];
		v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);
		y[idx] = f / (f + hexp_compat(-v));
	}
}

__global__ void sigmoid_f16x2_kernel(half* x, half* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
	const half f = __float2half(1.0f);
	half2 reg_x = HALF2(x[idx]);
	half2 reg_y;

	reg_x.x = __hmin(__hmax(reg_x.x, MIN_EXP_F16), MAX_EXP_F16);
	reg_x.y = __hmin(__hmax(reg_x.y, MIN_EXP_F16), MAX_EXP_F16);

	reg_y.x = f / (f + hexp_compat(-reg_x.x));
	reg_y.y = f / (f + hexp_compat(-reg_x.y));

	if ((idx + 0) < n) {
		HALF2(y[idx]) = reg_y;
	}
}

// unpack f16x8
__global__ void sigmoid_f16x8_kernel(half* x, half* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
	const half f = __float2half(1.0f);

	half2 reg_x_0 = HALF2(x[idx + 0]);
	half2 reg_x_1 = HALF2(x[idx + 2]);
	half2 reg_x_2 = HALF2(x[idx + 4]);
	half2 reg_x_3 = HALF2(x[idx + 6]);

	reg_x_0.x = __hmin(__hmax(reg_x_0.x, MIN_EXP_F16), MAX_EXP_F16);
	reg_x_0.y = __hmin(__hmax(reg_x_0.y, MIN_EXP_F16), MAX_EXP_F16);
	reg_x_1.x = __hmin(__hmax(reg_x_1.x, MIN_EXP_F16), MAX_EXP_F16);
	reg_x_1.y = __hmin(__hmax(reg_x_1.y, MIN_EXP_F16), MAX_EXP_F16);
	reg_x_2.x = __hmin(__hmax(reg_x_2.x, MIN_EXP_F16), MAX_EXP_F16);
	reg_x_2.y = __hmin(__hmax(reg_x_2.y, MIN_EXP_F16), MAX_EXP_F16);
	reg_x_3.x = __hmin(__hmax(reg_x_3.x, MIN_EXP_F16), MAX_EXP_F16);
	reg_x_3.y = __hmin(__hmax(reg_x_3.y, MIN_EXP_F16), MAX_EXP_F16);

	half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;

	reg_y_0.x = f / (f + hexp_compat(-reg_x_0.x));
	reg_y_0.y = f / (f + hexp_compat(-reg_x_0.y));
	reg_y_1.x = f / (f + hexp_compat(-reg_x_1.x));
	reg_y_1.y = f / (f + hexp_compat(-reg_x_1.y));
	reg_y_2.x = f / (f + hexp_compat(-reg_x_2.x));
	reg_y_2.y = f / (f + hexp_compat(-reg_x_2.y));
	reg_y_3.x = f / (f + hexp_compat(-reg_x_3.x));
	reg_y_3.y = f / (f + hexp_compat(-reg_x_3.y));

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

// pack f16x8
__global__ void sigmoid_f16x8_pack_kernel(half* x, half* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
	const half f = __float2half(1.0f);
	// temporary register(memory), .local space in ptx, addressable
	half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
	// reinterpret as float4 and load 128 bits in 1 memory issue.
	LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

#pragma unroll
	for (int i = 0; i < 8; ++i)
	{
		half v = __hmin(__hmax(pack_x[i], MIN_EXP_F16), MAX_EXP_F16);
		pack_y[i] = f / (f + hexp_compat(-v));
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

void sigmoid_f32(float* x, float* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 1, grid, block, N);
	sigmoid_f32_kernel << <grid, block, 0, stream >> > (x, y, (int)N);
}

void sigmoid_f32x4(float* x, float* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream) 
{
	if ((N % 4) != 0 || (ndim == 2 && (K % 4) != 0)) {
		sigmoid_f32(x, y, ndim, S, K, N, stream);
	}
	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 4, grid, block, N);
	sigmoid_f32x4_kernel << <grid, block, 0, stream >> > (x, y, (int)N);
}

void sigmoid_f16(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream) 
{
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 1, grid, block, N);
	sigmoid_f16_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}

void sigmoid_f16x2(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream) 
{
	if ((N % 2) != 0 || (ndim == 2 && (K % 2) != 0)) {
		sigmoid_f16(x, y, ndim, S, K, N, stream);
	}
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 2, grid, block, N);
	sigmoid_f16x2_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}

void sigmoid_f16x8(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream) 
{
	if ((N % 8) != 0 || (ndim == 2 && (K % 8) != 0)) {
		sigmoid_f16(x, y, ndim, S, K, N, stream);
	}
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 8, grid, block, N);
	sigmoid_f16x8_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}

void sigmoid_f16x8_pack(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream) 
{
	if ((N % 8) != 0 || (ndim == 2 && (K % 8) != 0)) {
		sigmoid_f16(x, y, ndim, S, K, N, stream);
	}
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 8, grid, block, N);
	sigmoid_f16x8_pack_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}