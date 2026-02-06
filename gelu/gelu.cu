#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "gelu.cuh"

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
#define M_SQRT2    1.41421356237309504880   
#define M_2_SQRTPI 1.12837916709551257390   
#define M_SQRT1_2  0.707106781186547524401
#define SQRT_2_PI M_SQRT2 * M_2_SQRTPI * 0.5f
#define HALF_1 __float2half(1.0f)
#define HALF_2 __float2half(2.0f)
#define HALF_DIV2 __float2half(0.5f)
// to clear the error among self defined gelu and pytorch gelu. Calculate
// $\sqrt{\frac{\pi}{2}}$ by $\sqrt{2 * \pi} / 2$
#define HALF_SQRT_2_PI                                                         \
  __float2half(M_SQRT2) * __float2half(M_2_SQRTPI) * HALF_DIV2
#define HALF_V_APP __float2half(0.044715f)

#define HALF_GELU_OPS gelu_tanh_approximate
#define GELU_OPS gelu_tanh_approximate

__device__ __forceinline__ half hexp_compat(half a)
{
	return __float2half(__expf(__half2float(a)));
}

__inline__ __device__ half gelu_tanh_approximate(half x)
{
	half x_cube = x * x * x;
	// compute mid value : inner = 0.7978845608 * (x + 0.044715 * x * x * x)
	half inner = HALF_SQRT_2_PI * (x + HALF_V_APP * x_cube);
	// compute tanh
	return HALF_DIV2 * x *
		(HALF_1 +
			((hexp_compat(inner * HALF_2) - HALF_1) / (hexp_compat(inner * HALF_2) + HALF_1)));
}

__inline__ __device__ float gelu_tanh_approximate(float x)
{
	return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715f * x * x * x)));
}

__inline__ __device__ float gelu_none_approximate(float x)
{
	return x * 0.5 * (1 + erff(x * M_SQRT1_2));
}

// FP32
// GELU tanh approximate: x, y:x 0.5 * x
// * (1.0 + tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x))) grid(N/256),
// block(K=256)
__global__ void gelu_f32_kernel(float* x, float* y, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < n) {
		float v = fminf(fmaxf(x[idx], MIN_EXP_F32), MAX_EXP_F32);
		y[idx] = GELU_OPS(v);
	}
}

// GELU tanh approximate; Vec4
// grid(N/256), block(256/4)
__global__ void gelu_f32x4_kernel(float* x, float* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
	float4 reg_x = FLOAT4(x[idx]);
	float4 reg_y;

	reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
	reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
	reg_x.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
	reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);

	reg_y.x = GELU_OPS(reg_x.x);
	reg_y.y = GELU_OPS(reg_x.y);
	reg_y.z = GELU_OPS(reg_x.z);
	reg_y.w = GELU_OPS(reg_x.w);

	if ((idx + 0) < n) {
		FLOAT4(y[idx]) = reg_y;
	}
}

// FP16
// GELU approximate: x, y:x 0.5 * x *
// (1.0 + tanh(0.7978845608 (x + 0.044715 * x * x * x))) Vec4
__global__ void gelu_f16_kernel(half* x, half* y, int n) 
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < n) {
		half v = x[idx];
		v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);
		y[idx] = HALF_GELU_OPS(v);
	}
}

__global__ void gelu_f16x2_kernel(half* x, half* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
	half2 reg_x = HALF2(x[idx]);
	half2 reg_y;

	reg_x.x = __hmin(__hmax(reg_x.x, MIN_EXP_F16), MAX_EXP_F16);
	reg_x.y = __hmin(__hmax(reg_x.y, MIN_EXP_F16), MAX_EXP_F16);

	reg_y.x = HALF_GELU_OPS(reg_x.x);
	reg_y.y = HALF_GELU_OPS(reg_x.y);

	if ((idx + 0) < n) {
		HALF2(y[idx]) = reg_y;
	}
}

// unpack f16x8
__global__ void gelu_f16x8_kernel(half* x, half* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 8;

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

	reg_x_0.x = HALF_GELU_OPS(reg_x_0.x);
	reg_x_0.y = HALF_GELU_OPS(reg_x_0.y);
	reg_x_1.x = HALF_GELU_OPS(reg_x_1.x);
	reg_x_1.y = HALF_GELU_OPS(reg_x_1.y);
	reg_x_2.x = HALF_GELU_OPS(reg_x_2.x);
	reg_x_2.y = HALF_GELU_OPS(reg_x_2.y);
	reg_x_3.x = HALF_GELU_OPS(reg_x_3.x);
	reg_x_3.y = HALF_GELU_OPS(reg_x_3.y);

	if ((idx + 0) < n) {
		HALF2(y[idx + 0]) = reg_x_0;
	}
	if ((idx + 2) < n) {
		HALF2(y[idx + 2]) = reg_x_1;
	}
	if ((idx + 4) < n) {
		HALF2(y[idx + 4]) = reg_x_2;
	}
	if ((idx + 6) < n) {
		HALF2(y[idx + 6]) = reg_x_3;
	}
}

// pack f16x8
__global__ void gelu_f16x8_pack_kernel(half* x, half* y, int n)
{
	int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 8;

	// temporary register(memory), .local space in ptx, addressable
	half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
	// reinterpret as float4 and load 128 bits in 1 memory issue.
	LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

#pragma unroll
	for (int i = 0; i < 8; ++i) 
	{
		half v = __hmin(__hmax(pack_x[i], MIN_EXP_F16), MAX_EXP_F16);
		pack_y[i] = HALF_GELU_OPS(v);
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
	}
	else {
		block = dim3((unsigned)(256 / n_elements));
		grid = dim3((unsigned)((N + 256 - 1) / 256));
	}
}

void gelu_f32(float* x, float* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 1, grid, block, N);
	gelu_f32_kernel << <grid, block, 0, stream >> > (x, y, (int)N);
}

void gelu_f32x4(float* x, float* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	if ((N % 4) != 0 || (ndim == 2 && (K % 4) != 0)) {
		gelu_f32(x, y, ndim, S, K, N, stream);
	}
	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 4, grid, block, N);
	gelu_f32x4_kernel << <grid, block, 0, stream >> > (x, y, (int)N);
}

void gelu_f16(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 1, grid, block, N);
	gelu_f16_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}

void gelu_f16x2(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	if ((N % 2) != 0 || (ndim == 2 && (K % 2) != 0)) {
		gelu_f16(x, y, ndim, S, K, N, stream);
	}
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 2, grid, block, N);
	gelu_f16x2_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}

void gelu_f16x8(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	if ((N % 8) != 0 || (ndim == 2 && (K % 8) != 0)) {
		gelu_f16(x, y, ndim, S, K, N, stream);
	}
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 8, grid, block, N);
	gelu_f16x8_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}

void gelu_f16x8_pack(void* x, void* y,
	int64_t ndim, int64_t S, int64_t K, int64_t N,
	cudaStream_t stream)
{
	if ((N % 8) != 0 || (ndim == 2 && (K % 8) != 0)) {
		gelu_f16(x, y, ndim, S, K, N, stream);
	}
	half* xp = reinterpret_cast<half*>(x);
	half* yp = reinterpret_cast<half*>(y);

	dim3 grid, block;
	launch_1d_or_2d(ndim, S, K, 8, grid, block, N);
	gelu_f16x8_pack_kernel << <grid, block, 0, stream >> > (xp, yp, (int)N);
}