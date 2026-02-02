#include "elementwise.cuh"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32_kernel(float* a, float* b, float* c, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		c[idx] = a[idx] + b[idx];
	}
}

// ElementWise Add + Vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_fp32x4_kernel(float* a, float* b, float* c, int n)
{
	int idx = 4 * (threadIdx.x + blockIdx.x * blockDim.x);

	if (idx < n) {
		float4 reg_a = FLOAT4(a[idx]);
		float4 reg_b = FLOAT4(b[idx]);
		float4 reg_c;
		reg_c.x = reg_a.x + reg_b.x;
		reg_c.y = reg_a.y + reg_b.y;
		reg_c.z = reg_a.z + reg_b.z;
		reg_c.w = reg_a.w + reg_b.w;
		FLOAT4(c[idx]) = reg_c;
	}
}

// FP16
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16_kernel(half* a, half* b, half* c, int n) 
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x ;
	if (idx < n) {
		c[idx] = __hadd(a[idx], b[idx]);
	}
}

// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16x2_kernel(half* a, half* b, half* c, int n) 
{
	int idx = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
	if (idx < n) {
		half2 reg_a = HALF2(a[idx]);
		half2 reg_b = HALF2(b[idx]);
		half2 reg_c;
		reg_c.x = __hadd(reg_a.x, reg_b.x);
		reg_c.y = __hadd(reg_a.y, reg_b.y);
		HALF2(c[idx]) = reg_c;
	}
}

__global__ void elementwise_add_fp16x8_kernel(half* a, half* b, half* c, int n)
{
	int idx = 8 * (threadIdx.x + blockIdx.x * blockDim.x);
	half2 reg_a_0 = HALF2(a[idx + 0]);
	half2 reg_a_1 = HALF2(a[idx + 2]);
	half2 reg_a_2 = HALF2(a[idx + 4]);
	half2 reg_a_3 = HALF2(a[idx + 6]);
	half2 reg_b_0 = HALF2(b[idx + 0]);
	half2 reg_b_1 = HALF2(b[idx + 2]);
	half2 reg_b_2 = HALF2(b[idx + 4]);
	half2 reg_b_3 = HALF2(b[idx + 6]);
	half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;

	reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
	reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
	reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
	reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
	reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
	reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
	reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
	reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);

	if ((idx + 0) < n) {
		HALF2(c[idx + 0]) = reg_c_0;
	}
	if ((idx + 2) < n) {
		HALF2(c[idx + 2]) = reg_c_1;
	}
	if ((idx + 4) < n) {
		HALF2(c[idx + 4]) = reg_c_2;
	}
	if ((idx + 6) < n) {
		HALF2(c[idx + 6]) = reg_c_3;
	}
}

__global__ void elementwise_add_f16x8_pack_kernel(half* a, half* b, half* c, int n)
{
	int idx = 8 * (threadIdx.x + blockIdx.x * blockDim.x);

	// temporary register(memory), .local space in ptx, addressable
	half pack_a[8], pack_b[8], pack_c[8]; // 8x16 bits=128 bits.
	// reinterpret as float4 and load 128 bits in 1 memory issue.
	LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
	LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);


#pragma unroll
	for (int i = 0; i < 8; ++i)
	{
		// __hadd2 for half2 x 4
		HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
	}

	// reinterpret as float4 and store 128 bits in 1 memory issue.
	if ((idx + 7) < n) {
		LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
	} else {
		for (int i = 0; idx + i < n; ++i)
		{
			c[idx + i] = __hadd(a[idx + i], b[idx + i]);
		}
	}
}

static inline void launch_1d(int vec, int64_t N, dim3& grid, dim3& block) 
{
	block = dim3(256 / vec);
	grid = dim3((unsigned)((N + 256 - 1) / 256));
}

static inline bool launch_2d_possible(int vec, int64_t S, int64_t K) 
{
	return (S > 0 && K > 0 && (K / vec) > 0 && (K / vec) <= 1024);
}

void elementwise_add_f32(float* a, float* b, float* c, int64_t N, int64_t S, int64_t K, cudaStream_t stream)
{
	dim3 block(256), grid((unsigned)((N + 255) / 256));
	elementwise_add_f32_kernel << <grid, block, 0, stream >> > (a, b, c, N);
}

void elementwise_add_f32x4(float* a, float* b, float* c, int64_t N, int64_t S, int64_t K, cudaStream_t stream)
{
	dim3 grid, block;
	if (launch_2d_possible(4, S, K)) { 
		block = dim3((unsigned)(K / 4)); 
		grid = dim3((unsigned)S); 
	} else { 
		launch_1d(4, N, grid, block); 
	}

	elementwise_add_fp32x4_kernel << <grid, block, 0, stream >> > (a, b, c, N);
}

void elementwise_add_f16(__half * a, __half * b, __half * c, int64_t N, int64_t S, int64_t K, cudaStream_t stream)
{
	dim3 block(256), grid((unsigned)((N + 255) / 256));
	elementwise_add_f16_kernel << <grid, block, 0, stream >> > (a, b, c, N);
}

void elementwise_add_f16x2(__half * a, __half * b, __half * c, int64_t N, int64_t S, int64_t K, cudaStream_t stream)
{
	dim3 grid, block;
	if (launch_2d_possible(2, S, K)) { 
		block = dim3((unsigned)(K / 2)); 
		grid = dim3((unsigned)S); 
	} else { 
		launch_1d(2, N, grid, block); 
	}

	elementwise_add_f16x2_kernel << <grid, block, 0, stream >> > (a, b, c, N);
}

void elementwise_add_f16x8(__half * a, __half * b, __half * c, int64_t N, int64_t S, int64_t K, cudaStream_t stream)
{
	dim3 grid, block;
	if (launch_2d_possible(8, S, K)) { 
		block = dim3((unsigned)(K / 8)); 
		grid = dim3((unsigned)S); 
	} else { 
		launch_1d(8, N, grid, block); 
	}
	elementwise_add_fp16x8_kernel << <grid, block, 0, stream >> > (a, b, c, N);
}

void elementwise_add_f16x8_pack(__half * a, __half * b, __half * c, int64_t N, int64_t S, int64_t K, cudaStream_t stream)
{
	dim3 grid, block;
	if (launch_2d_possible(8, S, K)) { 
		block = dim3((unsigned)(K / 8)); 
		grid = dim3((unsigned)S); 
	} else { 
		launch_1d(8, N, grid, block); 
	}
	elementwise_add_f16x8_pack_kernel << <grid, block, 0, stream >> > (a, b, c, N);
}