#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "rope.cuh"

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define BLOCK_SIZE 256
#define theta 10000.0f

__global__ void rope_f32_kernel(float* x, float* out, int seq_len, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float x1 = x[idx * 2];
	float x2 = x[idx * 2 + 1];
	int token_pos = idx / n;
	int token_idx = idx % n;
	float exp_v = 1.0f / powf(theta, 2 * token_idx / (n * 2.0f));
	float sin_v = sinf(token_pos * exp_v);
	float cos_v = cosf(token_pos * exp_v);
	float out1 = x1 * cos_v - x2 * sin_v;
	float out2 = x1 * sin_v + x2 * cos_v;
	out[idx * 2] = out1;
	out[idx * 2 + 1] = out2;
}

// another index method of rope
__global__ void rope_f32_v2_kernel(float* x, float* out, int seq_len, int n) 
{
	int token_pos = blockIdx.x;
	int tid = threadIdx.x;
	float x1 = x[token_pos * n * 2 + tid * 2];
	float x2 = x[token_pos * n * 2 + tid * 2 + 1];
	float exp_v = 1.0f / powf(theta, 2 * tid / (n * 2.0f));
	float sin_v = sinf(token_pos * exp_v);
	float cos_v = cosf(token_pos * exp_v);
	float out1 = x1 * cos_v - x2 * sin_v;
	float out2 = x1 * sin_v + x2 * cos_v;
	out[token_pos * n * 2 + tid * 2] = out1;
	out[token_pos * n * 2 + tid * 2 + 1] = out2;
}

__global__ void rope_f32x4_pack_kernel(float* x, float* out, int seq_len, int n) 
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float4 x_v = FLOAT4(x[idx * 4]);
	int token_pos = idx / n;
	int token_idx = idx % n;
	float exp_f_v = 1.0f / powf(theta, 2 * token_idx * 2 / (n * 4.0f));
	float exp_s_v = 1.0f / powf(theta, 2 * (token_idx * 2 + 1) / (n * 4.0f));
	float sin_f_v = sinf(token_pos * exp_f_v);
	float cos_f_v = cosf(token_pos * exp_f_v);
	float sin_s_v = sinf(token_pos * exp_s_v);
	float cos_s_v = cosf(token_pos * exp_s_v);
	float4 out_v;
	out_v.x = x_v.x * cos_f_v - x_v.y * sin_f_v;
	out_v.y = x_v.x * sin_f_v + x_v.y * cos_f_v;
	out_v.z = x_v.z * cos_s_v - x_v.w * sin_s_v;
	out_v.w = x_v.z * sin_s_v + x_v.w * cos_s_v;
	FLOAT4(out[idx * 4]) = out_v;
}

void rope_f32(float* x, float* out, int seq_len, int hidden_size, cudaStream_t stream) 
{
	int N = hidden_size / 2;
	dim3 grid((seq_len * N + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 block(BLOCK_SIZE);
	rope_f32_kernel << <grid, block, 0, stream >> > (x, out, seq_len, N);
}

void rope_f32_v2(float* x, float* out, int seq_len, int hidden_size, cudaStream_t stream) 
{
	int N = hidden_size / 2;
	dim3 grid(seq_len);
	dim3 block(N);
	rope_f32_v2_kernel << <grid, block, 0, stream >> > (x, out, seq_len, N);
}

void rope_f32x4_pack(float* x, float* out, int seq_len, int hidden_size, cudaStream_t stream) 
{
	int N = hidden_size / 4;
	dim3 grid((seq_len * N + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 block(BLOCK_SIZE);
	rope_f32x4_pack_kernel << <grid, block, 0, stream >> > (x, out, seq_len, N);
}