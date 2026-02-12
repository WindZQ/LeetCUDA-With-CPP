#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "embedding.cuh"

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void embedding_f32_kernel(const int* idx, float* weight, float* output, int emb_size)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int tid = tx + bx * blockDim.x;
	int offset = idx[bx] * emb_size;

	output[tx + bx * emb_size] = weight[tx + offset];
}

__global__ void embedding_f32x4_kernel(const int* idx, float* weight, float* output, int emb_size)
{
	int tx = threadIdx.x * 4;
	int bx = blockIdx.x;
	int offset = idx[bx] * emb_size;

	output[tx + bx * emb_size] = weight[tx + offset];
	output[tx + bx * emb_size + 1] = weight[tx + offset + 1];
	output[tx + bx * emb_size + 2] = weight[tx + offset + 2];
	output[tx + bx * emb_size + 3] = weight[tx + offset + 3];
}

__global__ void embedding_f32x4_pack_kernel(const int* idx, float* weight, float* output, int emb_size)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int tid = tx + bx * blockDim.x;
	int offset = idx[bx] * emb_size;
	LDST128BITS(output[tx * 4 + bx * emb_size]) = LDST128BITS(weight[tx * 4 + offset]);
}

__global__ void embedding_f16_kernel(const int* idx, half* weight, half* output, int emb_size)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int tid = tx + bx * blockDim.x;
	int offset = idx[bx] * emb_size;

	output[tx + bx * emb_size] = weight[tx + offset];
}

__global__ void embedding_f16x8_kernel(const int* idx, half* weight, half* output, int emb_size)
{
	int tx = threadIdx.x * 8;
	int bx = blockIdx.x;
	int tid = tx + bx * blockDim.x;
	int offset = idx[bx] * emb_size;

	output[tx + bx * emb_size] = weight[tx + offset];
	output[tx + bx * emb_size + 1] = weight[tx + offset + 1];
	output[tx + bx * emb_size + 2] = weight[tx + offset + 2];
	output[tx + bx * emb_size + 3] = weight[tx + offset + 3];
	output[tx + bx * emb_size + 4] = weight[tx + offset + 4];
	output[tx + bx * emb_size + 5] = weight[tx + offset + 5];
	output[tx + bx * emb_size + 6] = weight[tx + offset + 6];
	output[tx + bx * emb_size + 7] = weight[tx + offset + 7];
}

__global__ void embedding_f16x8_pack_kernel(const int* idx, half* weight, half* output, int emb_size)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int tid = tx + bx * blockDim.x;
	int offset = idx[bx] * emb_size;

	LDST128BITS(output[tx * 8 + bx * emb_size]) = LDST128BITS(weight[tx * 8 + offset]);
}

void embedding_f32(const int* idx, float* weight, float* out,
	int N, int K, cudaStream_t stream) 
{
	dim3 grid(N);
	dim3 block(K);
	embedding_f32_kernel << <grid, block, 0, stream >> > (idx, weight, out, K);
}

void embedding_f32x4(const int* idx, float* weight, float* out,
	int N, int K, cudaStream_t stream) 
{
	dim3 grid(N);
	dim3 block(K / 4);
	embedding_f32x4_kernel << <grid, block, 0, stream >> > (idx, weight, out, K);
}

void embedding_f32x4_pack(const int* idx, float* weight, float* out,
	int N, int K, cudaStream_t stream) 
{
	dim3 grid(N);
	dim3 block(K / 4);
	embedding_f32x4_pack_kernel << <grid, block, 0, stream >> > (idx, weight, out, K);
}

void embedding_f16(const int* idx, half* weight, half* out,
	int N, int K, cudaStream_t stream) 
{
	dim3 grid(N);
	dim3 block(K);
	embedding_f16_kernel << <grid, block, 0, stream >> > (idx, weight, out, K);
}

void embedding_f16x8(const int* idx, half* weight, half* out,
	int N, int K, cudaStream_t stream) 
{
	dim3 grid(N);
	dim3 block(K / 8);
	embedding_f16x8_kernel << <grid, block, 0, stream >> > (idx, weight, out, K);
}

void embedding_f16x8_pack(const int* idx, half* weight, half* out,
	int N, int K, cudaStream_t stream) 
{
	dim3 grid(N);
	dim3 block(K / 8);
	embedding_f16x8_pack_kernel << <grid, block, 0, stream >> > (idx, weight, out, K);
}