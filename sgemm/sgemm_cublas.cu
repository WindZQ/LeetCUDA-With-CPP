#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "cublas_v2.h"
#include "sgemm_cublas.cuh"

void cublas_sgemm(float* a, float* b, float* c, int M, int N, int K)
{
	cublasHandle_t handle = nullptr;
	cublasCreate(&handle);
	cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

	static float alpha = 1.0;
	static float beta = 0.0;

	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, CUDA_R_32F,
		N, a, CUDA_R_32F, K, &beta, c, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
		CUBLAS_GEMM_DEFAULT);
}

void cublas_sgemm_tf32(float* a, float* b, float* c, int M, int N, int K) 
{
	cublasHandle_t handle = nullptr;
	cublasCreate(&handle);
	cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

	static float alpha = 1.0;
	static float beta = 0.0;

	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, CUDA_R_32F,
		N, a, CUDA_R_32F, K, &beta, c, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
		CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void sgemm_cublas(float* a, float* b, float* c, int M, int N, int K)
{
	cublas_sgemm(a, b, c, M, N, K);
}

// cublas tensor op
void sgemm_cublas_tf32(float* a, float* b, float* c, int M, int N, int K)
{
	cublas_sgemm_tf32(a, b, c, M, N, K);
}