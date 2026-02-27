#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "mat_transpose.cuh"
#include "mat_transpose_cute.cuh"

namespace ix = torch::indexing;

#define CHECK(call)                                                                \
{                                                                                  \
	const cudaError_t error = call;                                                \
	if(error != cudaSuccess)                                                       \
	{                                                                              \
		printf("ERROR: %s:%d, ",__FILE__, __LINE__);                               \
		printf("code:%d, reason:%s\n",error, cudaGetErrorString(error));           \
		exit(1);                                                                   \
	}                                                                              \
}

double benchmark_ms(std::function<void()> fn, cudaStream_t stream, int warmup = 10, int iters = 1000)
{
	for (int i = 0; i < warmup; ++i) fn();
	CHECK(cudaStreamSynchronize(stream));

	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));

	CHECK(cudaEventRecord(start, stream));
	for (int i = 0; i < iters; ++i) fn();
	CHECK(cudaEventRecord(stop, stream));
	CHECK(cudaEventSynchronize(stop));

	float ms = 0.0f;
	CHECK(cudaEventElapsedTime(&ms, start, stop));

	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(stop));

	return ms / static_cast<float>(iters);
}

void print_result(const std::string& tag, const torch::Tensor& x, const torch::Tensor& out, double mean_ms) 
{
	bool ok = out.transpose(0, 1).equal(x);

	auto v = out.index({ ix::Slice(0, 2), ix::Slice(0, 2) }).reshape({ -1 }).to(torch::kCPU);
	std::vector<float> vals;
	for (int i = 0; i < std::min<int64_t>(3, v.numel()); ++i) 
	{
		vals.push_back(v[i].item<float>());
	}

	std::cout << std::setw(35) << ("out_" + tag) << ": [";
	for (size_t i = 0; i < vals.size(); ++i) 
	{
		std::cout << std::fixed << std::setprecision(8) << vals[i];
		if (i + 1 < vals.size()) std::cout << ", ";
	}

	std::cout << "], validate " << std::left << std::setw(5) << (ok ? "True" : "False")
		<< ", time:" << std::fixed << std::setprecision(8) << mean_ms << "ms\n";
}

void check_tensor_f32_cuda_contig(const torch::Tensor& t, const char* name) 
{
	TORCH_CHECK(t.is_cuda(), name, " must be CUDA tensor");
	TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
	TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
	TORCH_CHECK(t.dim() == 2, name, " must be 2D");
}

int main() 
{
	if (!torch::cuda::is_available()) {
		std::cerr << "CUDA is not available. Please use LibTorch with CUDA.\n";
		return 1;
	}

	c10::cuda::CUDAGuard device_guard(0);
	torch::NoGradGuard no_grad;

	cudaStream_t stream = at::cuda::getDefaultCUDAStream();

	std::vector<int64_t> Ms = { 1024, 2048, 4096, 8192 };
	std::vector<int64_t> Ns = { 1024, 2048, 4096, 8192 };

	for (auto M : Ms) 
	{
		for (auto N : Ns) 
		{
			std::cout << std::string(130, '-') << "\n";
			std::cout << std::string(55, ' ') << "M = " << M << ", N = " << N << "\n";

			auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
			auto x = torch::arange(0, M * N, opts).reshape({ M, N }).contiguous();
			auto y = torch::randn({ N, M }, opts).contiguous();

			check_tensor_f32_cuda_contig(x, "x");
			check_tensor_f32_cuda_contig(y, "y");

			float* px = x.data_ptr<float>();
			float* py = y.data_ptr<float>();

			{
				torch::Tensor out;
				double ms = benchmark_ms([&]() { out = x.clone(); }, stream);
				print_result("original", x, out, ms);
			}

			auto run_out = [&](const std::string& tag, auto&& body) 
			{
				y.zero_();
				double ms = benchmark_ms([&]() { body(); }, stream);
				print_result(tag, x, y, ms);
			};

			run_out("f32_col2row", [&]() 
			{
				mat_transpose_f32_col2row(px, py, (int)M, (int)N, stream);
			});

			run_out("f32_row2col", [&]() 
			{
				mat_transpose_f32_row2col(px, py, (int)M, (int)N, stream);
			});

			run_out("f32_col2row(2d)", [&]() 
			{
				mat_transpose_f32_col2row2d(px, py, (int)M, (int)N, stream);
			});

			run_out("f32_row2col(2d)", [&]() 
			{
				mat_transpose_f32_row2col2d(px, py, (int)M, (int)N, stream);
			});

			if (M == N) {
				run_out("f32_diagnonal", [&]()
				{
					mat_transpose_f32_diagonal2d(px, py, (int)M, (int)N, stream);
				});
			}

			run_out("f32x4_col2row", [&]() 
			{
				mat_transpose_f32x4_col2row(px, py, (int)M, (int)N, stream);
			});

			run_out("f32x4_row2col", [&]() 
			{
				mat_transpose_f32x4_row2col(px, py, (int)M, (int)N, stream);
			});

			run_out("f32x4_col2row(2d)", [&]() 
			{
				mat_transpose_f32x4_col2row2d(px, py, (int)M, (int)N, stream);
			});

			run_out("f32x4_row2col(2d)", [&]() 
			{
				mat_transpose_f32x4_row2col2d(px, py, (int)M, (int)N, stream);
			});

			run_out("f32x4_shared_col2row(2d)", [&]() 
			{
				mat_transpose_f32x4_shared_col2row2d(px, py, (int)M, (int)N, stream);
			});

			run_out("f32x4_shared_row2col(2d)", [&]() 
			{
				mat_transpose_f32x4_shared_row2col2d(px, py, (int)M, (int)N, stream);
			});

			run_out("f32x4_shared_bcf_col2row(2d)", [&]()
			{
				mat_transpose_f32x4_shared_bcf_col2row2d(px, py, (int)M, (int)N, stream);
			});

			run_out("f32x4_shared_bcf_row2col(2d)", [&]() 
			{
				mat_transpose_f32x4_shared_bcf_row2col2d(px, py, (int)M, (int)N, stream);
			});

			run_out("f32x4_shared_bcf_merge_write_row2col(2d)", [&]() 
			{
				mat_transpose_f32x4_shared_bcf_merge_write_row2col2d(px, py, (int)M, (int)N, stream);
			});

			run_out("mat_transpose_cute_col2row_reg", [&]() 
			{
				mat_transpose_cute_col2row_reg(px, py, (int)M, (int)N, stream);
			});

			run_out("mat_transpose_cute_row2col_reg", [&]()
			{
				mat_transpose_cute_row2col_reg(px, py, (int)M, (int)N, stream);
			});

			run_out("mat_transpose_cute_col_smem", [&]()
			{
				mat_transpose_cute_col_smem(px, py, (int)M, (int)N, stream);
			});

			run_out("mat_transpose_cute_row_smem", [&]()
			{
				mat_transpose_cute_row_smem(px, py, (int)M, (int)N, stream);
			});

			run_out("mat_transpose_cute_col_smem_swizzled", [&]()
			{
				mat_transpose_cute_col_smem_swizzled(px, py, (int)M, (int)N, stream);
			});

			run_out("mat_transpose_cute_row_smem_swizzled", [&]()
			{
				mat_transpose_cute_row_smem_swizzled(px, py, (int)M, (int)N, stream);
			});

			run_out("mat_transpose_cute_row_cvectorized", [&]()
			{
				mat_transpose_cute_row_cvectorized(px, py, (int)M, (int)N, stream);
			});

			run_out("mat_transpose_cute_row_rvectorized", [&]()
			{
				mat_transpose_cute_row_rvectorized(px, py, (int)M, (int)N, stream);
			});

			run_out("mat_transpose_cute_row_cvectorized_swizzled", [&]()
			{
				mat_transpose_cute_row_cvectorized_swizzled(px, py, (int)M, (int)N, stream);
			});

			run_out("mat_transpose_cute_row_rvectorized_swizzled", [&]()
			{
				mat_transpose_cute_row_rvectorized_swizzled(px, py, (int)M, (int)N, stream);
			});

			run_out("mat_transpose_cute_row_rvectorized_swizzled_optimized", [&]()
			{
				mat_transpose_cute_row_rvectorized_swizzled_optimized(px, py, (int)M, (int)N, stream);
			});

			run_out("f32_th", [&]() 
			{
				y.copy_(x.transpose(0, 1));
			});

			std::cout << std::string(130, '-') << "\n";
		}
	}

	return 0;
}