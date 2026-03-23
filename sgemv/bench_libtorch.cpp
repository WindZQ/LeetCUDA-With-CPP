#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "sgemv.cuh"


using PerfFunc = std::function<void(const torch::Tensor&, const torch::Tensor&, torch::Tensor&)>;

std::pair<torch::Tensor, double> run_benchmark(
	const PerfFunc& perf_func,
	const torch::Tensor& a,
	const torch::Tensor& b,
	const std::string& tag,
	torch::Tensor& out,
	int warmup = 10,
	int iters = 200,
	bool show_all = false)
{
	out.fill_(0);

	for (int i = 0; i < warmup; ++i) 
	{
		perf_func(a, b, out);
	}
	
	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < iters; ++i) 
	{
		perf_func(a, b, out);
	}

	cudaDeviceSynchronize();

	auto end = std::chrono::high_resolution_clock::now();
	double total_time_ms =
		std::chrono::duration<double, std::milli>(end - start).count();
	double mean_time_ms = total_time_ms / iters;

	auto out_cpu = out.flatten().to(torch::kCPU);
	std::vector<float> vals;
	int n = std::min<int>(3, out_cpu.numel());
	for (int i = 0; i < n; ++i) {
		vals.push_back(out_cpu[i].item<float>());
	}

	std::cout << std::setw(13) << ("out_" + tag) << ": [";
	for (int i = 0; i < vals.size(); ++i) 
	{
		std::cout << std::fixed << std::setprecision(8) << vals[i];
		if (i + 1 < vals.size()) std::cout << ", ";
	}
	std::cout << "], time:" << std::fixed << std::setprecision(8)
		<< mean_time_ms << "ms" << std::endl;

	if (show_all) {
		std::cout << out_cpu << std::endl;
	}

	return { out.clone(), mean_time_ms };
}

int main() 
{
	torch::NoGradGuard no_grad;

	if (!torch::cuda::is_available()) {
		std::cerr << "CUDA is not available." << std::endl;
		return -1;
	}

	auto options = torch::TensorOptions()
		.dtype(torch::kFloat32)
		.device(torch::kCUDA);

	std::cout << std::string(80, '-') << std::endl;

	{
		int M = 1024, N = 1, K = 128;
		auto a = torch::randn({ M, K }, options).contiguous();
		auto b = torch::randn({ K, N }, options).contiguous();
		auto c = torch::randn({ M, N }, options).contiguous();

		auto func_k32 = [](const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
			sgemv_k32_f32(
				a.data_ptr<float>(),
				b.data_ptr<float>(),
				c.data_ptr<float>(),
				a.size(0),
				a.size(1));
		};

		auto func_k128x4 = [](const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
			sgemv_k128_f32x4(
				a.data_ptr<float>(),
				b.data_ptr<float>(),
				c.data_ptr<float>(),
				a.size(0),
				a.size(1));
		};

		auto func_torch = [](const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
			torch::matmul_out(c, a, b);
		};

		run_benchmark(func_k32, a, b, "k32f32", c);
		run_benchmark(func_k128x4, a, b, "k128f32x4", c);
		run_benchmark(func_torch, a, b, "f32_th", c);
	}

	std::cout << std::string(80, '-') << std::endl;

	{
		int M = 1024, N = 1, K = 16;
		auto a = torch::randn({ M, K }, options).contiguous();
		auto b = torch::randn({ K, N }, options).contiguous();
		auto c = torch::randn({ M, N }, options).contiguous();

		auto func_k16 = [](const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
			sgemv_k16_f32(
				a.data_ptr<float>(),
				b.data_ptr<float>(),
				c.data_ptr<float>(),
				a.size(0),
				a.size(1));
		};

		auto func_torch = [](const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
			torch::matmul_out(c, a, b);
		};

		run_benchmark(func_k16, a, b, "k16f32", c);
		run_benchmark(func_torch, a, b, "f32_th", c);
	}

	std::cout << std::string(80, '-') << std::endl;

	return 0;
}