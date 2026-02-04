#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include "relu.cuh"

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

template <typename Fn>
double benchmark_ms(Fn&& fn, cudaStream_t stream, int warmup = 10, int iters = 1000)
{
	for (int i = 0; i < warmup; ++i) fn();
	CHECK(cudaStreamSynchronize(stream));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, stream);
	for (int i = 0; i < iters; ++i) fn();
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);

	float total_ms = 0.0f;
	cudaEventElapsedTime(&total_ms, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return (double)total_ms / (double)iters;
}

static std::vector<float> first2_as_float(torch::Tensor t)
{
	auto v = t.view({ -1 }).slice(0, 0, 2).to(torch::kFloat32).cpu();
	return { v[0].item<float>(), v[1].item<float>() };
}

static void print_result(const std::string& tag, torch::Tensor out, double ms)
{
	auto v = first2_as_float(out);
	std::cout << std::setw(18) << tag << ": ["
		<< std::fixed << std::setprecision(8) << v[0] << ", "
		<< std::fixed << std::setprecision(8) << v[1] << "], time:"
		<< std::fixed << std::setprecision(8) << ms << "ms\n";
}

int main()
{
	torch::NoGradGuard no_grad;
	if (!torch::cuda::is_available()) {
		std::cerr << "CUDA is not available in this LibTorch build.\n";
		return 1;
	}

	torch::manual_seed(0);

	std::vector<int> Ss = { 1024, 2048, 4096 };
	std::vector<int> Ks = { 1024, 2048, 4096 };

	for (int S : Ss)
	{
		for (int K : Ks)
		{
			std::cout << std::string(85, '-') << "\n";
			std::cout << std::string(40, ' ') << "S=" << S << ", K=" << K << "\n";

			auto opts_f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
			auto x = torch::randn({ S, K }, opts_f32).contiguous();
			auto y = torch::zeros_like(x).contiguous();

			c10::cuda::CUDAGuard device_guard(x.device());
			cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

			int64_t ndim = x.dim();
			int64_t N = x.numel();

			// f32
			auto ms_f32 = benchmark_ms([&]() {
				relu_f32(x.data_ptr<float>(), y.data_ptr<float>(), ndim, S, K, N, stream);
				}, stream);
			print_result("out_f32", y, ms_f32);

			auto ms_f32x4 = benchmark_ms([&]() {
				relu_f32x4(x.data_ptr<float>(), y.data_ptr<float>(), ndim, S, K, N, stream);
				}, stream);
			print_result("out_f32x4", y, ms_f32x4);

			auto ms_f32_th = benchmark_ms([&]() {
				torch::sigmoid_out(y, x);
				}, stream);
			print_result("out_f32_th", y, ms_f32_th);

			std::cout << std::string(85, '-') << "\n";

			// f16
			auto x16 = x.to(torch::kFloat16).contiguous();
			auto y16 = torch::zeros_like(x16).contiguous();

			auto ms_f16 = benchmark_ms([&]() {
				relu_f16(x16.data_ptr(), y16.data_ptr(), ndim, S, K, N, stream);
			}, stream);
			print_result("out_f16", y16, ms_f16);

			auto ms_f16x2 = benchmark_ms([&]() {
				relu_f16x2(x16.data_ptr(), y16.data_ptr(), ndim, S, K, N, stream);
			}, stream);
			print_result("out_f16x2", y16, ms_f16x2);

			auto ms_f16x8 = benchmark_ms([&]() {
				relu_f16x8(x16.data_ptr(), y16.data_ptr(), ndim, S, K, N, stream);
			}, stream);
			print_result("out_f16x8", y16, ms_f16x8);

			auto ms_f16x8p = benchmark_ms([&]() {
				relu_f16x8_pack(x16.data_ptr(), y16.data_ptr(), ndim, S, K, N, stream);
			}, stream);
			print_result("out_f16x8pack", y16, ms_f16x8p);

			auto ms_f16_th = benchmark_ms([&]() {
				torch::sigmoid_out(y16, x16);
			}, stream);
			print_result("out_f16_th", y16, ms_f16_th);

			std::cout << std::string(85, '-') << "\n";
		}
	}

	return 0;
}