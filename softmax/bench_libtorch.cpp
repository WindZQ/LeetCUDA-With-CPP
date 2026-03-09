#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "softmax.cuh"

using Tensor = torch::Tensor;

static void cuda_check(cudaError_t err, const char* file, int line) 
{
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at " << file << ":" << line
			<< ", code=" << err
			<< ", msg=" << cudaGetErrorString(err) << std::endl;
		std::exit(EXIT_FAILURE);
	}
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

template <typename Func>
std::pair<Tensor, double> run_benchmark(
	Func perf_func,
	const Tensor& x,
	const std::string& tag,
	Tensor& out,
	int warmup = 10,
	int iters = 100,
	bool show_all = false)
{
	out.fill_(0);

	for (int i = 0; i < warmup; ++i) 
	{
		perf_func(x, out);
	}

	CUDA_CHECK(cudaDeviceSynchronize());

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < iters; ++i) 
	{
		perf_func(x, out);
	}

	CUDA_CHECK(cudaDeviceSynchronize());

	auto end = std::chrono::high_resolution_clock::now();
	double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
	double mean_ms = total_ms / iters;

	auto flat = out.flatten().to(torch::kCPU);
	int n = std::min<int64_t>(3, flat.numel());

	std::vector<float> vals;
	if (flat.scalar_type() == torch::kFloat16) {
		auto flat_fp32 = flat.to(torch::kFloat32);
		auto acc = flat_fp32.accessor<float, 1>();
		for (int i = 0; i < n; ++i) vals.push_back(acc[i]);
	} else {
		auto acc = flat.accessor<float, 1>();
		for (int i = 0; i < n; ++i) vals.push_back(acc[i]);
	}

	std::cout << std::setw(24) << ("out_" + tag) << ": [";
	for (int i = 0; i < n; ++i) 
	{
		std::cout << std::fixed << std::setprecision(8) << vals[i];
		if (i != n - 1) std::cout << ", ";
	}
	std::cout << "], time:" << std::fixed << std::setprecision(8)
		<< mean_ms << "ms" << std::endl;

	if (show_all) {
		std::cout << out << std::endl;
	}

	return { out, mean_ms };
}

void benchmark_shape(int S, int H) 
{
	std::cout << std::string(100, '-') << std::endl;
	std::cout << std::setw(55) << ("S=" + std::to_string(S) + ", H=" + std::to_string(H)) << std::endl;
	std::cout << std::string(100, '-') << std::endl;

	auto opts_f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
	auto opts_f16 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16);

	Tensor x = torch::randn({ S, H }, opts_f32).contiguous();
	Tensor out = torch::zeros_like(x).contiguous();

	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	auto bench_softmax_f32_per_token = [&](const Tensor& in, Tensor& o) {
		softmax_f32_per_token(
			in.data_ptr<float>(),
			o.data_ptr<float>(),
			S, H, stream
		);
	};

	auto bench_softmax_f32x4_per_token = [&](const Tensor& in, Tensor& o) {
		softmax_f32x4_per_token(
			in.data_ptr<float>(),
			o.data_ptr<float>(),
			S, H, stream
		);
	};

	auto bench_safe_softmax_f32_per_token = [&](const Tensor& in, Tensor& o) {
		safe_softmax_f32_per_token(
			in.data_ptr<float>(),
			o.data_ptr<float>(),
			S, H, stream
		);
	};

	auto bench_safe_softmax_f32x4_pre_token = [&](const Tensor& in, Tensor& o) {
		safe_softmax_f32x4_pre_token(
			in.data_ptr<float>(),
			o.data_ptr<float>(),
			S, H, stream
		);
	};

	auto bench_online_safe_softmax_f32_per_token = [&](const Tensor& in, Tensor& o) {
		online_safe_softmax_f32_per_token(
			in.data_ptr<float>(),
			o.data_ptr<float>(),
			S, H, stream
		);
	};

	auto bench_online_safe_softmax_f32x4_pack_per_token = [&](const Tensor& in, Tensor& o) {
		online_safe_softmax_f32x4_pack_per_token(
			in.data_ptr<float>(),
			o.data_ptr<float>(),
			S, H, stream
		);
	};

	auto bench_torch_softmax_f32 = [&](const Tensor& in, Tensor& o) {
		torch::softmax_out(o, in, 1);
	};

	Tensor x_f16 = x.to(torch::kFloat16).contiguous();
	Tensor out_f16 = torch::zeros_like(x_f16).contiguous();

	auto bench_safe_softmax_f16_f32_per_token = [&](const Tensor& in, Tensor& o) {
		safe_softmax_f16_f32_per_token(
			reinterpret_cast<half*>(in.data_ptr<at::Half>()),
			reinterpret_cast<half*>(o.data_ptr<at::Half>()),
			S, H, stream
		);
	};

	auto bench_safe_softmax_f16x2_f32_per_token = [&](const Tensor& in, Tensor& o) {
		safe_softmax_f16x2_f32_per_token(
			reinterpret_cast<half*>(in.data_ptr<at::Half>()),
			reinterpret_cast<half*>(o.data_ptr<at::Half>()),
			S, H, stream
		);
	};

	auto bench_safe_softmax_f16x8_pack_f32_per_token = [&](const Tensor& in, Tensor& o) {
		safe_softmax_f16x8_pack_f32_per_token(
			reinterpret_cast<half*>(in.data_ptr<at::Half>()),
			reinterpret_cast<half*>(o.data_ptr<at::Half>()),
			S, H, stream
		);
	};

	auto bench_torch_softmax_f16 = [&](const Tensor& in, Tensor& o) {
		torch::softmax_out(o, in, 1);
	};

	if (H == 256 || H == 512 || H == 1024) {
		run_benchmark(bench_softmax_f32_per_token, x, "f32(per)", out);
		run_benchmark(bench_softmax_f32x4_per_token, x, "f32x4(per)", out);
		run_benchmark(bench_safe_softmax_f32_per_token, x, "f32(safe)", out);
		run_benchmark(bench_online_safe_softmax_f32_per_token, x, "f32(safe+online)", out);
		run_benchmark(bench_online_safe_softmax_f32x4_pack_per_token, x, "f32x4(safe+online)", out);
		run_benchmark(bench_safe_softmax_f32x4_pre_token, x, "f32x4(safe)", out);
		run_benchmark(bench_torch_softmax_f32, x, "f32_th(per)", out);

		std::cout << std::string(100, '-') << std::endl;

		run_benchmark(bench_safe_softmax_f16_f32_per_token, x_f16, "f16f32(safe)", out_f16);
		run_benchmark(bench_safe_softmax_f16x2_f32_per_token, x_f16, "f16x2f32(safe)", out_f16);
		run_benchmark(bench_safe_softmax_f16x8_pack_f32_per_token, x_f16, "f16x8packf32(safe)", out_f16);
		run_benchmark(bench_torch_softmax_f16, x_f16, "f16_th(per)", out_f16);
	} else if (H == 2048) {
		run_benchmark(bench_softmax_f32x4_per_token, x, "f32x4(per)", out);
		run_benchmark(bench_safe_softmax_f32x4_pre_token, x, "f32x4(safe)", out);
		run_benchmark(bench_online_safe_softmax_f32x4_pack_per_token, x, "f32x4(safe+online)", out);
		run_benchmark(bench_torch_softmax_f32, x, "f32_th(per)", out);

		std::cout << std::string(100, '-') << std::endl;

		run_benchmark(bench_safe_softmax_f16x2_f32_per_token, x_f16, "f16x2f32(safe)", out_f16);
		run_benchmark(bench_safe_softmax_f16x8_pack_f32_per_token, x_f16, "f16x8packf32(safe)", out_f16);
		run_benchmark(bench_torch_softmax_f16, x_f16, "f16_th(per)", out_f16);
	} else if (H == 4096) {
		run_benchmark(bench_softmax_f32x4_per_token, x, "f32x4(per)", out);
		run_benchmark(bench_safe_softmax_f32x4_pre_token, x, "f32x4(safe)", out);
		run_benchmark(bench_online_safe_softmax_f32x4_pack_per_token, x, "f32x4(safe+online)", out);
		run_benchmark(bench_torch_softmax_f32, x, "f32_th(per)", out);

		std::cout << std::string(100, '-') << std::endl;

		run_benchmark(bench_safe_softmax_f16x8_pack_f32_per_token, x_f16, "f16x8packf32(safe)", out_f16);
		run_benchmark(bench_torch_softmax_f16, x_f16, "f16_th(per)", out_f16);
	} else if (H == 8192) {
		run_benchmark(bench_safe_softmax_f16x8_pack_f32_per_token, x_f16, "f16x8packf32(safe)", out_f16);
		run_benchmark(bench_torch_softmax_f16, x_f16, "f16_th(per)", out_f16);
	}

	std::cout << std::string(100, '-') << std::endl;
}

int main()
{
	if (!torch::cuda::is_available()) {
		std::cerr << "CUDA is not available." << std::endl;
		return -1;
	}

	torch::NoGradGuard no_grad;

	benchmark_shape(4096, 256);
	benchmark_shape(4096, 512);
	benchmark_shape(4096, 1024);
	benchmark_shape(4096, 2048);
	benchmark_shape(4096, 4096);
	benchmark_shape(4096, 8192);
	benchmark_shape(8192, 8192);

	return 0;
}