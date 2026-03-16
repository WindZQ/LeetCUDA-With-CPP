#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "rms_norm.cuh"

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

Tensor naive_rms_norm(const Tensor& x, float g)
{
	const float eps = 1e-5f;
	auto xf = (x.scalar_type() == torch::kFloat16) ? x.to(torch::kFloat32) : x;
	auto s_rms = torch::rsqrt(torch::mean(xf * xf, 1, true) + eps);
	auto y = xf * s_rms * g;
	return (x.scalar_type() == torch::kFloat16) ? y.to(torch::kFloat16) : y;
}

template <typename Func>
std::pair<Tensor, double> run_benchmark_out(
	Func perf_func,
	const Tensor& x,
	const std::string& tag,
	Tensor& out,
	int warmup = 10,
	int iters = 1000,
	bool show_all = false)
{
	constexpr float g = 1.0f;

	out.fill_(0);

	for (int i = 0; i < warmup; ++i)
	{
		perf_func(x, out, g);
	}

	CUDA_CHECK(cudaDeviceSynchronize());
	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < iters; ++i)
	{
		perf_func(x, out, g);
	}

	CUDA_CHECK(cudaDeviceSynchronize());
	auto end = std::chrono::high_resolution_clock::now();

	double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
	double mean_ms = total_ms / iters;

	auto flat = out.flatten().to(torch::kCPU);
	int n = std::min<int64_t>(3, flat.numel());

	std::vector<float> vals;
	if (flat.scalar_type() == torch::kFloat16) {
		auto fp32 = flat.to(torch::kFloat32);
		auto acc = fp32.accessor<float, 1>();
		for (int i = 0; i < n; ++i) vals.push_back(acc[i]);
	} else {
		auto acc = flat.accessor<float, 1>();
		for (int i = 0; i < n; ++i) vals.push_back(acc[i]);
	}

	std::cout << std::setw(17) << ("out_" + tag) << ": [";
	for (int i = 0; i < n; ++i)
	{
		std::cout << std::fixed << std::setprecision(8) << vals[i];
		if (i != n - 1) std::cout << ", ";
	}
	std::cout << "], time:" << std::fixed << std::setprecision(8)
		<< mean_ms << "ms" << std::endl;

	if (show_all) std::cout << out << std::endl;
	return { out, mean_ms };
}

template <typename Func>
std::pair<Tensor, double> run_benchmark_ret(
	Func perf_func,
	const Tensor& x,
	const std::string& tag,
	int warmup = 10,
	int iters = 1000,
	bool show_all = false)
{
	constexpr float g = 1.0f;
	Tensor out;

	for (int i = 0; i < warmup; ++i)
	{
		out = perf_func(x, g);
	}

	CUDA_CHECK(cudaDeviceSynchronize());
	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < iters; ++i)
	{
		out = perf_func(x, g);
	}

	CUDA_CHECK(cudaDeviceSynchronize());
	auto end = std::chrono::high_resolution_clock::now();

	double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
	double mean_ms = total_ms / iters;

	auto flat = out.flatten().to(torch::kCPU);
	int n = std::min<int64_t>(3, flat.numel());

	std::vector<float> vals;
	if (flat.scalar_type() == torch::kFloat16) {
		auto fp32 = flat.to(torch::kFloat32);
		auto acc = fp32.accessor<float, 1>();
		for (int i = 0; i < n; ++i) vals.push_back(acc[i]);
	} else {
		auto acc = flat.accessor<float, 1>();
		for (int i = 0; i < n; ++i) vals.push_back(acc[i]);
	}

	std::cout << std::setw(17) << ("out_" + tag) << ": [";
	for (int i = 0; i < n; ++i)
	{
		std::cout << std::fixed << std::setprecision(8) << vals[i];
		if (i != n - 1) std::cout << ", ";
	}
	std::cout << "], time:" << std::fixed << std::setprecision(8)
		<< mean_ms << "ms" << std::endl;

	if (show_all) std::cout << out << std::endl;
	return { out, mean_ms };
}

void benchmark_case(int N, int K)
{
	std::cout << std::string(85, '-') << std::endl;
	std::cout << std::setw(45) << ("N=" + std::to_string(N) + ", K=" + std::to_string(K)) << std::endl;

	auto opts_f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
	auto opts_f16 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16);

	cudaStream_t stream = at::cuda::getDefaultCUDAStream();

	auto bench_naive = [&](const Tensor& in, float g) -> Tensor {
		return naive_rms_norm(in, g);
	};

	if (K <= 4096) {
		Tensor x = torch::randn({ N, K }, opts_f32).contiguous();
		Tensor out = torch::zeros_like(x).contiguous();

		auto bench_rms_norm_f32 = [&](const Tensor& in, Tensor& o, float g) {
			rms_norm_f32(in.data_ptr<float>(), o.data_ptr<float>(), g, N, K, stream);
		};
		auto bench_rms_norm_f32x4 = [&](const Tensor& in, Tensor& o, float g) {
			rms_norm_f32x4(in.data_ptr<float>(), o.data_ptr<float>(), g, N, K, stream);
		};

		if (K == 512 || K == 1024) {
			run_benchmark_out(bench_rms_norm_f32, x, "f32", out);
			run_benchmark_out(bench_rms_norm_f32x4, x, "f32x4", out);
			run_benchmark_ret(bench_naive, x, "f32_th");
		} else if (K == 2048 || K == 4096) {
			run_benchmark_out(bench_rms_norm_f32x4, x, "f32x4", out);
			run_benchmark_ret(bench_naive, x, "f32_th");
		}

		std::cout << std::string(85, '-') << std::endl;

		Tensor x_f16 = x.to(torch::kFloat16).contiguous();
		Tensor out_f16 = torch::zeros_like(x_f16).contiguous();

		auto bench_rms_norm_f16_f16 = [&](const Tensor& in, Tensor& o, float g) {
			rms_norm_f16_f16(
				reinterpret_cast<half*>(in.data_ptr<at::Half>()),
				reinterpret_cast<half*>(o.data_ptr<at::Half>()),
				g, N, K, stream);
		};
		auto bench_rms_norm_f16_f32 = [&](const Tensor& in, Tensor& o, float g) {
			rms_norm_f16_f32(
				reinterpret_cast<half*>(in.data_ptr<at::Half>()),
				reinterpret_cast<half*>(o.data_ptr<at::Half>()),
				g, N, K, stream);
		};
		auto bench_rms_norm_f16x2_f16 = [&](const Tensor& in, Tensor& o, float g) {
			rms_norm_f16x2_f16(
				reinterpret_cast<half*>(in.data_ptr<at::Half>()),
				reinterpret_cast<half*>(o.data_ptr<at::Half>()),
				g, N, K, stream);
		};
		auto bench_rms_norm_f16x8_f16 = [&](const Tensor& in, Tensor& o, float g) {
			rms_norm_f16x8_f16(
				reinterpret_cast<half*>(in.data_ptr<at::Half>()),
				reinterpret_cast<half*>(o.data_ptr<at::Half>()),
				g, N, K, stream);
		};
		auto bench_rms_norm_f16x8_f32 = [&](const Tensor& in, Tensor& o, float g) {
			rms_norm_f16x8_f32(
				reinterpret_cast<half*>(in.data_ptr<at::Half>()),
				reinterpret_cast<half*>(o.data_ptr<at::Half>()),
				g, N, K, stream);
		};
		auto bench_rms_norm_f16x8_pack_f16 = [&](const Tensor& in, Tensor& o, float g) {
			rms_norm_f16x8_pack_f16(
				reinterpret_cast<half*>(in.data_ptr<at::Half>()),
				reinterpret_cast<half*>(o.data_ptr<at::Half>()),
				g, N, K, stream);
		};
		auto bench_rms_norm_f16x8_pack_f32 = [&](const Tensor& in, Tensor& o, float g) {
			rms_norm_f16x8_pack_f32(
				reinterpret_cast<half*>(in.data_ptr<at::Half>()),
				reinterpret_cast<half*>(o.data_ptr<at::Half>()),
				g, N, K, stream);
		};

		if (K == 512 || K == 1024) {
			run_benchmark_out(bench_rms_norm_f16_f16, x_f16, "f16f16", out_f16);
			run_benchmark_out(bench_rms_norm_f16_f32, x_f16, "f16f32", out_f16);
			run_benchmark_out(bench_rms_norm_f16x2_f16, x_f16, "f16x2f16", out_f16);
			run_benchmark_out(bench_rms_norm_f16x8_f16, x_f16, "f16x8f16", out_f16);
			run_benchmark_out(bench_rms_norm_f16x8_f32, x_f16, "f16x8f32", out_f16);
			run_benchmark_out(bench_rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16", out_f16);
			run_benchmark_out(bench_rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32", out_f16);
			run_benchmark_ret(bench_naive, x_f16, "f16_th");
		} else if (K == 2048) {
			run_benchmark_out(bench_rms_norm_f16x2_f16, x_f16, "f16x2f16", out_f16);
			run_benchmark_out(bench_rms_norm_f16x8_f16, x_f16, "f16x8f16", out_f16);
			run_benchmark_out(bench_rms_norm_f16x8_f32, x_f16, "f16x8f32", out_f16);
			run_benchmark_out(bench_rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16", out_f16);
			run_benchmark_out(bench_rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32", out_f16);
			run_benchmark_ret(bench_naive, x_f16, "f16_th");
		} else if (K == 4096) {
			run_benchmark_out(bench_rms_norm_f16x8_f16, x_f16, "f16x8f16", out_f16);
			run_benchmark_out(bench_rms_norm_f16x8_f32, x_f16, "f16x8f32", out_f16);
			run_benchmark_out(bench_rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16", out_f16);
			run_benchmark_out(bench_rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32", out_f16);
			run_benchmark_ret(bench_naive, x_f16, "f16_th");
		}

		if (N == 4096 && K == 512) {
			std::cout << std::string(85, '-') << std::endl;
			std::cout << std::setw(40) << "f16 overflow without f32" << std::endl;
			std::cout << std::string(85, '-') << std::endl;

			x_f16 = (x.to(torch::kFloat16) * 100).contiguous();

			run_benchmark_out(bench_rms_norm_f16_f16, x_f16, "f16f16", out_f16);
			run_benchmark_out(bench_rms_norm_f16_f32, x_f16, "f16f32", out_f16);
			run_benchmark_out(bench_rms_norm_f16x2_f16, x_f16, "f16x2f16", out_f16);
			run_benchmark_out(bench_rms_norm_f16x8_f16, x_f16, "f16x8f16", out_f16);
			run_benchmark_out(bench_rms_norm_f16x8_f32, x_f16, "f16x8f32", out_f16);
			run_benchmark_out(bench_rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16", out_f16);
			run_benchmark_out(bench_rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32", out_f16);
			run_benchmark_ret(bench_naive, x_f16, "f16_th");
		}
	} else {
		Tensor x_f16 = torch::randn({ N, K }, opts_f16).contiguous();
		Tensor out_f16 = torch::zeros_like(x_f16).contiguous();

		auto bench_rms_norm_f16x8_f16 = [&](const Tensor& in, Tensor& o, float g) {
			rms_norm_f16x8_f16(
				reinterpret_cast<half*>(in.data_ptr<at::Half>()),
				reinterpret_cast<half*>(o.data_ptr<at::Half>()),
				g, N, K, stream);
		};
		auto bench_rms_norm_f16x8_f32 = [&](const Tensor& in, Tensor& o, float g) {
			rms_norm_f16x8_f32(
				reinterpret_cast<half*>(in.data_ptr<at::Half>()),
				reinterpret_cast<half*>(o.data_ptr<at::Half>()),
				g, N, K, stream);
		};
		auto bench_rms_norm_f16x8_pack_f16 = [&](const Tensor& in, Tensor& o, float g) {
			rms_norm_f16x8_pack_f16(
				reinterpret_cast<half*>(in.data_ptr<at::Half>()),
				reinterpret_cast<half*>(o.data_ptr<at::Half>()),
				g, N, K, stream);
		};
		auto bench_rms_norm_f16x8_pack_f32 = [&](const Tensor& in, Tensor& o, float g) {
			rms_norm_f16x8_pack_f32(
				reinterpret_cast<half*>(in.data_ptr<at::Half>()),
				reinterpret_cast<half*>(o.data_ptr<at::Half>()),
				g, N, K, stream);
		};

		run_benchmark_out(bench_rms_norm_f16x8_f16, x_f16, "f16x8f16", out_f16);
		run_benchmark_out(bench_rms_norm_f16x8_f32, x_f16, "f16x8f32", out_f16);
		run_benchmark_out(bench_rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16", out_f16);
		run_benchmark_out(bench_rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32", out_f16);
		run_benchmark_ret(bench_naive, x_f16, "f16_th");
	}

	std::cout << std::string(85, '-') << std::endl;
}

int main()
{
	if (!torch::cuda::is_available()) {
		std::cerr << "CUDA is not available." << std::endl;
		return -1;
	}

	torch::NoGradGuard no_grad;

	benchmark_case(4096, 512);
	benchmark_case(4096, 1024);
	benchmark_case(4096, 2048);
	benchmark_case(4096, 4096);
	benchmark_case(4096, 8192);
	benchmark_case(8192, 8192);

	return 0;
}