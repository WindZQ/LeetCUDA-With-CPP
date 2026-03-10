#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "rope.cuh"

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
std::pair<Tensor, double> run_benchmark_out(
	Func perf_func,
	const Tensor& a,
	const std::string& tag,
	Tensor& out,
	int warmup = 2,
	int iters = 20,
	bool show_all = false)
{
	out.fill_(0);

	for (int i = 0; i < warmup; ++i) 
	{
		perf_func(a, out);
	}

	CUDA_CHECK(cudaDeviceSynchronize());
	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < iters; ++i) 
	{
		perf_func(a, out);
	}

	CUDA_CHECK(cudaDeviceSynchronize());
	auto end = std::chrono::high_resolution_clock::now();

	double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
	double mean_ms = total_ms / iters;

	auto flat = out.flatten().to(torch::kCPU);
	auto acc = flat.accessor<float, 1>();
	int n = std::min<int64_t>(3, flat.numel());

	std::cout << std::setw(20) << ("out_" + tag) << ": [";
	for (int i = 0; i < n; ++i) 
	{
		std::cout << std::fixed << std::setprecision(8) << acc[i];
		if (i != n - 1) std::cout << ", ";
	}
	std::cout << "], time:" << std::fixed << std::setprecision(6)
		<< mean_ms << "ms" << std::endl;

	if (show_all) {
		std::cout << out << std::endl;
	}

	return { out.clone(), mean_ms };
}

template <typename Func>
std::pair<Tensor, double> run_benchmark_ret(
	Func perf_func,
	const Tensor& a,
	const std::string& tag,
	int warmup = 2,
	int iters = 20,
	bool show_all = false)
{
	Tensor out;

	for (int i = 0; i < warmup; ++i) 
	{
		out = perf_func(a);
	}

	CUDA_CHECK(cudaDeviceSynchronize());
	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < iters; ++i) 
	{
		out = perf_func(a);
	}

	CUDA_CHECK(cudaDeviceSynchronize());
	auto end = std::chrono::high_resolution_clock::now();

	double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
	double mean_ms = total_ms / iters;

	auto flat = out.flatten().to(torch::kCPU);
	auto acc = flat.accessor<float, 1>();
	int n = std::min<int64_t>(3, flat.numel());

	std::cout << std::setw(20) << ("out_" + tag) << ": [";
	for (int i = 0; i < n; ++i) 
	{
		std::cout << std::fixed << std::setprecision(8) << acc[i];
		if (i != n - 1) std::cout << ", ";
	}
	std::cout << "], time:" << std::fixed << std::setprecision(6)
		<< mean_ms << "ms" << std::endl;

	if (show_all) {
		std::cout << out << std::endl;
	}

	return { out.clone(), mean_ms };
}

Tensor naive_rope(const Tensor& x, double theta = 10000.0) 
{
	TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
	TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
	TORCH_CHECK(x.dim() == 2, "x must be 2D, got dim=", x.dim());

	int64_t seq_len = x.size(-2);
	int64_t dim = x.size(-1);

	TORCH_CHECK(dim % 2 == 0, "last dim must be even");

	auto x_ = x.to(torch::kFloat32).reshape({ seq_len, dim / 2, 2 });
	auto x_complex = torch::view_as_complex(x_);

	auto opts = torch::TensorOptions().device(x.device()).dtype(torch::kFloat32);

	auto ar = torch::arange(0, dim, 2, opts);
	auto freqs = 1.0 / torch::pow(torch::tensor(theta, opts), ar / (double)dim);

	auto t = torch::arange(seq_len, opts);
	auto outer = torch::outer(t, freqs);

	auto ones = torch::ones_like(outer);
	auto freqs_cis = torch::polar(ones, outer);

	auto out = torch::view_as_real(x_complex * freqs_cis).flatten(1);
	return out.to(x.scalar_type());
}

void benchmark_case(int M, int N) 
{
	std::cout << std::setw(50) << ("M=" + std::to_string(M) + ", N=" + std::to_string(N)) << std::endl;
	std::cout << std::string(100, '-') << std::endl;

	auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);

	Tensor x = torch::randn({ M, N }, opts).contiguous();
	Tensor out = torch::zeros_like(x).contiguous();

	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	auto bench_rope_f32 = [&](const Tensor& in, Tensor& o) {
		rope_f32(
			in.data_ptr<float>(),
			o.data_ptr<float>(),
			M,
			N,
			stream
		);
	};

	auto bench_rope_f32x4_pack = [&](const Tensor& in, Tensor& o) {
		rope_f32x4_pack(
			in.data_ptr<float>(),
			o.data_ptr<float>(),
			M,
			N,
			stream
		);
	};

	auto bench_naive_rope = [&](const Tensor& in) -> Tensor {
		return naive_rope(in);
	};

	run_benchmark_out(bench_rope_f32, x, "f32", out);
	run_benchmark_out(bench_rope_f32x4_pack, x, "f32x4_pack", out);
	run_benchmark_ret(bench_naive_rope, x, "f32_th");

	std::cout << std::string(100, '-') << std::endl;
}

int main() 
{
	if (!torch::cuda::is_available()) {
		std::cerr << "CUDA is not available." << std::endl;
		return -1;
	}

	torch::NoGradGuard no_grad;

	std::cout << std::string(100, '-') << std::endl;

	benchmark_case(4096, 512);
	benchmark_case(4096, 1024);
	benchmark_case(8192, 512);
	benchmark_case(8192, 1024);

	return 0;
}