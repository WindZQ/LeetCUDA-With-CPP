#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "dot_product.cuh"

static inline int to_int_numel(int64_t n) 
{
	TORCH_CHECK(n >= 0 && n <= std::numeric_limits<int>::max(), "numel out of int range");
	return static_cast<int>(n);
}

torch::Tensor call_dot_f32_f32(const torch::Tensor& a, const torch::Tensor& b) 
{
	TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a/b must be CUDA tensors");
	TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32, "a/b must be float32");
	auto aa = a.contiguous().view({ -1 });
	auto bb = b.contiguous().view({ -1 });
	TORCH_CHECK(aa.numel() == bb.numel(), "numel mismatch");
	auto out = torch::zeros({ 1 }, aa.options().dtype(torch::kFloat32));
	auto stream = at::cuda::getDefaultCUDAStream().stream();
	dot_prod_f32_f32(aa.data_ptr<float>(), bb.data_ptr<float>(), to_int_numel(aa.numel()), out.data_ptr<float>(), stream);

	return out;
}

torch::Tensor call_dot_f32x4_f32(const torch::Tensor& a, const torch::Tensor& b)
{
	TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a/b must be CUDA tensors");
	TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32,
		"a/b must be float32");
	auto aa = a.contiguous().view({ -1 });
	auto bb = b.contiguous().view({ -1 });
	TORCH_CHECK(aa.numel() == bb.numel(), "numel mismatch");
	auto out = torch::zeros({ 1 }, aa.options().dtype(torch::kFloat32));
	auto stream = at::cuda::getDefaultCUDAStream().stream();
	dot_prod_f32x4_f32(aa.data_ptr<float>(), bb.data_ptr<float>(), to_int_numel(aa.numel()), out.data_ptr<float>(), stream);

	return out;
}

torch::Tensor call_dot_f16_f32(const torch::Tensor& a, const torch::Tensor& b)
{
	TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a/b must be CUDA tensors");
	TORCH_CHECK(a.scalar_type() == torch::kFloat16 && b.scalar_type() == torch::kFloat16,
		"a/b must be float16");
	auto aa = a.contiguous().view({ -1 });
	auto bb = b.contiguous().view({ -1 });
	TORCH_CHECK(aa.numel() == bb.numel(), "numel mismatch");
	auto out = torch::zeros({ 1 }, aa.options().dtype(torch::kFloat32));
	auto stream = at::cuda::getDefaultCUDAStream().stream();

	dot_prod_f16_f32(reinterpret_cast<const half*>(aa.data_ptr<at::Half>()),
		reinterpret_cast<const half*>(bb.data_ptr<at::Half>()), 
		to_int_numel(aa.numel()), out.data_ptr<float>(), stream);

	return out;
}

torch::Tensor call_dot_f16x2_f32(const torch::Tensor& a, const torch::Tensor& b) 
{
	TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a/b must be CUDA tensors");
	TORCH_CHECK(a.scalar_type() == torch::kFloat16 && b.scalar_type() == torch::kFloat16,
		"a/b must be float16");
	auto aa = a.contiguous().view({ -1 });
	auto bb = b.contiguous().view({ -1 });
	TORCH_CHECK(aa.numel() == bb.numel(), "numel mismatch");
	auto out = torch::zeros({ 1 }, aa.options().dtype(torch::kFloat32));
	auto stream = at::cuda::getDefaultCUDAStream().stream();

	dot_prod_f16x2_f32(reinterpret_cast<const half*>(aa.data_ptr<at::Half>()),
		reinterpret_cast<const half*>(bb.data_ptr<at::Half>()),
		to_int_numel(aa.numel()), out.data_ptr<float>(), stream);

	return out;
}

torch::Tensor call_dot_f16x8_pack_f32(const torch::Tensor& a, const torch::Tensor& b) 
{
	TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a/b must be CUDA tensors");
	TORCH_CHECK(a.scalar_type() == torch::kFloat16 && b.scalar_type() == torch::kFloat16,
		"a/b must be float16");
	auto aa = a.contiguous().view({ -1 });
	auto bb = b.contiguous().view({ -1 });
	TORCH_CHECK(aa.numel() == bb.numel(), "numel mismatch");
	auto out = torch::zeros({ 1 }, aa.options().dtype(torch::kFloat32));
	auto stream = at::cuda::getDefaultCUDAStream().stream();

	dot_prod_f16x8_pack_f32(reinterpret_cast<const half*>(aa.data_ptr<at::Half>()),
		reinterpret_cast<const half*>(bb.data_ptr<at::Half>()),
		to_int_numel(aa.numel()), out.data_ptr<float>(), stream);

	return out;
}

using PerfFn = std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)>;

double run_benchmark(const PerfFn& fn, const torch::Tensor& a, const torch::Tensor& b,
	const std::string& tag, int warmup = 10, int iters = 1000) {
	torch::Tensor out;
	for (int i = 0; i < warmup; ++i) out = fn(a, b);
	cudaDeviceSynchronize();

	auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iters; ++i) out = fn(a, b);
	cudaDeviceSynchronize();
	auto t1 = std::chrono::high_resolution_clock::now();

	double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	double mean_ms = total_ms / iters;
	float out_val = out.to(torch::kFloat32).item<float>();

	std::cout << std::setw(17) << ("out_" + tag) << ": "
		<< std::setw(15) << std::fixed << std::setprecision(8) << out_val
		<< ", time:" << std::setprecision(8) << mean_ms << "ms\n";
	return mean_ms;
}

int main() 
{
	torch::NoGradGuard no_grad;

	if (!torch::cuda::is_available()) {
		std::cerr << "CUDA is not available.\n";
		return 1;
	}

	std::vector<int> Ss{ 1024, 2048, 4096 };
	std::vector<int> Ks{ 1024, 2048, 4096 };

	auto f32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);

	for (int S : Ss) 
	{
		for (int K : Ks) 
		{
			std::cout << std::string(80, '-') << "\n";
			std::cout << std::string(25, ' ') << "S=" << S << ", K=" << K << "\n";

			auto a = torch::randn({ S * K }, f32_opts);
			auto b = torch::randn({ S * K }, f32_opts);

			run_benchmark(call_dot_f32_f32, a, b, "f32f32");
			run_benchmark(call_dot_f32x4_f32, a, b, "f32x4f32");
			run_benchmark(
				[](const torch::Tensor& x, const torch::Tensor& y) { return torch::dot(x, y); },
				a, b, "f32f32_th");

			std::cout << std::string(80, '-') << "\n";

			auto a_f16 = a.to(torch::kFloat16);
			auto b_f16 = b.to(torch::kFloat16);

			run_benchmark(call_dot_f16_f32, a_f16, b_f16, "f16f32");
			run_benchmark(call_dot_f16x2_f32, a_f16, b_f16, "f16x2f32");
			run_benchmark(call_dot_f16x8_pack_f32, a_f16, b_f16, "f16x8packf32");
			run_benchmark(
				[](const torch::Tensor& x, const torch::Tensor& y) { return torch::dot(x, y); },
				a_f16, b_f16, "f16f16_th");

			std::cout << std::string(80, '-') << "\n";
		}
	}

	return 0;
}