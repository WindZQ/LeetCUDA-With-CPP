#include <torch/torch.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "hgemv.cuh"
#include "hgemv_cute.cuh"

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__              \
                << " - " << cudaGetErrorString(err) << std::endl;               \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

inline void check_hgemv_tensors(const torch::Tensor& a,
	const torch::Tensor& b,
	const torch::Tensor& c) {
	TORCH_CHECK(a.defined() && b.defined() && c.defined(), "tensors must be defined");
	TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "all tensors must be CUDA");
	TORCH_CHECK(a.scalar_type() == torch::kFloat16, "a must be float16");
	TORCH_CHECK(b.scalar_type() == torch::kFloat16, "b must be float16");
	TORCH_CHECK(c.scalar_type() == torch::kFloat16, "c must be float16");
	TORCH_CHECK(a.dim() == 2, "a must be 2D");
	TORCH_CHECK(b.dim() == 2, "b must be 2D");
	TORCH_CHECK(c.dim() == 2, "c must be 2D");
	TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
	TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
	TORCH_CHECK(c.is_contiguous(), "c must be contiguous");

	TORCH_CHECK(a.size(1) == b.size(0), "a.size(1) must equal b.size(0)");
	TORCH_CHECK(a.size(0) == c.size(0), "a.size(0) must equal c.size(0)");
	TORCH_CHECK(b.size(1) == c.size(1), "b.size(1) must equal c.size(1)");
}

inline void print_first3(const torch::Tensor& out,
	const std::string& tag,
	double mean_ms)
{
	auto cpu = out.flatten().slice(0, 0, 3).to(torch::kFloat).cpu();
	std::vector<float> vals(cpu.data_ptr<float>(), cpu.data_ptr<float>() + cpu.numel());

	std::cout << std::setw(24) << ("out_" + tag) << ": [";
	for (size_t i = 0; i < vals.size(); ++i) {
		std::cout << std::fixed << std::setprecision(8) << vals[i];
		if (i + 1 != vals.size()) std::cout << ", ";
	}
	std::cout << "], time: " << std::fixed << std::setprecision(8)
		<< mean_ms << " ms" << std::endl;
}

template <auto KernelFunc>
struct KernelTraits 
{
	static void check([[maybe_unused]] int M, [[maybe_unused]] int N, [[maybe_unused]] int K) {}
	static constexpr const char* name = "unnamed_kernel";
};

template <auto KernelFunc>
struct RawKernelCaller 
{
	static void run(const torch::Tensor& a,
		const torch::Tensor& b,
		torch::Tensor& c) 
	{
		check_hgemv_tensors(a, b, c);

		int M = static_cast<int>(a.size(0));
		int K = static_cast<int>(a.size(1));
		int N = static_cast<int>(b.size(1));

		TORCH_CHECK(N == 1, "This HGEMV benchmark expects b shape [K, 1] and c shape [M, 1]");
		KernelTraits<KernelFunc>::check(M, N, K);

		KernelFunc(
			reinterpret_cast<half*>(a.data_ptr<at::Half>()),
			reinterpret_cast<half*>(b.data_ptr<at::Half>()),
			reinterpret_cast<half*>(c.data_ptr<at::Half>()),
			M, K);
	}
};

struct TorchMatmulCaller 
{
	static void run(const torch::Tensor& a,
		const torch::Tensor& b,
		torch::Tensor& c) 
	{
		check_hgemv_tensors(a, b, c);
		torch::matmul_out(c, a, b);
	}
};

template <typename Caller>
torch::Tensor run_benchmark(const std::string& tag,
	const torch::Tensor& a,
	const torch::Tensor& b,
	torch::Tensor c,
	int warmup = 10,
	int iters = 200,
	bool show_all = false) 
{
	c.zero_();

	for (int i = 0; i < warmup; ++i) 
	{
		Caller::run(a, b, c);
	}

	CUDA_CHECK(cudaDeviceSynchronize());

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < iters; ++i) 
	{
		Caller::run(a, b, c);
	}

	CUDA_CHECK(cudaDeviceSynchronize());

	auto end = std::chrono::high_resolution_clock::now();
	double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
	double mean_ms = total_ms / iters;

	print_first3(c, tag, mean_ms);

	if (show_all) {
		std::cout << c << std::endl;
	}

	return c.clone();
}

template <auto KernelFunc>
torch::Tensor bench_kernel(const std::string& tag,
	const torch::Tensor& a,
	const torch::Tensor& b,
	torch::Tensor c,
	int warmup = 10,
	int iters = 200,
	bool show_all = false) 
{
	return run_benchmark<RawKernelCaller<KernelFunc>>(
		tag, a, b, c, warmup, iters, show_all);
}

inline torch::Tensor bench_torch(const std::string& tag,
	const torch::Tensor& a,
	const torch::Tensor& b,
	torch::Tensor c,
	int warmup = 10,
	int iters = 200,
	bool show_all = false) 
{
	return run_benchmark<TorchMatmulCaller>(tag, a, b, c, warmup, iters, show_all);
}

template <>
struct KernelTraits<hgemv_k32_f16> 
{
	static void check(int, int, int K) 
	{
		TORCH_CHECK(K % 32 == 0, "hgemv_k32_f16 requires K % 32 == 0");
	}
	static constexpr const char* name = "k32f16";
};

template <>
struct KernelTraits<hgemv_k128_f16x4> 
{
	static void check(int, int, int K) 
	{
		TORCH_CHECK(K % 128 == 0, "hgemv_k128_f16x4 requires K % 128 == 0");
	}
	static constexpr const char* name = "k128f16x4";
};

template <>
struct KernelTraits<hgemv_k16_f16> {
	static void check(int, int, int K) {
		TORCH_CHECK(K == 16, "hgemv_k16_f16 requires K == 16");
	}
	static constexpr const char* name = "k16f16";
};

template <>
struct KernelTraits<hgemv_f16_cute> 
{
	static void check(int, int, int) {}
	static constexpr const char* name = "hgemv_f16_cute";
};

template <>
struct KernelTraits<hgemv_f16x8_cute> 
{
	static void check(int, int, int K) {
		TORCH_CHECK(K % 8 == 0, "hgemv_f16x8_cute requires K % 8 == 0");
	}
	static constexpr const char* name = "hgemv_f16x8_cute";
};

template <>
struct KernelTraits<hgemv_tensor_core_cute> 
{
	static void check(int, int, int) {}
	static constexpr const char* name = "hgemv_tensor_core_cute";
};

int main() 
{
	torch::NoGradGuard no_grad;

	if (!torch::cuda::is_available()) {
		std::cerr << "CUDA is not available." << std::endl;
		return -1;
	}

	std::cout << std::string(80, '-') << std::endl;

	{
		int M = 1024, N = 1, K = 128;
		auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16);

		auto a = torch::randn({ M, K }, opts).contiguous();
		auto b = torch::randn({ K, N }, opts).contiguous();
		auto c = torch::randn({ M, N }, opts).contiguous();

		bench_kernel<hgemv_k32_f16>("k32f16", a, b, c);
		bench_kernel<hgemv_k128_f16x4>("k128f16x4", a, b, c);
		bench_kernel<hgemv_f16_cute>("hgemv_f16_cute", a, b, c);
		bench_kernel<hgemv_f16x8_cute>("hgemv_f16x8_cute", a, b, c);
		bench_kernel<hgemv_tensor_core_cute>("hgemv_tensor_core_cute", a, b, c);
		bench_torch("f16_th", a, b, c);
	}

	std::cout << std::string(80, '-') << std::endl;

	{
		int M = 1024, N = 1, K = 16;
		auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16);

		auto a = torch::randn({ M, K }, opts).contiguous();
		auto b = torch::randn({ K, N }, opts).contiguous();
		auto c = torch::randn({ M, N }, opts).contiguous();

		bench_kernel<hgemv_k16_f16>("k16f16", a, b, c);
		bench_kernel<hgemv_f16_cute>("hgemv_f16_cute", a, b, c);
		bench_kernel<hgemv_f16x8_cute>("hgemv_f16x8_cute", a, b, c);
		bench_kernel<hgemv_tensor_core_cute>("hgemv_tensor_core_cute", a, b, c);
		bench_torch("f16_th", a, b, c);
	}

	std::cout << std::string(80, '-') << std::endl;
	return 0;
}