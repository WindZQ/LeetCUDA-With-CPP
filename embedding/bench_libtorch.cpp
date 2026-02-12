#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "embedding.cuh"

static inline void sync_all() { cudaDeviceSynchronize(); }

static void check_inputs(const torch::Tensor& idx,
	const torch::Tensor& weight,
	const torch::Tensor& out) 
{
	TORCH_CHECK(idx.is_cuda() && weight.is_cuda() && out.is_cuda(), "All must be CUDA tensors");
	TORCH_CHECK(idx.is_contiguous() && weight.is_contiguous() && out.is_contiguous(), "All must be contiguous");
	TORCH_CHECK(idx.dim() == 1, "idx must be 1D");
	TORCH_CHECK(weight.dim() == 2 && out.dim() == 2, "weight/out must be 2D");
	TORCH_CHECK(out.size(0) == idx.size(0), "out.size(0) must equal idx.size(0)");
	TORCH_CHECK(out.size(1) == weight.size(1), "out.size(1) must equal weight.size(1)");
}

template <typename LaunchFn>
void bench_inplace(const std::string& tag,
	LaunchFn&& launch,
	const torch::Tensor& idx,
	const torch::Tensor& weight,
	torch::Tensor& out,
	int warmup = 2, int iters = 20)
{
	out.zero_();

	for (int i = 0; i < warmup; ++i) launch(idx, weight, out);

	sync_all();
	auto t0 = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < iters; ++i) launch(idx, weight, out);

	sync_all();
	auto t1 = std::chrono::high_resolution_clock::now();
	double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	double mean_ms = total_ms / iters;

	auto flat = out.flatten().to(torch::kCPU);
	auto first3 = flat.slice(0, 0, 3).to(torch::kFloat32);
	auto acc = first3.accessor<float, 1>();

	std::cout << std::setw(23) << ("out_" + tag) << ": ["
		<< std::fixed << std::setprecision(8)
		<< std::setw(12) << acc[0] << ", "
		<< std::setw(12) << acc[1] << ", "
		<< std::setw(12) << acc[2] << "], time:"
		<< std::setprecision(6) << mean_ms << "ms\n";
}

int main() 
{
	torch::NoGradGuard ng;

	TORCH_CHECK(torch::cuda::is_available(), "CUDA is not available in this LibTorch build.");

	std::vector<int> Ms = { 1024, 4096 };
	std::vector<int> Ns = { 2048, 4096 };
	std::vector<int> Ks = { 512, 1024 };

	for (int M : Ms) for (int N : Ns) for (int K : Ks)
	{
		std::cout << std::string(110, '-') << "\n";
		std::cout << std::string(45, ' ')
			<< "MaxV=" << M << ", SeqLen=" << N << ", EmbSize=" << K << "\n";

		auto idx = torch::randint(0, M, { N },
			torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32)).contiguous();

		auto w_f32 = torch::randn({ M,K },
			torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).contiguous();

		auto o_f32 = torch::zeros({ N,K },
			torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).contiguous();

		check_inputs(idx, w_f32, o_f32);

		cudaStream_t stream = at::cuda::getDefaultCUDAStream();

		TORCH_CHECK(K <= 1024, "K must be <= 1024 for f32/f16 simple kernel");
		bench_inplace("f32",
			[&](const torch::Tensor& a, const torch::Tensor& w, torch::Tensor& o) {
				embedding_f32(a.data_ptr<int>(), w.data_ptr<float>(), o.data_ptr<float>(), N, K, stream);
		}, idx, w_f32, o_f32);

		TORCH_CHECK(K % 4 == 0 && (K / 4) <= 1024, "K must be divisible by 4 and K/4<=1024");
		bench_inplace("f32x4",
			[&](const torch::Tensor& a, const torch::Tensor& w, torch::Tensor& o) {
				embedding_f32x4(a.data_ptr<int>(), w.data_ptr<float>(), o.data_ptr<float>(), N, K, stream);
		}, idx, w_f32, o_f32);

		bench_inplace("f32x4_pack",
			[&](const torch::Tensor& a, const torch::Tensor& w, torch::Tensor& o) {
				embedding_f32x4_pack(a.data_ptr<int>(), w.data_ptr<float>(), o.data_ptr<float>(), N, K, stream);
		}, idx, w_f32, o_f32);

		{
			auto a64 = idx.to(torch::kInt64);
			for (int i = 0; i < 2; ++i) (void)w_f32.index_select(0, a64);
			sync_all();
			auto t0 = std::chrono::high_resolution_clock::now();
			torch::Tensor out;
			for (int i = 0; i < 20; ++i) out = w_f32.index_select(0, a64);
			sync_all();
			auto t1 = std::chrono::high_resolution_clock::now();
			double mean_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 20.0;

			auto flat = out.flatten().to(torch::kCPU);
			auto first3 = flat.slice(0, 0, 3).to(torch::kFloat32);
			auto acc = first3.accessor<float, 1>();
			std::cout << std::setw(23) << "out_f32_th" << ": ["
				<< std::fixed << std::setprecision(8)
				<< std::setw(12) << acc[0] << ", "
				<< std::setw(12) << acc[1] << ", "
				<< std::setw(12) << acc[2] << "], time:"
				<< std::setprecision(6) << mean_ms << "ms\n";
		}

		std::cout << std::string(110, '-') << "\n";

		auto w_f16 = torch::randn({ M,K },
			torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16)).contiguous();
		auto o_f16 = torch::zeros({ N,K },
			torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16)).contiguous();

		check_inputs(idx, w_f16, o_f16);

		bench_inplace("f16",
			[&](const torch::Tensor& a, const torch::Tensor& w, torch::Tensor& o) {
				embedding_f16(
					a.data_ptr<int>(),
					reinterpret_cast<half*>(w.data_ptr<at::Half>()),
					reinterpret_cast<half*>(o.data_ptr<at::Half>()),
					N, K, stream);
			}, idx, w_f16, o_f16);

		TORCH_CHECK(K % 8 == 0 && (K / 8) <= 1024, "K must be divisible by 8 and K/8<=1024");
		bench_inplace("f16x8",
			[&](const torch::Tensor& a, const torch::Tensor& w, torch::Tensor& o) {
				embedding_f16x8(
					a.data_ptr<int>(),
					reinterpret_cast<half*>(w.data_ptr<at::Half>()),
					reinterpret_cast<half*>(o.data_ptr<at::Half>()),
					N, K, stream);
			}, idx, w_f16, o_f16);

		bench_inplace("f16x8_pack",
			[&](const torch::Tensor& a, const torch::Tensor& w, torch::Tensor& o) {
				embedding_f16x8_pack(
					a.data_ptr<int>(),
					reinterpret_cast<half*>(w.data_ptr<at::Half>()),
					reinterpret_cast<half*>(o.data_ptr<at::Half>()),
					N, K, stream);
			}, idx, w_f16, o_f16);

		{
			auto a64 = idx.to(torch::kInt64);
			for (int i = 0; i < 2; ++i) (void)w_f16.index_select(0, a64);
			sync_all();
			auto t0 = std::chrono::high_resolution_clock::now();
			torch::Tensor out;
			for (int i = 0; i < 20; ++i) out = w_f16.index_select(0, a64);
			sync_all();
			auto t1 = std::chrono::high_resolution_clock::now();
			double mean_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 20.0;

			auto flat = out.flatten().to(torch::kCPU).to(torch::kFloat32);
			auto first3 = flat.slice(0, 0, 3);
			auto acc = first3.accessor<float, 1>();
			std::cout << std::setw(23) << "out_f16_th" << ": ["
				<< std::fixed << std::setprecision(8)
				<< std::setw(12) << acc[0] << ", "
				<< std::setw(12) << acc[1] << ", "
				<< std::setw(12) << acc[2] << "], time:"
				<< std::setprecision(6) << mean_ms << "ms\n";
		}

		std::cout << std::string(110, '-') << "\n";
	}

	return 0;
}