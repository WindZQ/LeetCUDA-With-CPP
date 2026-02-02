#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>

#include <iostream>
#include <vector>

#include "elementwise.cuh"

static float bench(std::function<void()> fn, int warmup = 10, int iters = 1000) 
{
	for (int i = 0; i < warmup; i++) fn();
	cudaDeviceSynchronize();

	cudaEvent_t st, ed;
	cudaEventCreate(&st); cudaEventCreate(&ed);
	cudaEventRecord(st);

	for (int i = 0; i < iters; i++) fn();

	cudaEventRecord(ed);
	cudaEventSynchronize(ed);

	float ms = 0.f;
	cudaEventElapsedTime(&ms, st, ed);
	cudaEventDestroy(st); cudaEventDestroy(ed);
	return ms / iters;
}

static void print_head2(const torch::Tensor& t, const std::string& tag, float ms) 
{
	auto cpu = t.flatten().slice(0, 0, 2).to(torch::kCPU);
	std::cout << std::string(18 - (int)tag.size(), ' ') << "out_" << tag
		<< ": [" << cpu[0].item<float>() << ", " << cpu[1].item<float>() << "]"
		<< ", time:" << ms << "ms\n";
}

int main() 
{
	torch::NoGradGuard no_grad;

	std::vector<int64_t> Ss{ 1024, 2048, 4096 };
	std::vector<int64_t> Ks{ 1024, 2048, 4096 };

	for (auto S : Ss) for (auto K : Ks) 
	{
		std::cout << "-------------------------------------------------------------------------------------\n";
		std::cout << "                                        S=" << S << ", K=" << K << "\n";

		auto a = torch::randn({ S, K }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).contiguous();
		auto b = torch::randn({ S, K }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).contiguous();
		auto c = torch::zeros_like(a).contiguous();

		int64_t N = a.numel();

		cudaStream_t stream = at::cuda::getDefaultCUDAStream();

		float t1 = bench([&] { elementwise_add_f32(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N, S, K, stream); });
		print_head2(c, "f32", t1);

		float t2 = bench([&] { elementwise_add_f32x4(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N, S, K, stream); });
		print_head2(c, "f32x4", t2);

		float t3 = bench([&] { torch::add_out(c, a, b); });
		print_head2(c, "f32_th", t3);

		std::cout << "-------------------------------------------------------------------------------------\n";

		auto a16 = a.to(torch::kFloat16).contiguous();
		auto b16 = b.to(torch::kFloat16).contiguous();
		auto c16 = torch::zeros_like(a16).contiguous();

		auto pa = reinterpret_cast<__half*>(a16.data_ptr<at::Half>());
		auto pb = reinterpret_cast<__half*>(b16.data_ptr<at::Half>());
		auto pc = reinterpret_cast<__half*>(c16.data_ptr<at::Half>());

		float h1 = bench([&] { elementwise_add_f16(pa, pb, pc, N, S, K, stream); });
		print_head2(c16.to(torch::kFloat32), "f16", h1);

		float h2 = bench([&] { elementwise_add_f16x2(pa, pb, pc, N, S, K, stream); });
		print_head2(c16.to(torch::kFloat32), "f16x2", h2);

		float h3 = bench([&] { elementwise_add_f16x8(pa, pb, pc, N, S, K, stream); });
		print_head2(c16.to(torch::kFloat32), "f16x8", h3);

		float h4 = bench([&] { elementwise_add_f16x8_pack(pa, pb, pc, N, S, K, stream); });
		print_head2(c16.to(torch::kFloat32), "f16x8pack", h4);

		float h5 = bench([&] { torch::add_out(c16, a16, b16); });
		print_head2(c16.to(torch::kFloat32), "f16_th", h5);

		std::cout << "-------------------------------------------------------------------------------------\n";
	}

	return 0;
}