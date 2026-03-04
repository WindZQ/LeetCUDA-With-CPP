#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "block_all_reduce.cuh"

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

	return ms / static_cast<double>(iters);
}

void print_scalar(const std::string& tag, const torch::Tensor& out_scalar, double mean_ms)
{
	if (tag.rfind("i8", 0) == 0) {
		auto v = out_scalar.item<int32_t>();
		std::cout << std::setw(25) << ("out_" + tag) << ": " << std::left << std::setw(15) << v
			<< ", time:" << std::fixed << std::setprecision(8) << mean_ms << "ms" << std::endl;
	} else {
		auto v = out_scalar.item<float>();
		std::cout << std::setw(25) << ("out_" + tag) << ": " << std::left << std::setw(15)
			<< std::fixed << std::setprecision(8) << v
			<< ", time:" << std::fixed << std::setprecision(8) << mean_ms << "ms" << std::endl;
	}
}

torch::Tensor reduce_y_to_scalar(const torch::Tensor& y_vec_float) 
{
	return y_vec_float.sum().reshape({}); 
}

torch::Tensor reduce_y_to_scalar_i32(const torch::Tensor& y_vec_i32) 
{
	return y_vec_i32.sum().reshape({}); 
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

	std::vector<int64_t> Ss = { 1024, 2048, 4096 };
	std::vector<int64_t> Ks = { 1024, 2048, 4096 };

	for (auto S : Ss) 
	{
		for (auto K : Ks) 
		{
			std::cout << std::string(80, '-') << std::endl;
			std::cout << std::string(40, ' ') << "S=" << S << ", K=" << K << std::endl;

			auto opts_f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
			auto values = torch::randn({ S, K }, opts_f32);

			{
				auto y = torch::empty({ S }, opts_f32);

				auto ms = benchmark_ms([&]()
				{
					block_all_reduce_sum_f32_f32(values.data_ptr<float>(), y.data_ptr<float>(), (int)S, (int)K, stream);
				}, stream);

				auto out_scalar = reduce_y_to_scalar(y);
				print_scalar("f32f32", out_scalar, ms);
			}

			{
				auto y = torch::empty({ S }, opts_f32);
				auto ms = benchmark_ms([&]() 
				{
					block_all_reduce_sum_f32x4_f32(values.data_ptr<float>(), y.data_ptr<float>(), (int)S, (int)K, stream);
				}, stream);
				auto out_scalar = reduce_y_to_scalar(y);
				print_scalar("f32x4f32", out_scalar, ms);
			}

			{
				torch::Tensor out;
				auto ms = benchmark_ms([&]() { out = values.sum(); }, stream);
				print_scalar("f32f32_th", out, ms);
			}

			std::cout << std::string(80, '-') << "\n";

			auto values_half = values.to(torch::kFloat16);
			auto opts_yf32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);

			auto run_f16 = [&](const char* tag, auto fn_launcher) 
			{
				auto y = torch::empty({ S }, opts_yf32);
				auto ms = benchmark_ms([&]() 
				{
					fn_launcher((const at::Half*)values_half.data_ptr<at::Half>(),
						y.data_ptr<float>(), (int)S, (int)K, stream);
				}, stream);
				auto out_scalar = reduce_y_to_scalar(y);
				print_scalar(tag, out_scalar, ms);
			};

			run_f16("f16f16", block_all_reduce_sum_f16_f16);
			run_f16("f16f32", block_all_reduce_sum_f16_f32);
			run_f16("f16x2f32", block_all_reduce_sum_f16x2_f32);
			run_f16("f16x2f16", block_all_reduce_sum_f16x2_f16);
			run_f16("f16x8packf16", block_all_reduce_sum_f16x8_pack_f16);
			run_f16("f16x8packf32", block_all_reduce_sum_f16x8_pack_f32);

			{
				torch::Tensor out;
				auto ms = benchmark_ms([&]() { out = values_half.sum(); }, stream);
				print_scalar("f16f16_th", out.to(torch::kFloat32), ms);
			}

			std::cout << std::string(80, '-') << "\n";

			auto values_bf16 = values.to(torch::kBFloat16);
			auto run_bf16 = [&](const char* tag, auto fn_launcher) 
			{
				auto y = torch::empty({ S }, opts_yf32);
				auto ms = benchmark_ms([&]()
				{
					fn_launcher((const at::BFloat16*)values_bf16.data_ptr<at::BFloat16>(),
					y.data_ptr<float>(), (int)S, (int)K, stream);
				}, stream);
				auto out_scalar = reduce_y_to_scalar(y);
				print_scalar(tag, out_scalar, ms);
			};

			run_bf16("bf16bf16", block_all_reduce_sum_bf16_bf16);
			run_bf16("bf16f32", block_all_reduce_sum_bf16_f32);
			run_bf16("bf16x2f32", block_all_reduce_sum_bf16x2_f32);
			run_bf16("bf16x2bf16", block_all_reduce_sum_bf16x2_bf16);
			run_bf16("bf16x8packf32", block_all_reduce_sum_bf16x8_pack_f32);
			run_bf16("bf16x8packbf16", block_all_reduce_sum_bf16x8_pack_bf16);

			{
				torch::Tensor out;
				auto ms = benchmark_ms([&]() { out = values_bf16.sum(); }, stream);
				print_scalar("bf16bf16_th", out.to(torch::kFloat32), ms);
			}

			std::cout << std::string(80, '-') << "\n";

			bool fp8_ok = false;
			try 
			{
				(void)torch::empty({ 1 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat8_e4m3fn));
				fp8_ok = true;
			}
			catch (...) 
			{
				fp8_ok = false;
			}

			if (fp8_ok) {
				auto values_f8e4m3 = values.to(torch::kFloat8_e4m3fn);
				auto y = torch::empty({ S }, opts_yf32);

				auto ms = benchmark_ms([&]()
				{
					block_all_reduce_sum_fp8_e4m3_f16(
						(const uint8_t*)values_f8e4m3.data_ptr(),
						y.data_ptr<float>(), (int)S, (int)K, stream);
				}, stream);
				print_scalar("f8e4m3f16", reduce_y_to_scalar(y), ms);

				y.zero_();
				ms = benchmark_ms([&]() 
				{
					block_all_reduce_sum_fp8_e4m3x16_pack_f16(
						(const uint8_t*)values_f8e4m3.data_ptr(),
						y.data_ptr<float>(), (int)S, (int)K, stream);
				}, stream);
				print_scalar("f8e4m3x16packf16", reduce_y_to_scalar(y), ms);

				torch::Tensor out;
				auto ms_th = benchmark_ms([&]() { out = values_f8e4m3.to(torch::kFloat16).sum(); }, stream);
				print_scalar("f8e4m3f16_th", out.to(torch::kFloat32), ms_th);

				std::cout << std::string(80, '-') << "\n";

				auto values_f8e5m2 = values.to(torch::kFloat8_e5m2);
				y.zero_();
				ms = benchmark_ms([&]()
				{
					block_all_reduce_sum_fp8_e5m2_f16(
						(const uint8_t*)values_f8e5m2.data_ptr(),
						y.data_ptr<float>(), (int)S, (int)K, stream);
				}, stream);
				print_scalar("f8e5m2f16", reduce_y_to_scalar(y), ms);

				y.zero_();
				ms = benchmark_ms([&]()
				{
					block_all_reduce_sum_fp8_e5m2x16_pack_f16(
						(const uint8_t*)values_f8e5m2.data_ptr(),
						y.data_ptr<float>(), (int)S, (int)K, stream);
				}, stream);
				print_scalar("f8e5m2x16packf16", reduce_y_to_scalar(y), ms);

				ms_th = benchmark_ms([&]() { out = values_f8e5m2.to(torch::kFloat16).sum(); }, stream);
				print_scalar("f8e5m2f16_th", out.to(torch::kFloat32), ms_th);
			} else {
				std::cout << "(Skip FP8: LibTorch build does not support float8 on this setup)\n";
			}

			std::cout << std::string(80, '-') << "\n";

			auto values_i8 = values.to(torch::kInt8);
			auto opts_i32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);

			{
				auto y = torch::empty({ S }, opts_i32);
				auto ms = benchmark_ms([&]() 
				{
					block_all_reduce_sum_i8_i32(values_i8.data_ptr<int8_t>(), y.data_ptr<int32_t>(),
						(int)S, (int)K, stream);
				}, stream);

				auto out_scalar = reduce_y_to_scalar_i32(y);
				print_scalar("i8i32", out_scalar, ms);
			}

			{
				auto y = torch::empty({ S }, opts_i32);
				auto ms = benchmark_ms([&]() 
				{
					block_all_reduce_sum_i8x16_pack_i32(values_i8.data_ptr<int8_t>(), y.data_ptr<int32_t>(), 
						(int)S, (int)K, stream);
				}, stream);

				auto out_scalar = reduce_y_to_scalar_i32(y);
				print_scalar("i8x16packi32", out_scalar, ms);
			}

			{
				torch::Tensor out;
				auto ms = benchmark_ms([&]() { out = values_i8.sum(); }, stream);
				print_scalar("i8i32_th", out.to(torch::kInt32), ms);
			}

			std::cout << std::string(80, '-') << "\n";
		}
	}

	return 0;
}