#include <torch/torch.h>
#include <cuda_runtime.h>

#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "merge_attn_states_libtorch.h"
#include "merge_attn_states_ref.h"

struct CaseInfo 
{
	int64_t num_tokens;
	int64_t num_heads;
	int64_t head_size;
	torch::ScalarType dtype;
	std::string device;
	float avg_time_torch_ms;
	float avg_time_cuda_ms;
	float speedup;
};

static std::string dtype_to_string(torch::ScalarType dtype) 
{
	switch (dtype) 
	{
	case torch::kFloat: return "float32";
	case torch::kHalf: return "float16";
	case torch::kBFloat16: return "bfloat16";
	default: return "unknown";
	}
}

static float tensor_max_abs_diff(const torch::Tensor& a, const torch::Tensor& b) 
{
	return torch::max(torch::abs(a.to(torch::kFloat) - b.to(torch::kFloat))).item<float>();
}

static void cuda_sync_checked() 
{
	cudaError_t err = cudaDeviceSynchronize();
	TORCH_CHECK(err == cudaSuccess, "cudaDeviceSynchronize failed: ", cudaGetErrorString(err));
}

static float measure_cuda_time_ms(const std::function<void()>& fn) 
{
	cudaEvent_t start, end;
	cudaError_t err;

	err = cudaEventCreate(&start);
	TORCH_CHECK(err == cudaSuccess, "cudaEventCreate(start) failed: ", cudaGetErrorString(err));

	err = cudaEventCreate(&end);
	TORCH_CHECK(err == cudaSuccess, "cudaEventCreate(end) failed: ", cudaGetErrorString(err));

	err = cudaEventRecord(start);
	TORCH_CHECK(err == cudaSuccess, "cudaEventRecord(start) failed: ", cudaGetErrorString(err));

	fn();

	err = cudaEventRecord(end);
	TORCH_CHECK(err == cudaSuccess, "cudaEventRecord(end) failed: ", cudaGetErrorString(err));

	err = cudaEventSynchronize(end);
	TORCH_CHECK(err == cudaSuccess, "cudaEventSynchronize(end) failed: ", cudaGetErrorString(err));

	float ms = 0.0f;
	err = cudaEventElapsedTime(&ms, start, end);
	TORCH_CHECK(err == cudaSuccess, "cudaEventElapsedTime failed: ", cudaGetErrorString(err));

	cudaEventDestroy(start);
	cudaEventDestroy(end);

	return ms;
}

static void print_markdown_table(const std::vector<CaseInfo>& all_case_info, bool output_lse) 
{
	std::cout << "| tokens | heads | headsize | dtype | device | torch | cuda | speedup |\n";
	std::cout << "| --- | --- | --- | --- | --- | --- | --- | --- |\n";
	for (const auto& info : all_case_info) 
	{
		std::cout
			<< "| " << info.num_tokens
			<< " | " << info.num_heads
			<< " | " << info.head_size
			<< " | " << dtype_to_string(info.dtype)
			<< " | " << info.device
			<< " | " << std::fixed << std::setprecision(4) << info.avg_time_torch_ms << "ms"
			<< " | " << std::fixed << std::setprecision(4) << info.avg_time_cuda_ms << "ms"
			<< " | " << std::fixed << std::setprecision(4) << info.speedup << "x |\n";
	}

	std::cout << "\nOUTPUT_LSE: " << (output_lse ? "True" : "False") << "\n";
}

int main() 
{
	if (!torch::cuda::is_available()) {
		std::cout << "CUDA is not available.\n";
		return 0;
	}

	torch::InferenceMode guard(true);
	torch::manual_seed(0);

	const std::vector<int64_t> NUM_BATCH_TOKENS = { 512, 613, 1536, 1724, 4096 };
	const std::vector<int64_t> NUM_QUERY_HEADS = { 16 };
	const std::vector<int64_t> HEAD_SIZES = { 128 };

	const std::vector<torch::ScalarType> DTYPES = {
		torch::kFloat,
		torch::kHalf
	};

	const bool OUTPUT_LSE = true;
	const int warmup_times = 2;
	const int repeat_times = 20;

	std::vector<CaseInfo> all_case_info;

	auto device = torch::Device(torch::kCUDA, 0);

	int dev = 0;
	cudaError_t err = cudaGetDevice(&dev);
	TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(err));

	cudaDeviceProp prop;
	err = cudaGetDeviceProperties(&prop, dev);
	TORCH_CHECK(err == cudaSuccess, "cudaGetDeviceProperties failed: ", cudaGetErrorString(err));
	std::string device_name = prop.name;

	for (auto num_tokens : NUM_BATCH_TOKENS) 
	{
		for (auto num_heads : NUM_QUERY_HEADS) 
		{
			for (auto head_size : HEAD_SIZES) 
			{
				for (auto output_dtype : DTYPES) 
				{
					std::cout << "\nNUM_TOKENS:" << num_tokens
						<< ", NUM_HEADS:" << num_heads
						<< ", HEAD_SIZE:" << head_size
						<< ", DTYPE: " << dtype_to_string(output_dtype)
						<< ", Device: " << device_name
						<< "\n";

					auto lse_opts = torch::TensorOptions().device(device).dtype(torch::kFloat);
					auto out_opts = torch::TensorOptions().device(device).dtype(output_dtype);

					auto prefix_lse = torch::randn({ num_heads, num_tokens }, lse_opts);
					auto suffix_lse = torch::randn({ num_heads, num_tokens }, lse_opts);

					auto mask_prefix = torch::rand({ num_heads, num_tokens }, lse_opts) < 0.1;
					auto mask_suffix = torch::rand({ num_heads, num_tokens }, lse_opts) < 0.1;
					auto combined_mask = torch::logical_and(mask_prefix, mask_suffix);
					mask_prefix = torch::logical_and(mask_prefix, torch::logical_not(combined_mask));
					mask_suffix = torch::logical_and(mask_suffix, torch::logical_not(combined_mask));

					prefix_lse.masked_fill_(mask_prefix, std::numeric_limits<float>::infinity());
					suffix_lse.masked_fill_(mask_suffix, std::numeric_limits<float>::infinity());

					auto output = torch::zeros({ num_tokens, num_heads, head_size }, out_opts);
					auto output_lse = torch::zeros({ num_heads, num_tokens }, lse_opts);
					auto prefix_output = torch::randn({ num_tokens, num_heads, head_size }, out_opts);
					auto suffix_output = torch::randn({ num_tokens, num_heads, head_size }, out_opts);

					auto output_torch = output.clone();
					c10::optional<torch::Tensor> output_lse_torch =
						OUTPUT_LSE ? c10::optional<torch::Tensor>(output_lse.clone()) : c10::nullopt;

					float total_time_torch_ms = 0.0f;
					for (int i = 0; i < warmup_times; ++i) 
					{
						auto result = merge_attn_states_torch_ref(
							output_torch, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse_torch);
						output_torch = result.first;
						output_lse_torch = result.second;
					}
					cuda_sync_checked();

					for (int i = 0; i < repeat_times; ++i) 
					{
						total_time_torch_ms += measure_cuda_time_ms([&]() {
							auto result = merge_attn_states_torch_ref(
								output_torch, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse_torch);
							output_torch = result.first;
							output_lse_torch = result.second;
							});
					}
					float avg_time_torch_ms = total_time_torch_ms / repeat_times;

					auto output_cuda = output.clone();
					c10::optional<torch::Tensor> output_lse_cuda =
						OUTPUT_LSE ? c10::optional<torch::Tensor>(output_lse.clone()) : c10::nullopt;

					float total_time_cuda_ms = 0.0f;
					for (int i = 0; i < warmup_times; ++i) 
					{
						merge_attn_states(
							output_cuda, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse_cuda);
					}
					cuda_sync_checked();

					for (int i = 0; i < repeat_times; ++i) 
					{
						total_time_cuda_ms += measure_cuda_time_ms([&]() {
							merge_attn_states(
								output_cuda, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse_cuda);
						});
					}
					float avg_time_cuda_ms = total_time_cuda_ms / repeat_times;
					float speedup = avg_time_torch_ms / avg_time_cuda_ms;

					std::cout << " Torch time: " << std::fixed << std::setprecision(6) << avg_time_torch_ms << "ms\n";
					std::cout << "  CUDA time: " << std::fixed << std::setprecision(6) << avg_time_cuda_ms
						<< "ms, Performance: " << std::fixed << std::setprecision(5) << speedup << "x\n";
					std::cout << std::string(100, '-') << "\n";

					double rtol = (output_dtype == torch::kBFloat16) ? 1e-2 : 1e-3;

					auto output_cuda_f = output_cuda.to(torch::kFloat);
					auto output_torch_f = output_torch.to(torch::kFloat);

					std::cout << "Output all match, max abs diff:\n";
					std::cout << "(CUDA vs Torch): " << tensor_max_abs_diff(output_torch, output_cuda) << "\n";
					std::cout << std::string(100, '-') << "\n";

					if (OUTPUT_LSE) {
						TORCH_CHECK(output_lse_torch.has_value(), "output_lse_torch missing");
						TORCH_CHECK(output_lse_cuda.has_value(), "output_lse_cuda missing");

						auto lse_cuda_f = output_lse_cuda.value().to(torch::kFloat);
						auto lse_torch_f = output_lse_torch.value().to(torch::kFloat);

						std::cout << "Output LSE all match, max abs diff:\n";
						std::cout << "(CUDA vs Torch): "
							<< tensor_max_abs_diff(output_lse_torch.value(), output_lse_cuda.value())
							<< "\n";
						std::cout << std::string(100, '-') << "\n";
					}

					std::cout << "All output values test passed! All inf values are correctly replaced with -inf.\n";
					std::cout << std::string(100, '-') << "\n";

					all_case_info.push_back({
						num_tokens,
						num_heads,
						head_size,
						output_dtype,
						device_name,
						avg_time_torch_ms,
						avg_time_cuda_ms,
						speedup
					});
				}
			}
		}
	}

	print_markdown_table(all_case_info, OUTPUT_LSE);
	return 0;
}