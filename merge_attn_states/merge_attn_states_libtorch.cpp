#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "merge_attn_states_libtorch.h"
#include "merge_attn_states.cuh"

int get_scalar_type_id(torch::ScalarType dtype) 
{
	switch (dtype) 
	{
	case torch::kFloat:
		return SCALAR_FLOAT;
	case torch::kHalf:
		return SCALAR_HALF;
	case torch::kBFloat16:
		return SCALAR_BFLOAT16;
	default:
		TORCH_CHECK(false, "Unsupported dtype for output/prefix_output/suffix_output: ", dtype);
	}
}

void check_same_shape_3d(const torch::Tensor& a, const torch::Tensor& b, const char* aname, const char* bname) 
{
	TORCH_CHECK(a.dim() == 3, aname, " must be 3D");
	TORCH_CHECK(b.dim() == 3, bname, " must be 3D");
	TORCH_CHECK(a.sizes() == b.sizes(), aname, " and ", bname, " must have same shape");
}

void check_same_shape_2d(const torch::Tensor& a, const torch::Tensor& b, const char* aname, const char* bname) 
{
	TORCH_CHECK(a.dim() == 2, aname, " must be 2D");
	TORCH_CHECK(b.dim() == 2, bname, " must be 2D");
	TORCH_CHECK(a.sizes() == b.sizes(), aname, " and ", bname, " must have same shape");
}

void merge_attn_states(
	torch::Tensor& output,
	const torch::Tensor& prefix_output,
	const torch::Tensor& prefix_lse,
	const torch::Tensor& suffix_output,
	const torch::Tensor& suffix_lse,
	const c10::optional<torch::Tensor>& output_lse)
{
	TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
	TORCH_CHECK(prefix_output.is_cuda(), "prefix_output must be a CUDA tensor");
	TORCH_CHECK(prefix_lse.is_cuda(), "prefix_lse must be a CUDA tensor");
	TORCH_CHECK(suffix_output.is_cuda(), "suffix_output must be a CUDA tensor");
	TORCH_CHECK(suffix_lse.is_cuda(), "suffix_lse must be a CUDA tensor");

	check_same_shape_3d(output, prefix_output, "output", "prefix_output");
	check_same_shape_3d(output, suffix_output, "output", "suffix_output");
	check_same_shape_2d(prefix_lse, suffix_lse, "prefix_lse", "suffix_lse");

	TORCH_CHECK(output.scalar_type() == prefix_output.scalar_type(),
		"output and prefix_output must have same dtype");
	TORCH_CHECK(output.scalar_type() == suffix_output.scalar_type(),
		"output and suffix_output must have same dtype");

	TORCH_CHECK(
		output.scalar_type() == torch::kFloat ||
		output.scalar_type() == torch::kHalf ||
		output.scalar_type() == torch::kBFloat16,
		"output/prefix_output/suffix_output dtype must be float/half/bfloat16");

	TORCH_CHECK(prefix_lse.scalar_type() == torch::kFloat,
		"prefix_lse must be float32");
	TORCH_CHECK(suffix_lse.scalar_type() == torch::kFloat,
		"suffix_lse must be float32");

	TORCH_CHECK(output.dim() == 3, "output must be [num_tokens, num_heads, head_size]");
	TORCH_CHECK(prefix_lse.dim() == 2, "prefix_lse must be [num_heads, num_tokens]");
	TORCH_CHECK(suffix_lse.dim() == 2, "suffix_lse must be [num_heads, num_tokens]");

	const auto num_tokens = static_cast<int>(output.size(0));
	const auto num_heads = static_cast<int>(output.size(1));
	const auto head_size = static_cast<int>(output.size(2));

	TORCH_CHECK(prefix_lse.size(0) == static_cast<int64_t>(num_heads) &&
		prefix_lse.size(1) == static_cast<int64_t>(num_tokens),
		"prefix_lse must have shape [num_heads, num_tokens]");
	TORCH_CHECK(suffix_lse.size(0) == static_cast<int64_t>(num_heads) &&
		suffix_lse.size(1) == static_cast<int64_t>(num_tokens),
		"suffix_lse must have shape [num_heads, num_tokens]");

	TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
	TORCH_CHECK(prefix_output.is_contiguous(), "prefix_output must be contiguous");
	TORCH_CHECK(suffix_output.is_contiguous(), "suffix_output must be contiguous");
	TORCH_CHECK(prefix_lse.is_contiguous(), "prefix_lse must be contiguous");
	TORCH_CHECK(suffix_lse.is_contiguous(), "suffix_lse must be contiguous");

	float* output_lse_ptr = nullptr;
	if (output_lse.has_value()) {
		TORCH_CHECK(output_lse.value().is_cuda(), "output_lse must be a CUDA tensor");
		TORCH_CHECK(output_lse.value().scalar_type() == torch::kFloat,
			"output_lse must be float32");
		TORCH_CHECK(output_lse.value().dim() == 2,
			"output_lse must be [num_heads, num_tokens]");
		TORCH_CHECK(output_lse.value().size(0) == static_cast<int64_t>(num_heads) &&
			output_lse.value().size(1) == static_cast<int64_t>(num_tokens),
			"output_lse must have shape [num_heads, num_tokens]");
		TORCH_CHECK(output_lse.value().is_contiguous(), "output_lse must be contiguous");
		output_lse_ptr = output_lse.value().data_ptr<float>();
	}

	TORCH_CHECK(output.device() == prefix_output.device(), "device mismatch");
	TORCH_CHECK(output.device() == suffix_output.device(), "device mismatch");
	TORCH_CHECK(output.device() == prefix_lse.device(), "device mismatch");
	TORCH_CHECK(output.device() == suffix_lse.device(), "device mismatch");
	if (output_lse.has_value()) {
		TORCH_CHECK(output.device() == output_lse.value().device(), "device mismatch");
	}

	c10::cuda::CUDAGuard device_guard(output.device());
	cudaStream_t stream = at::cuda::getDefaultCUDAStream(output.get_device());

	const int scalar_type = get_scalar_type_id(output.scalar_type());

	merge_attn_states(
		output.data_ptr(),
		output_lse_ptr,
		prefix_output.data_ptr(),
		prefix_lse.data_ptr<float>(),
		suffix_output.data_ptr(),
		suffix_lse.data_ptr<float>(),
		num_tokens,
		num_heads,
		head_size,
		scalar_type,
		stream);
}