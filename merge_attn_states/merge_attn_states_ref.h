#ifndef MERGE_ATTN_STATES_REF_H
#define MERGE_ATTN_STATES_REF_H

#include <optional>
#include <torch/torch.h>

std::pair<torch::Tensor, c10::optional<torch::Tensor>> merge_attn_states_torch_ref(
	const torch::Tensor& output,        
	const torch::Tensor& prefix_output,  
	const torch::Tensor& prefix_lse,     
	const torch::Tensor& suffix_output,  
	const torch::Tensor& suffix_lse,     
	const c10::optional<torch::Tensor>& output_lse = c10::nullopt
);

#endif