#ifndef MERGE_ATTN_STATES_LIBTORCH_H
#define MERGE_ATTN_STATES_LIBTORCH_H

#include <optional>
#include <torch/torch.h>

void merge_attn_states(
	torch::Tensor& output,                        
	const torch::Tensor& prefix_output,          
	const torch::Tensor& prefix_lse,             
	const torch::Tensor& suffix_output,          
	const torch::Tensor& suffix_lse,             
	const c10::optional<torch::Tensor>& output_lse = c10::nullopt
);

#endif 
