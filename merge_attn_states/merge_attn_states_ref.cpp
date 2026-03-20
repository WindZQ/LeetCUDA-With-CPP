#include <limits>

#include "merge_attn_states_ref.h"

std::pair<torch::Tensor, c10::optional<torch::Tensor>> merge_attn_states_torch_ref(
	const torch::Tensor& output,
	const torch::Tensor& prefix_output,
	const torch::Tensor& prefix_lse,
	const torch::Tensor& suffix_output,
	const torch::Tensor& suffix_lse,
	const c10::optional<torch::Tensor>& output_lse)
{
	auto p_lse = prefix_lse.clone();
	auto s_lse = suffix_lse.clone();

	p_lse.masked_fill_(torch::isinf(p_lse), -std::numeric_limits<float>::infinity());
	s_lse.masked_fill_(torch::isinf(s_lse), -std::numeric_limits<float>::infinity());

	auto max_lse = torch::maximum(p_lse, s_lse);
	p_lse = p_lse - max_lse;
	s_lse = s_lse - max_lse;

	auto p_lse_exp = torch::exp(p_lse);
	auto s_lse_exp = torch::exp(s_lse);
	auto out_se = p_lse_exp + s_lse_exp;

	c10::optional<torch::Tensor> out_lse_opt = c10::nullopt;
	if (output_lse.has_value()) {
		out_lse_opt = torch::log(out_se) + max_lse;
	}

	auto p_scale = (p_lse_exp / out_se).transpose(0, 1).unsqueeze(2);
	auto s_scale = (s_lse_exp / out_se).transpose(0, 1).unsqueeze(2);

	auto out = prefix_output * p_scale.to(prefix_output.scalar_type()) +
		suffix_output * s_scale.to(suffix_output.scalar_type());

	return { out, out_lse_opt };
}