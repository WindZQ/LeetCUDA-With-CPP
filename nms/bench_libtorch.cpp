#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "nms.cuh"

struct BenchmarkResult 
{
	std::vector<int> keep_indices;
	double mean_time_ms;
};

void generate_random_data(int nboxes,
	std::vector<float>& boxes,
	std::vector<float>& scores) 
{
	boxes.resize(nboxes * 4);
	scores.resize(nboxes);

	std::mt19937 rng(12345);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	for (int i = 0; i < nboxes; ++i) 
	{
		float a = dist(rng);
		float b = dist(rng);
		float c = dist(rng);
		float d = dist(rng);

		float x1 = std::min(a, c);
		float y1 = std::min(b, d);
		float x2 = std::max(a, c);
		float y2 = std::max(b, d);

		boxes[i * 4 + 0] = x1;
		boxes[i * 4 + 1] = y1;
		boxes[i * 4 + 2] = x2;
		boxes[i * 4 + 3] = y2;

		scores[i] = dist(rng);
	}
}

BenchmarkResult run_benchmark(
	const std::vector<float>& boxes,
	const std::vector<float>& scores,
	float threshold,
	const std::string& tag,
	int warmup = 10,
	int iters = 100,
	bool show_all = false) 
{
	int nboxes = static_cast<int>(scores.size());
	std::vector<int> keep_indices(nboxes);
	int keep_count = 0;

	for (int i = 0; i < warmup; ++i) 
	{
		keep_count = nms(boxes.data(), scores.data(), nboxes, threshold, keep_indices.data());
	}

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < iters; ++i) 
	{
		keep_count = nms(boxes.data(), scores.data(), nboxes, threshold, keep_indices.data());
	}

	auto end = std::chrono::high_resolution_clock::now();

	double total_time_ms =
		std::chrono::duration<double, std::milli>(end - start).count();
	double mean_time_ms = total_time_ms / iters;

	std::vector<int> out(keep_indices.begin(), keep_indices.begin() + keep_count);
	std::sort(out.begin(), out.end());

	int len_val = static_cast<int>(out.size());
	int begin_idx = std::max(0, len_val - 3);

	std::cout << tag << ": [";
	for (int i = begin_idx; i < len_val; ++i) 
	{
		std::cout << out[i];
		if (i + 1 < len_val) std::cout << ", ";
	}
	std::cout << "], len of keep: " << len_val
		<< ", time:" << mean_time_ms << "ms" << std::endl;

	if (show_all) {
		for (int i = 0; i < len_val; ++i) 
		{
			std::cout << out[i] << " ";
		}
		std::cout << std::endl;
	}

	return { out, mean_time_ms };
}

int main() 
{
	std::vector<int> nboxes_list = { 1024, 2048, 4096, 8192 };
	float threshold = 0.5f;

	for (int nboxes : nboxes_list)
	{
		std::cout << std::string(85, '-') << std::endl;
		std::cout << std::string(35, ' ') << "nboxes=" << nboxes << std::endl;

		std::vector<float> boxes;
		std::vector<float> scores;
		generate_random_data(nboxes, boxes, scores);

		run_benchmark(boxes, scores, threshold, "nms");

		std::cout << std::string(85, '-') << std::endl;
	}

	return 0;
}