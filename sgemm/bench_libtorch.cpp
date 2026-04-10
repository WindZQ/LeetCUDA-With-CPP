#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "sgemm.cuh"
#include "sgemm_async.cuh"
#include "sgemm_cublas.cuh"
#include "sgemm_wmma_tf32_stage.cuh"

static void cuda_check(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at " << file << ":" << line
			<< ", code=" << err
			<< ", msg=" << cudaGetErrorString(err) << std::endl;
		std::exit(EXIT_FAILURE);
	}
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)


double MAX_TFLOPS = -1.0;

void run_benchmark(
	std::function<void(int, bool, int)> perf_call,
	torch::Tensor a,
	torch::Tensor b,
	const std::string& tag,
	torch::Tensor& out,
	int stages = -1,
	bool enable_swizzle = false,
	int warmup = 2,
	int iters = 20,
	bool show_all = false)
{
	int M = a.size(0);
	int K = a.size(1);
	int N = b.size(1);

	if (M > 1024 || K >= 1024 || N > 1024) iters = 10;

	int swizzle_stride = 1;
	bool swizzle = enable_swizzle;
	if (swizzle) {
		swizzle_stride = static_cast<int>((static_cast<int>(N / 8) / 256) * 256);
		swizzle_stride = (swizzle_stride >= 256) ? swizzle_stride : 1;
		swizzle = (swizzle_stride >= 256);
	}

	out.zero_();

	for (int i = 0; i < warmup; ++i) 
	{
		perf_call(stages, swizzle, swizzle_stride);
	}

	CUDA_CHECK(cudaDeviceSynchronize());
	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < iters; ++i) 
	{
		perf_call(stages, swizzle, swizzle_stride);
	}

	CUDA_CHECK(cudaDeviceSynchronize());
	auto end = std::chrono::high_resolution_clock::now();
	double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
	double mean_time = total_ms / iters;

	auto flat = out.flatten().to(torch::kCPU);
	float v0 = flat[0].item<float>();
	float v1 = flat[1].item<float>();
	std::string out_val = "[" + std::to_string(std::round(v0 * 1e8) / 1e8) + ", " +
		std::to_string(std::round(v1 * 1e8) / 1e8) + "]";

	double tflops = (2.0 * M * N * K) * 1e-9 / mean_time;

	std::string sw_str = (swizzle_stride == 1) ? "NOOP" : std::to_string(swizzle_stride);
	if (tflops > MAX_TFLOPS) {
		double improve = (MAX_TFLOPS > 0) ? ((tflops - MAX_TFLOPS) / MAX_TFLOPS) * 100.0 : 0.0;
		MAX_TFLOPS = tflops;
		std::cout << std::right << std::setw(35) << ("out_" + tag) << ": "
			<< std::left << std::setw(20) << out_val
			<< " time:" << std::fixed << std::setprecision(3) << mean_time << "ms, "
			<< "swizzle: " << std::left << std::setw(4) << sw_str
			<< " TFLOPS: " << std::fixed << std::setprecision(2) << tflops
			<< "(+" << std::fixed << std::setprecision(2) << improve << "%)" << std::endl;
	} else {
		std::cout << std::right << std::setw(35) << ("out_" + tag) << ": "
			<< std::left << std::setw(20) << out_val
			<< " time:" << std::fixed << std::setprecision(3) << mean_time << "ms, "
			<< "swizzle: " << std::left << std::setw(4) << sw_str
			<< " TFLOPS: " << std::fixed << std::setprecision(2) << tflops << std::endl;
	}

	if (show_all) std::cout << out << std::endl;
}

int main() 
{
	if (!torch::cuda::is_available()) {
		std::cerr << "CUDA is not available." << std::endl;
		return -1;
	}

	torch::NoGradGuard no_grad;

	const std::vector<int> Ms = { 4096, 8192, 16384 };
	const std::vector<int> Ns = { 4096, 8192, 16384 };
	const std::vector<int> Ks = { 2048, 4096, 8192 };

	const int MAX_M = 16384, MAX_N = 16384, MAX_K = 8192;
	auto A = torch::randn({ MAX_M, MAX_K }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
	auto B = torch::randn({ MAX_K, MAX_N }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
	auto C = torch::randn({ MAX_M, MAX_N }, torch::dtype(torch::kFloat32).device(torch::kCUDA));

	for (int M : Ms)
	{
		for (int N : Ns) 
		{
			for (int K : Ks) 
			{
				MAX_TFLOPS = -1.0;
				std::cout << std::string(130, '-') << std::endl;
				std::cout << std::string(55, ' ') << "M=" << M << ", N=" << N << ", K=" << K << std::endl;

				auto a = A.slice(0, 0, M).slice(1, 0, K).contiguous();
				auto b = B.slice(0, 0, K).slice(1, 0, N).contiguous();
				auto c = C.slice(0, 0, M).slice(1, 0, N).contiguous();

				float* pa = a.data_ptr<float>();
				float* pb = b.data_ptr<float>();
				float* pc = c.data_ptr<float>();

				run_benchmark([&](int, bool, int) { sgemm_t_8x8_sliced_k_f32x4(pa, pb, pc, M, N, K); }, a, b, "f32x4(t8x8sk)", c);
				run_benchmark([&](int, bool, int) { sgemm_t_8x8_sliced_k_f32x4_bcf(pa, pb, pc, M, N, K); }, a, b, "f32x4(t8x8bcf)", c);
				run_benchmark([&](int, bool, int) { sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf(pa, pb, pc, M, N, K); }, a, b, "f32x4(t8x8dbuf)", c);
				run_benchmark([&](int, bool, int) { sgemm_cublas(pa, pb, pc, M, N, K); }, a, b, "f32(cublas)", c);
				run_benchmark([&](int, bool, int) { torch::matmul_out(c, a, b); }, a, b, "f32_th", c);

				std::cout << std::string(62, '-') << "WMMA" << std::string(64, '-') << std::endl;

				run_benchmark([&](int s, bool sw, int sws) { sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages(pa, pb, pc, M, N, K, s, sw, sws); },
					a, b, "tf32(mma2x4+warp2x4+stage3)", c, 3, false);
				run_benchmark([&](int s, bool sw, int sws) { sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages(pa, pb, pc, M, N, K, s, sw, sws); },
					a, b, "tf32(mma2x4+warp2x4+stage2)", c, 2, false);

				run_benchmark([&](int s, bool sw, int sws) { sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem(pa, pb, pc, M, N, K, s, sw, sws); },
					a, b, "tf32(mma2x4+...+stage3+dsmem)", c, 3, false);
				run_benchmark([&](int s, bool sw, int sws) { sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem(pa, pb, pc, M, N, K, s, sw, sws); },
					a, b, "tf32(mma2x4+...+stage2+dsmem)", c, 2, false);

				run_benchmark([&](int s, bool sw, int sws) { sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages(pa, pb, pc, M, N, K, s, sw, sws); },
					a, b, "tf32(mma2x4+...+stage3+swizzle)", c, 3, true);
				run_benchmark([&](int s, bool sw, int sws) { sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages(pa, pb, pc, M, N, K, s, sw, sws); },
					a, b, "tf32(mma2x4+...+stage2+swizzle)", c, 2, true);

				run_benchmark([&](int s, bool sw, int sws) { sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem(pa, pb, pc, M, N, K, s, sw, sws); },
					a, b, "tf32(...+stage3+dsmem+swizzle)", c, 3, true);
				run_benchmark([&](int s, bool sw, int sws) { sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem(pa, pb, pc, M, N, K, s, sw, sws); },
					a, b, "tf32(...+stage2+dsmem+swizzle)", c, 2, true);

				run_benchmark([&](int, bool, int) { sgemm_cublas_tf32(pa, pb, pc, M, N, K); }, a, b, "tf32(cublas+tf32)", c);

				std::cout << std::string(130, '-') << std::endl;
			}
		}
	}

	return 0;
}