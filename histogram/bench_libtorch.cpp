#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

#include "histogram.cuh"

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

void print_hist(const char* tag, const torch::Tensor& y_cuda) 
{
	auto y = y_cuda.to(torch::kCPU).contiguous();
	const int* p = y.data_ptr<int>();
	int64_t n = y.numel();

	for (int64_t i = 0; i < n; ++i) 
	{
		std::cout << tag << " " << i << ": " << p[i] << "\n";
	}
}

int main()
{
	if (!torch::cuda::is_available()) {
		std::cerr << "CUDA not available\n";
		return 1;
	}

	auto opt_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
	auto base = torch::arange(0, 10, opt_cpu);        
	auto a_cpu = base.repeat({ 1000 }).contiguous();    
	auto a = a_cpu.to(torch::kCUDA).contiguous();

	const int N = (int)a.numel();
	auto opt_cuda = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

	auto h_i32 = torch::zeros({ 10 }, opt_cuda).contiguous();
	cudaStream_t s = at::cuda::getDefaultCUDAStream().stream();
	histogram_i32(a.data_ptr<int>(), h_i32.data_ptr<int>(), N, s);
	CHECK(cudaGetLastError());
	CHECK(cudaStreamSynchronize(s));

	std::cout << std::string(80, '-') << std::endl;
	print_hist("h_i32  ", h_i32);

	auto h_i32x4 = torch::zeros({ 10 }, opt_cuda).contiguous();
	histogram_i32x4(a.data_ptr<int>(),  h_i32x4.data_ptr<int>(), N, s);
	CHECK(cudaGetLastError());
	CHECK(cudaStreamSynchronize(s));

	std::cout << std::string(80, '-') << std::endl;
	print_hist("h_i32x4", h_i32x4);
	std::cout << std::string(80, '-') << std::endl;

	return 0;
}