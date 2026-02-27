#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cassert>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
using namespace cute;

#include "mat_transpose_cute.cuh"

#define UNIT_BLK_SIZE 16

#define CUDA_CHECK(call)                                                       \
do {                                                                           \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      /* Optionally, you could also call cudaDeviceReset here */               \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
 } while (0)

constexpr bool is_pow2_constexpr(int x) { return x > 0 && ((x & (x - 1)) == 0); }
constexpr int log2_pow2_constexpr(int x) { return (x <= 1) ? 0 : 1 + log2_pow2_constexpr(x >> 1); }

__host__ __device__ inline bool is_aligned_128(const void* ptr)
{
    return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

template <typename T, int BLK_M, int BLK_N, typename ThreadLayoutA, typename ThreadLayoutB>
__global__ void mat_transpose_cute_reg_kernel(const T* pA, T* pB, int M, int N, ThreadLayoutA tA, ThreadLayoutB tB)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;

    auto mA = make_tensor(make_gmem_ptr(pA),
        make_layout(make_shape(M, N), GenRowMajor{})); // (M, N)
    auto mB = make_tensor(make_gmem_ptr(pB),
        make_layout(make_shape(N, M), GenRowMajor{})); // (N, M)

    auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(bx, by)); // (BM, BN)
    auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}), make_coord(by, bx)); // (BN, BM)
    auto cA = local_tile(make_identity_tensor(mA.shape()),
        make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
        make_coord(bx, by)); // (BM, BN)

    Tensor tAgA = local_partition(gA, tA, tx);
    Tensor tBgB = local_partition(gB, tB, tx);
    Tensor tAcA = local_partition(cA, tA, tx);

    Tensor tApA = make_tensor<bool>(tAcA.shape(), tAcA.stride());
    CUTE_UNROLL
        for (int i = 0; i < size<0>(tApA); i++) 
        {
            CUTE_UNROLL
                for (int j = 0; j < size<1>(tApA); j++) 
                {
                    tApA(i, j) = get<0>(tAcA(i, j)) < M && get<1>(tAcA(i, j)) < N;
                }
        }
    copy_if(tApA, tAgA, tBgB);
}

template <typename T, int BLK_M, int BLK_N, typename ThreadLayoutA, typename ThreadLayoutB, typename SmemLayoutA, typename SmemLayoutB>
__global__ void mat_transpose_cute_smem_kernel(const T* pA, T* pB, int M, int N,
	ThreadLayoutA tA, ThreadLayoutB tB,
	SmemLayoutA sA_layout, SmemLayoutB sB_layout) 
{
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;

    auto mA = make_tensor(make_gmem_ptr(pA),
        make_layout(make_shape(M, N), GenRowMajor{})); // (M, N)
    auto mB = make_tensor(make_gmem_ptr(pB),
        make_layout(make_shape(N, M), GenRowMajor{})); // (N, M)

    auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(bx, by)); // (BM, BN)
    auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}), make_coord(by, bx)); // (BN, BM)
    auto cA = local_tile(make_identity_tensor(mA.shape()),
        make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
        make_coord(bx, by)); // (BM, BN)
    auto cB = local_tile(make_identity_tensor(mB.shape()),
        make_shape(Int<BLK_N>{}, Int<BLK_M>{}),
        make_coord(by, bx)); // (BN, BM)

    __shared__ T smem[BLK_M * BLK_N];
    auto sA = make_tensor(make_smem_ptr(smem), sA_layout); // (BM, BN)
    auto sB = make_tensor(make_smem_ptr(smem), sB_layout); // (BN, BM)

    Tensor tAgA = local_partition(gA, tA, tx);
    Tensor tBgB = local_partition(gB, tB, tx);
    Tensor tAsA = local_partition(sA, tA, tx);
    Tensor tBsB = local_partition(sB, tB, tx);
    Tensor tAcA = local_partition(cA, tA, tx);
    Tensor tBcB = local_partition(cB, tB, tx);

    Tensor tApA = make_tensor<bool>(tAcA.shape(), tAcA.stride());
    Tensor tBpB = make_tensor<bool>(tBcB.shape(), tBcB.stride());

    CUTE_UNROLL
        for (int i = 0; i < size<0>(tApA); i++) 
        {
            CUTE_UNROLL
                for (int j = 0; j < size<1>(tApA); j++) 
                {
                    tApA(i, j) = get<0>(tAcA(i, j)) < M && get<1>(tAcA(i, j)) < N;
                }
        }
    CUTE_UNROLL
        for (int i = 0; i < size<0>(tBpB); i++) 
        {
            CUTE_UNROLL
                for (int j = 0; j < size<1>(tBpB); j++) 
                {
                    tBpB(i, j) = get<0>(tBcB(i, j)) < N && get<1>(tBcB(i, j)) < M;
                }
        }

    copy_if(tApA, tAgA, tAsA);
    __syncthreads();
    copy_if(tBpB, tBsB, tBgB);
}

template <typename T, int BLK_M, int BLK_N, typename TiledCopyA, typename TiledCopyB, typename SmemLayoutA, typename SmemLayoutB>
__global__ void mat_transpose_cute_smem_vectorized_kernel(const T* pA, T* pB, int M, int N, 
	TiledCopyA copy_a, TiledCopyB copy_b,
    SmemLayoutA sA_layout, SmemLayoutB sB_layout)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;

    auto mA = make_tensor(make_gmem_ptr(pA),
        make_layout(make_shape(M, N), GenRowMajor{})); // (M, N)
    auto mB = make_tensor(make_gmem_ptr(pB),
        make_layout(make_shape(N, M), GenRowMajor{})); // (N, M)

    auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(bx, by)); // (BM, BN)
    auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}), make_coord(by, bx)); // (BN, BM)

    __shared__ T smem[BLK_M * BLK_N];
    auto sA = make_tensor(make_smem_ptr(smem), sA_layout); // (BM, BN)
    auto sB = make_tensor(make_smem_ptr(smem), sB_layout); // (BN, BM)

    auto thr_copy_a = copy_a.get_slice(tx);
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tAsA = thr_copy_a.partition_D(sA);

    auto thr_copy_b = copy_b.get_slice(tx);
    Tensor tBsB = thr_copy_b.partition_S(sB);
    Tensor tBgB = thr_copy_b.partition_D(gB);

    copy(copy_a, tAgA, tAsA);
    __syncthreads();
    copy(copy_b, tBsB, tBgB);
}

template <typename T, int BLK_M, int BLK_N, typename TiledCopyA, typename TiledCopyTrans, typename TiledCopyB, typename SmemLayoutB>
__global__ void mat_transpose_cute_smem_vectorized_optimized_kernel(const T* pA, T* pB, int M, int N,
	TiledCopyA copy_a, TiledCopyTrans copy_trans,
	TiledCopyB copy_b, SmemLayoutB sB_layout) 
{
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;

    auto mA = make_tensor(make_gmem_ptr(pA),
        make_layout(make_shape(M, N), GenRowMajor{})); // (M, N)
    auto mB = make_tensor(make_gmem_ptr(pB),
        make_layout(make_shape(N, M), GenRowMajor{})); // (N, M)

    auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(bx, by)); // (BM, BN)
    auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_M>{}), make_coord(by, bx)); // (BN, BM)

    __shared__ T smem[BLK_M * BLK_N];
    auto sB = make_tensor(make_smem_ptr(smem), sB_layout); // (BN, BM)

    auto thr_copy_a = copy_a.get_slice(tx);
    Tensor tAgA = thr_copy_a.partition_S(gA);
    auto tAsA = make_tensor_like(tAgA);
    Tensor tAsA_view = thr_copy_a.retile_D(tAsA);
    copy(copy_a, tAgA, tAsA_view);

    auto thr_copy_trans = copy_trans.get_slice(tx);
    auto tAsB = thr_copy_trans.retile_S(tAsA);
    auto tBsB_trans = thr_copy_trans.partition_D(sB);
    copy(copy_trans, tAsB, tBsB_trans);

    auto thr_copy_b = copy_b.get_slice(tx);
    Tensor tBsB = thr_copy_b.partition_S(sB);
    Tensor tBgB = thr_copy_b.partition_D(gB);

    copy(copy_b, tBsB, tBgB);
}

void mat_transpose_cute_row2col_reg(const float* dA, float* dB, int M, int N, cudaStream_t stream) 
{
	const int BM = UNIT_BLK_SIZE;
	const int BN = UNIT_BLK_SIZE;

	auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenColMajor{});
	auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{});
	static_assert(size(tA) == size(tB));

	dim3 block(size(tA));
	dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

	mat_transpose_cute_reg_kernel<float, BM, BN, decltype(tA), decltype(tB)> 
		<< <grid, block, 0, stream >> > (dA, dB, M, N, tA, tB);
}

void mat_transpose_cute_col2row_reg(const float* dA, float* dB, int M, int N, cudaStream_t stream) 
{
	const int BM = UNIT_BLK_SIZE;
	const int BN = UNIT_BLK_SIZE;

	auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
	auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});
	static_assert(size(tA) == size(tB));

	dim3 block(size(tA));
	dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

	mat_transpose_cute_reg_kernel<float, BM, BN, decltype(tA), decltype(tB)>
		<< <grid, block, 0, stream >> > (dA, dB, M, N, tA, tB);
}

void mat_transpose_cute_col_smem(const float* dA, float* dB, int M, int N, cudaStream_t stream) 
{
	const int BM = UNIT_BLK_SIZE;
	const int BN = UNIT_BLK_SIZE;

	auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenColMajor{});
	auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});
	auto sA_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
	auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});
	static_assert(size(tA) == size(tB));

	dim3 block(size(tA));
	dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

	mat_transpose_cute_smem_kernel<float, BM, BN, decltype(tA), decltype(tB),
		decltype(sA_layout), decltype(sB_layout)>
		<< <grid, block, 0, stream >> > (dA, dB, M, N, tA, tB, sA_layout, sB_layout);
}

void mat_transpose_cute_row_smem(const float* dA, float* dB, int M, int N, cudaStream_t stream) 
{
	const int BM = UNIT_BLK_SIZE;
	const int BN = UNIT_BLK_SIZE;

	auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
	auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{});
	auto sA_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
	auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});
	static_assert(size(tA) == size(tB));

	dim3 block(size(tA));
	dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

	mat_transpose_cute_smem_kernel<float, BM, BN, decltype(tA), decltype(tB),
		decltype(sA_layout), decltype(sB_layout)>
		<< <grid, block, 0, stream >> > (dA, dB, M, N, tA, tB, sA_layout, sB_layout);
}

void mat_transpose_cute_col_smem_swizzled(const float* dA, float* dB, int M, int N, cudaStream_t stream) 
{
	const int BM = UNIT_BLK_SIZE;
	const int BN = UNIT_BLK_SIZE;
	static_assert(is_pow2_constexpr(BM), "BM must be power of 2 for swizzle");
	const int S = log2_pow2_constexpr(BM);

	auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenColMajor{});
	auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});

	auto swizzle_func = Swizzle<S, 0, S>{};
	auto sA_layout = composition(swizzle_func,
		make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{}));
	auto sB_layout = composition(swizzle_func,
		make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{}));
	static_assert(size(tA) == size(tB));

	dim3 block(size(tA));
	dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

	mat_transpose_cute_smem_kernel<float, BM, BN, decltype(tA), decltype(tB),
		decltype(sA_layout), decltype(sB_layout)>
		<< <grid, block, 0, stream >> > (dA, dB, M, N, tA, tB, sA_layout, sB_layout);
}

void mat_transpose_cute_row_smem_swizzled(const float* dA, float* dB, int M, int N, cudaStream_t stream) 
{
	const int BM = UNIT_BLK_SIZE;
	const int BN = UNIT_BLK_SIZE;
	static_assert(is_pow2_constexpr(BM), "BM must be power of 2 for swizzle");
	const int S = log2_pow2_constexpr(BM);

	auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
	auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{});

	auto swizzle_func = Swizzle<S, 0, S>{};
	auto sA_layout = composition(swizzle_func,
		make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{}));
	auto sB_layout = composition(swizzle_func,
		make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{}));
	static_assert(size(tA) == size(tB));

	dim3 block(size(tA));
	dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

	mat_transpose_cute_smem_kernel<float, BM, BN, decltype(tA), decltype(tB),
		decltype(sA_layout), decltype(sB_layout)>
		<< <grid, block, 0, stream >> > (dA, dB, M, N, tA, tB, sA_layout, sB_layout);
}

void mat_transpose_cute_row_cvectorized(const float* dA, float* dB, int M, int N, cudaStream_t stream) 
{
	const int BM = UNIT_BLK_SIZE * 4;
	const int BN = UNIT_BLK_SIZE;

	// runtime sanity checks (no torch)
	if ((M % 4) || (N % 4)) return ;
	if (!is_aligned_128(dA) || !is_aligned_128(dB)) return ;

	auto tile_copy_a = make_tiled_copy(
		Copy_Atom<AutoVectorizingCopy, float>{},
		make_layout(make_shape(Int<BM / 4>{}, Int<BN>{}), GenRowMajor{}),
		make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));
	auto tile_copy_b = make_tiled_copy(
		Copy_Atom<AutoVectorizingCopy, float>{},
		make_layout(make_shape(Int<BN>{}, Int<BM / 4>{}), GenRowMajor{}),
		make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));

	auto sA_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
	auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});

	static_assert(size(tile_copy_a) == size(tile_copy_b));
	dim3 block(size(tile_copy_a));
	dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

	mat_transpose_cute_smem_vectorized_kernel<float, BM, BN,
		decltype(tile_copy_a), decltype(tile_copy_b),
		decltype(sA_layout), decltype(sB_layout)>
		<< <grid, block, 0, stream >> > (dA, dB, M, N, tile_copy_a, tile_copy_b, sA_layout, sB_layout);
}

void mat_transpose_cute_row_cvectorized_swizzled(const float* dA, float* dB, int M, int N, cudaStream_t stream)
{
	const int BM = UNIT_BLK_SIZE * 4;
	const int BN = UNIT_BLK_SIZE;
	static_assert(is_pow2_constexpr(BN), "BN must be power of 2 for swizzle");
	const int S = log2_pow2_constexpr(BN);

	if ((M % 4) || (N % 4)) return ;
	if (!is_aligned_128(dA) || !is_aligned_128(dB)) return ;

	auto tile_copy_a = make_tiled_copy(
		Copy_Atom<AutoVectorizingCopy, float>{},
		make_layout(make_shape(Int<BM / 4>{}, Int<BN>{}), GenRowMajor{}),
		make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));
	auto tile_copy_b = make_tiled_copy(
		Copy_Atom<AutoVectorizingCopy, float>{},
		make_layout(make_shape(Int<BN>{}, Int<BM / 4>{}), GenRowMajor{}),
		make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));

	auto swizzle_func = Swizzle<S, 0, S>{};
	auto sA_layout = composition(swizzle_func,
		make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{}));
	auto sB_layout = composition(swizzle_func,
		make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{}));

	static_assert(size(tile_copy_a) == size(tile_copy_b));
	dim3 block(size(tile_copy_a));
	dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

	mat_transpose_cute_smem_vectorized_kernel<float, BM, BN,
		decltype(tile_copy_a), decltype(tile_copy_b),
		decltype(sA_layout), decltype(sB_layout)>
		<< <grid, block, 0, stream >> > (dA, dB, M, N, tile_copy_a, tile_copy_b, sA_layout, sB_layout);
}

void mat_transpose_cute_row_rvectorized(const float* dA, float* dB, int M, int N, cudaStream_t stream)
{
	const int BM = UNIT_BLK_SIZE;
	const int BN = UNIT_BLK_SIZE * 4;

	if ((M % 4) || (N % 4)) return ;
	if (!is_aligned_128(dA) || !is_aligned_128(dB)) return ;

	auto tile_copy_a = make_tiled_copy(
		Copy_Atom<AutoVectorizingCopy, float>{},
		make_layout(make_shape(Int<BM>{}, Int<BN / 4>{}), GenRowMajor{}),
		make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));
	auto tile_copy_b = make_tiled_copy(
		Copy_Atom<AutoVectorizingCopy, float>{},
		make_layout(make_shape(Int<BN / 4>{}, Int<BM>{}), GenRowMajor{}),
		make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));

	auto sA_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
	auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});

	static_assert(size(tile_copy_a) == size(tile_copy_b));
	dim3 block(size(tile_copy_a));
	dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

	mat_transpose_cute_smem_vectorized_kernel<float, BM, BN,
		decltype(tile_copy_a), decltype(tile_copy_b),
		decltype(sA_layout), decltype(sB_layout)>
		<< <grid, block, 0, stream >> > (dA, dB, M, N, tile_copy_a, tile_copy_b, sA_layout, sB_layout);
}

void mat_transpose_cute_row_rvectorized_swizzled(const float* dA, float* dB, int M, int N, cudaStream_t stream) 
{
	const int BM = UNIT_BLK_SIZE;
	const int BN = UNIT_BLK_SIZE * 4;
	static_assert(is_pow2_constexpr(BM), "BM must be power of 2 for swizzle");
	const int S = log2_pow2_constexpr(BM);

	if ((M % 4) || (N % 4)) return ;
	if (!is_aligned_128(dA) || !is_aligned_128(dB)) return ;

	auto tile_copy_a = make_tiled_copy(
		Copy_Atom<AutoVectorizingCopy, float>{},
		make_layout(make_shape(Int<BM>{}, Int<BN / 4>{}), GenRowMajor{}),
		make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));
	auto tile_copy_b = make_tiled_copy(
		Copy_Atom<AutoVectorizingCopy, float>{},
		make_layout(make_shape(Int<BN / 4>{}, Int<BM>{}), GenRowMajor{}),
		make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));

	auto swizzle_func = Swizzle<S, 0, S>{};
	auto sA_layout = composition(swizzle_func,
		make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{}));
	auto sB_layout = composition(swizzle_func,
		make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{}));

	static_assert(size(tile_copy_a) == size(tile_copy_b));
	dim3 block(size(tile_copy_a));
	dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

	mat_transpose_cute_smem_vectorized_kernel<float, BM, BN,
		decltype(tile_copy_a), decltype(tile_copy_b),
		decltype(sA_layout), decltype(sB_layout)>
		<< <grid, block, 0, stream >> > (dA, dB, M, N, tile_copy_a, tile_copy_b, sA_layout, sB_layout);
}

void mat_transpose_cute_row_rvectorized_swizzled_optimized(const float* dA, float* dB, int M, int N, cudaStream_t stream) 
{
	const int BM = 8;
	const int BN = 16 * 8;

	if ((M % 4) || (N % 4)) return ;
	if (!is_aligned_128(dA) || !is_aligned_128(dB)) return ;

	// 一次性加载 8*16 大小的矩阵
	auto tile_copy_a = make_tiled_copy(
		Copy_Atom<AutoVectorizingCopy, float>{},
		make_layout(make_shape(Int<BM>{}, make_shape(Int<4>{}, Int<BN / 16>{})),
			make_stride(Int<4>{}, make_stride(Int<1>{}, Int<32>{}))),
		make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));

	// 转换数据
	auto tile_copy_trans = make_tiled_copy(
		Copy_Atom<AutoVectorizingCopy, float>{},
		make_layout(make_shape(make_shape(Int<4>{}, Int<BN / 16>{}), Int<BM>{}),
			make_stride(make_stride(Int<1>{}, Int<32>{}), Int<4>{})),
		make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));

	// 一次性存储 16*8 大小的矩阵
	auto tile_copy_b = make_tiled_copy(
		Copy_Atom<AutoVectorizingCopy, float>{},
		make_layout(make_shape(Int<BN>{}, Int<BM / 4>{}), GenRowMajor{}),
		make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));

	auto swizzle_func = Swizzle<2, 3, 2>{};
	auto sB_layout = composition(swizzle_func,
		make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{}));

	static_assert(size(tile_copy_a) == size(tile_copy_b));
	dim3 block(size(tile_copy_a));
	dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

	mat_transpose_cute_smem_vectorized_optimized_kernel<
		float, BM, BN,
		decltype(tile_copy_a), decltype(tile_copy_trans),
		decltype(tile_copy_b), decltype(sB_layout)>
		<< <grid, block, 0, stream >> > (dA, dB, M, N, tile_copy_a, tile_copy_trans, tile_copy_b, sB_layout);
}