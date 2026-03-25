#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "hgemv_cute.cuh"

using namespace cute;

template <const int warp_size = 32>
__device__ __forceinline__ half warp_reduce_sum_f16(half val) 
{
#pragma unroll
    for (int mask = warp_size >> 1; mask >= 1; mask >>= 1) 
    {
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    }

    return val;
}

template <typename T_, int NWarpPerBlock_>
struct HgemvConfig 
{
    using T = T_;
    static constexpr int NWarpPerBlock = NWarpPerBlock_;
    static constexpr int NumThreads = NWarpPerBlock * 32;

    static constexpr int BlockM = 16 * NWarpPerBlock;
    static constexpr int BlockN = 8;
    static constexpr int BlockK = 16;

    using MMA_Atom = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
    using TiledMMA = decltype(make_tiled_mma(
        MMA_Atom{},
        make_layout(Shape<Int<NWarpPerBlock>, _1, _1>{}, GenColMajor{})));

    static_assert(size(TiledMMA{}) == NumThreads && size(TiledMMA{}) <= 1024,
        "NumThreads must be <= 1024");
};

template <typename TiledCopy, int BlockM, int BlockK, int WARP_SIZE = 32>
__global__ void hgemv_f16_cute_kernel(half* Aptr, half* Bptr, half* Cptr, const int M, const int K) 
{
    int thrid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockid = blockIdx.x;

    int laneid = threadIdx.x % WARP_SIZE;
    int warpid = threadIdx.y;

    auto A = make_tensor(make_gmem_ptr(Aptr),
        make_layout(make_shape(M, K), make_stride(K, Int<1>{})));
    auto B = make_tensor(make_gmem_ptr(Bptr),
        make_layout(make_shape(M, K), make_stride(0, Int<1>{})));
    auto C = make_tensor(make_gmem_ptr(Cptr),
        make_layout(make_shape(M, 1), make_stride(Int<1>{}, 0)));

    auto ABPre = make_identity_tensor(shape(A));
    auto CPre = make_identity_tensor(shape(C));

    auto gA = local_tile(A, make_shape(Int<BlockM>{}, Int<BlockK>{}),
        make_coord(blockid, _));
    auto gB = local_tile(B, make_shape(Int<BlockM>{}, Int<BlockK>{}),
        make_coord(blockid, _));
    auto gC = local_tile(C, make_shape(Int<BlockM>{}, Int<1>{}),
        make_coord(blockid, 0));

    auto gABPre = local_tile(ABPre, make_shape(Int<BlockM>{}, Int<BlockK>{}),
        make_coord(blockid, _));
    auto gCPre = local_tile(CPre, make_shape(Int<BlockM>{}, Int<1>{}),
        make_coord(blockid, _));

    TiledCopy tiled_copy;
    auto thr_copy = tiled_copy.get_slice(thrid);

    auto tAgA = thr_copy.partition_S(gA);
    auto tBgB = thr_copy.partition_S(gB);
    auto rABPre = thr_copy.partition_S(gABPre);

    int num_tile_k = size<2>(gA);

    auto tArA = make_tensor_like(tAgA(_, _, _, 0));
    auto tBrB = make_tensor_like(tBgB(_, _, _, 0));

    auto sum = make_tensor_like(gC(0, _));
    clear(sum);

#pragma unroll
    for (int num_iter_k = 0; num_iter_k < num_tile_k; num_iter_k++) 
    {
        auto pre_ = rABPre(_, _, _, num_iter_k);
        auto pred = [&](auto... coords) {
            return cute::elem_less(pre_(0), shape(A));
        };

        clear(tArA);
        copy_if(tiled_copy, pred, tAgA(_, _, _, num_iter_k), tArA);
        clear(tBrB);
        copy_if(tiled_copy, pred, tBgB(_, _, _, num_iter_k), tBrB);

        sum(0) = __hadd(sum(0), __hmul(tArA(0), tBrB(0)));
    }

    sum(0) = warp_reduce_sum_f16<WARP_SIZE>(sum(0));

    auto store_pred = [&](auto... coords) {
        return cute::elem_less(gCPre(warpid), shape(C)) && laneid == 0;
    };
    copy_if(store_pred, sum, gC(warpid, _));
}

template <typename TiledCopy, int BlockM, int BlockK, int NumElemPerThread, int WARP_SIZE = 32>
__global__ void hgemv_f16x8_cute_kernel(half* Aptr, half* Bptr, half* Cptr, const int M, const int K) 
{
    int thrid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockid = blockIdx.x;

    int laneid = threadIdx.x % WARP_SIZE;
    int warpid = threadIdx.y;

    auto A = make_tensor(make_gmem_ptr(Aptr),
        make_layout(make_shape(M, K), make_stride(K, Int<1>{})));
    auto B = make_tensor(make_gmem_ptr(Bptr),
        make_layout(make_shape(M, K), make_stride(0, Int<1>{})));
    auto C = make_tensor(make_gmem_ptr(Cptr),
        make_layout(make_shape(M, 1), make_stride(Int<1>{}, 0)));

    auto ABPre = make_identity_tensor(shape(A));
    auto CPre = make_identity_tensor(shape(C));

    auto gA = local_tile(A, make_shape(Int<BlockM>{}, Int<BlockK>{}),
        make_coord(blockid, _));
    auto gB = local_tile(B, make_shape(Int<BlockM>{}, Int<BlockK>{}),
        make_coord(blockid, _));
    auto gC = local_tile(C, make_shape(Int<BlockM>{}, Int<1>{}),
        make_coord(blockid, 0));

    auto gABPre = local_tile(ABPre, make_shape(Int<BlockM>{}, Int<BlockK>{}),
        make_coord(blockid, _));
    auto gCPre = local_tile(CPre, make_shape(Int<BlockM>{}, Int<1>{}),
        make_coord(blockid, _));

    TiledCopy tiled_copy;
    auto thr_copy = tiled_copy.get_slice(thrid);

    auto tAgA = thr_copy.partition_S(gA);
    auto tBgB = thr_copy.partition_S(gB);
    auto rABPre = thr_copy.partition_S(gABPre);

    int num_tile_k = size<2>(gA);

    auto tArA = make_tensor_like(tAgA(_, _, _, 0));
    auto tBrB = make_tensor_like(tBgB(_, _, _, 0));

    auto sum = make_tensor_like(gC(0, _));
    clear(sum);

#pragma unroll
    for (int iter_k = 0; iter_k < num_tile_k; iter_k++) 
    {
        auto pre_ = rABPre(_, _, _, iter_k);
        auto pred = [&](auto... coords) {
            return cute::elem_less(pre_(NumElemPerThread - 1), shape(A));
        };

        clear(tArA);
        copy_if(tiled_copy, pred, tAgA(_, _, _, iter_k), tArA);
        clear(tBrB);
        copy_if(tiled_copy, pred, tBgB(_, _, _, iter_k), tBrB);

        auto tArA_half2 = recast<half2>(tArA);
        auto tBrB_half2 = recast<half2>(tBrB);

        half2 sum_half2 = __float2half2_rn(0.0f);

#pragma unroll
        for (int iter_elem = 0; iter_elem < size(tArA_half2); iter_elem++) 
        {
            sum_half2 = __hadd2(__hmul2(tArA_half2(iter_elem), tBrB_half2(iter_elem)),
                sum_half2);
        }

        sum(0) = __hadd(sum(0), __hadd(__low2half(sum_half2), __high2half(sum_half2)));
    }

    sum(0) = warp_reduce_sum_f16<WARP_SIZE>(sum(0));

    auto store_pred = [&](auto... coords) {
        return cute::elem_less(gCPre(warpid), shape(C)) && laneid == 0;
    };
    copy_if(store_pred, sum, gC(warpid, _));
}

// using tensor core
template <typename HgemvConfig_>
__global__ void hgemv_tensor_core_cute_kernel(typename HgemvConfig_::T* Aptr,
    typename HgemvConfig_::T* Bptr,
    typename HgemvConfig_::T* Cptr,
    const int M, const int K) 
{
    using T = typename HgemvConfig_::T;
    using TiledMMA = typename HgemvConfig_::TiledMMA;
    constexpr int BlockM = HgemvConfig_::BlockM;
    constexpr int BlockN = HgemvConfig_::BlockN;
    constexpr int BlockK = HgemvConfig_::BlockK;

    int thrid = threadIdx.x;
    int blockid = blockIdx.x;

    int warpid = threadIdx.x / 32;
    int laneid = threadIdx.x % 32;

    auto A = make_tensor(make_gmem_ptr(Aptr),
        make_layout(make_shape(M, K), make_stride(K, Int<1>{})));
    auto B = make_tensor(make_gmem_ptr(Bptr),
        make_layout(make_shape(M, K), make_stride(0, Int<1>{})));
    auto C = make_tensor(make_gmem_ptr(Cptr),
        make_layout(make_shape(M, 1), make_stride(Int<1>{}, 0)));

    auto ABPre = make_identity_tensor(shape(A));
    auto CPre = make_identity_tensor(shape(C));

    auto gA = local_tile(A, make_shape(Int<BlockM>{}, Int<BlockK>{}),
        make_coord(blockid, _));
    auto gB = local_tile(B, make_shape(Int<BlockN>{}, Int<BlockK>{}),
        make_coord(blockid, _));
    auto gC = local_tile(C, make_shape(Int<BlockM>{}, Int<1>{}),
        make_coord(blockid, 0));

    auto gABPre = local_tile(ABPre, make_shape(Int<BlockM>{}, Int<BlockK>{}),
        make_coord(blockid, _));
    auto gCPre = local_tile(CPre, make_shape(Int<BlockM>{}, Int<1>{}),
        make_coord(blockid, _));

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thrid);

    auto tAgA = thr_mma.partition_A(gA);
    auto tBgB = thr_mma.partition_B(gB);

    auto rAPre = thr_mma.partition_A(gABPre);
    auto rBPre = thr_mma.partition_B(gABPre);

    auto tArA = make_tensor_like(tAgA(_, _, _, 0));
    auto tBrB = make_tensor_like(tBgB(_, _, _, 0));

    auto tCrC =
        partition_fragment_C(tiled_mma, Shape<Int<BlockM>, Int<BlockN>>{});

    clear(tCrC);

    int num_tile_k = size<2>(gA);

#pragma unroll
    for (int itile = 0; itile < num_tile_k; itile++) 
    {
        auto pre_A = rAPre(_, _, _, itile);
        auto pre_B = rBPre(_, _, _, itile);

        auto pred_A = [&](auto... coords) 
        {
            return cute::elem_less(pre_A(coords...), shape(A));
        };
        auto pred_B = [&](auto... coords) 
        {
            return cute::elem_less(pre_B(coords...), shape(A));
        };

        clear(tArA);
        copy_if(pred_A, tAgA(_, _, _, itile), tArA);
        clear(tBrB);
        copy_if(pred_B, tBgB(_, _, _, itile), tBrB);

        gemm(tiled_mma, tArA, tBrB, tCrC);
    }

    int elem_index1 = warpid * 16 + laneid / 4;
    int elem_index2 = warpid * 16 + laneid / 4 + 8;

    auto sum = make_tensor_like(gC(0, _));

    sum(0) = tCrC(0);
    auto elem_pred1 = [&](auto... coords) 
    {
        return (laneid % 4 == 0) && cute::elem_less(gCPre(elem_index1), shape(C));
    };
    copy_if(elem_pred1, sum, gC(elem_index1, _));

    sum(0) = tCrC(2);
    auto elem_pred2 = [&](auto... coords) 
    {
        return (laneid % 4 == 0) && cute::elem_less(gCPre(elem_index2), shape(C));
    };
    copy_if(elem_pred2, sum, gC(elem_index2, _));
}

void hgemv_f16_cute(half * A, half * B, half * C, int M, int K) 
{
    if (!A || !B || !C) {
        fprintf(stderr, "hgemv_f16_cute: null pointer.\n");
        return;
    }

    constexpr int NumThreadPerRow = 32;
    constexpr int NumThreadPerBlock = 128;
    constexpr int NumRowPerBlock = NumThreadPerBlock / 32;

    using LoadType = uint16_t;
    constexpr int NumElemPerThread = sizeof(LoadType) / sizeof(half);

    using CopyAtom = Copy_Atom<UniversalCopy<LoadType>, half>;
    using TiledCopy = decltype(make_tiled_copy(
        CopyAtom{},
        make_layout(Shape<Int<NumRowPerBlock>, Int<NumThreadPerRow>>{},
            GenRowMajor{}),
        make_layout(Shape<_1, Int<NumElemPerThread>>{}, GenRowMajor{})));

    dim3 block(NumThreadPerRow, NumRowPerBlock);
    dim3 grid((M + NumRowPerBlock - 1) / NumRowPerBlock);

    hgemv_f16_cute_kernel<TiledCopy, NumRowPerBlock,
        NumThreadPerRow* NumElemPerThread>
        << <grid, block >> > (const_cast<half*>(A), const_cast<half*>(B), C, M, K);
}

void hgemv_f16x8_cute(half * A, half * B, half * C, int M, int K) 
{
    if (!A || !B || !C) {
        fprintf(stderr, "hgemv_f16x8_cute: null pointer.\n");
        return;
    }

    if (K % 8 != 0) {
        fprintf(stderr, "hgemv_f16x8_cute: K must be multiple of 8.\n");
        return;
    }

    constexpr int NumThreadPerRow = 32;
    constexpr int NumThreadPerBlock = 128;
    constexpr int NumRowPerBlock = NumThreadPerBlock / 32;

    using LoadType = uint4;   // 16 bytes = 8 half
    constexpr int NumElemPerThread = sizeof(LoadType) / sizeof(half);

    using CopyAtom = Copy_Atom<UniversalCopy<LoadType>, half>;
    using TiledCopy = decltype(make_tiled_copy(
        CopyAtom{},
        make_layout(Shape<Int<NumRowPerBlock>, Int<NumThreadPerRow>>{},
            GenRowMajor{}),
        make_layout(Shape<_1, Int<NumElemPerThread>>{}, GenRowMajor{})));

    dim3 block(NumThreadPerRow, NumRowPerBlock);
    dim3 grid((M + NumRowPerBlock - 1) / NumRowPerBlock);

    hgemv_f16x8_cute_kernel<TiledCopy, NumRowPerBlock,
        NumThreadPerRow* NumElemPerThread,
        NumElemPerThread>
        << <grid, block >> > (const_cast<half*>(A), const_cast<half*>(B), C, M, K);
}

void hgemv_tensor_core_cute(half * A, half * B, half * C, int M, int K) 
{
    if (!A || !B || !C) {
        fprintf(stderr, "hgemv_tensor_core_cute: null pointer.\n");
        return;
    }

    using config = HgemvConfig<half, 4>;

    dim3 block(config::NumThreads);
    dim3 grid((M + config::BlockM - 1) / config::BlockM);

    hgemv_tensor_core_cute_kernel<config>
        << <grid, block >> > (const_cast<half*>(A), const_cast<half*>(B), C, M, K);
}