#ifndef SGEMM_WMMA_TF32_STAGE_CUH
#define SGEMM_WMMA_TF32_STAGE_CUH

#ifdef __cplusplus
extern "C"
{
#endif
void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages(float* a, float* b, float* c,
        int M, int N, int K,
        int stages, bool swizzle,
        int swizzle_stride);

void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem(float* a, float* b, float* c,
    int M, int N, int K,
    int stages, bool swizzle,
    int swizzle_stride);
#ifdef __cplusplus
}
#endif

#endif 