#ifndef MERGE_ATTN_STATES_H
#define MERGE_ATTN_STATES_H

#ifdef __cplusplus
extern "C"
{
#endif

enum ScalarType {
        SCALAR_FLOAT = 0,
        SCALAR_HALF = 1,
        SCALAR_BFLOAT16 = 2
 };

void merge_attn_states(
        void* output,
        float* output_lse,
        const void* prefix_output,
        const float* prefix_lse,
        const void* suffix_output,
        const float* suffix_lse,
        unsigned int num_tokens,
        unsigned int num_heads,
        unsigned int head_size,
        int scalar_type,
        cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif