#ifndef ROPE_CUH
#define ROPE_CUH

#ifdef __cplusplus
extern "C"
{
#endif

void rope_f32(float* x, float* out, int seq_len, int hidden_size, cudaStream_t stream);
void rope_f32_v2(float* x, float* out, int seq_len, int hidden_size, cudaStream_t stream);
void rope_f32x4_pack(float* x, float* out, int seq_len, int hidden_size, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif