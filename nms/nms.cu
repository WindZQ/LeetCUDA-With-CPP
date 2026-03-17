#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>
#include <cmath>

#include "nms.cuh"

#define WARP_SIZE 32

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                       \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

__device__ float dev_iou(const float* boxes, int a, int b) 
{
    float x1 = boxes[a * 4 + 0];
    float y1 = boxes[a * 4 + 1];
    float x2 = boxes[a * 4 + 2];
    float y2 = boxes[a * 4 + 3];

    float x1b = boxes[b * 4 + 0];
    float y1b = boxes[b * 4 + 1];
    float x2b = boxes[b * 4 + 2];
    float y2b = boxes[b * 4 + 3];

    float inter_x1 = fmaxf(x1, x1b);
    float inter_y1 = fmaxf(y1, y1b);
    float inter_x2 = fminf(x2, x2b);
    float inter_y2 = fminf(y2, y2b);

    float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
    float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float area_b = fmaxf(0.0f, x2b - x1b) * fmaxf(0.0f, y2b - y1b);
    float denom = area_a + area_b - inter_area;

    return denom > 0.0f ? inter_area / denom : 0.0f;
}


__global__ void nms_kernel(const float* boxes_sorted,
    int num_boxes,
    float iou_threshold,
    int* keep) 
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int i = 0; i < num_boxes; ++i) 
    {
        int suppressed = 0;
        for (int j = 0; j < i; ++j) 
        {
            if (keep[j] == 0) continue;
            float iou = dev_iou(boxes_sorted, i, j);
            if (iou > iou_threshold) {
                suppressed = 1;
                break;
            }
        }

        keep[i] = suppressed ? 0 : 1;
    }
}

int nms(const float* h_boxes,
    const float* h_scores,
    int num_boxes,
    float iou_threshold,
    int* h_keep_indices) 
{
    std::vector<int> order(num_boxes);
    std::iota(order.begin(), order.end(), 0);

    std::stable_sort(order.begin(), order.end(),
        [&](int a, int b) { return h_scores[a] > h_scores[b]; });

    std::vector<float> h_boxes_sorted(num_boxes * 4);
    for (int i = 0; i < num_boxes; ++i) 
    {
        int idx = order[i];
        h_boxes_sorted[i * 4 + 0] = h_boxes[idx * 4 + 0];
        h_boxes_sorted[i * 4 + 1] = h_boxes[idx * 4 + 1];
        h_boxes_sorted[i * 4 + 2] = h_boxes[idx * 4 + 2];
        h_boxes_sorted[i * 4 + 3] = h_boxes[idx * 4 + 3];
    }

    float* d_boxes = nullptr;
    int* d_keep = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_boxes, num_boxes * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_keep, num_boxes * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_boxes, h_boxes_sorted.data(),
        num_boxes * 4 * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_keep, 0, num_boxes * sizeof(int)));
    dim3 block(WARP_SIZE);
    dim3 grid((num_boxes + WARP_SIZE - 1) / WARP_SIZE);
    nms_kernel << < grid, block >> > (d_boxes, num_boxes, iou_threshold, d_keep);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_keep(num_boxes);
    CUDA_CHECK(cudaMemcpy(h_keep.data(), d_keep,
        num_boxes * sizeof(int),
        cudaMemcpyDeviceToHost));

    int keep_count = 0;
    for (int i = 0; i < num_boxes; ++i) 
    {
        if (h_keep[i]) {
            h_keep_indices[keep_count++] = order[i];
        }
    }

    CUDA_CHECK(cudaFree(d_boxes));
    CUDA_CHECK(cudaFree(d_keep));

    return keep_count;
}