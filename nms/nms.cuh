#ifndef NMS_CUH
#define NMS_CUH

#ifdef __cplusplus
extern "C"
{
#endif

int nms(const float* h_boxes, const float* h_scores, int num_boxes, float iou_threshold, int* h_keep_indices);

#ifdef __cplusplus
}
#endif

#endif 