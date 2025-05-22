#ifndef DECODE_HPP__
#define DECODE_HPP__

#include <cuda_runtime.h>
#include <memory>

namespace cuda
{

// yolo v5 目标检测后处理kernel
__global__ void decode_kernel_v5(float *predict,
                                 int num_bboxes,
                                 int num_classes,
                                 int output_cdim,
                                 float confidence_threshold,
                                 float *invert_affine_matrix,
                                 float *parray,
                                 int *box_count,
                                 int max_image_boxes,
                                 int num_box_element,
                                 int start_x,
                                 int start_y,
                                 int batch_index);

// yolo v8 v11 目标检测后处理kernel
__global__ void decode_kernel_v11(float *predict,
                                  int num_bboxes,
                                  int num_classes,
                                  int output_cdim,
                                  float confidence_threshold,
                                  float *invert_affine_matrix,
                                  float *parray,
                                  int *box_count,
                                  int max_image_boxes,
                                  int num_box_element,
                                  int start_x,
                                  int start_y,
                                  int batch_index);

// yolo v8 v11 姿态估计后处理kernel
__global__ void decode_kernel_v11_pose(float *predict,
                                       int num_bboxes,
                                       int num_classes,
                                       int output_cdim,
                                       float confidence_threshold,
                                       float *invert_affine_matrix,
                                       float *parray,
                                       int *box_count,
                                       int max_image_boxes,
                                       int num_box_element,
                                       int num_key_point,
                                       int start_x,
                                       int start_y,
                                       int batch_index);

// yolo v8 v11 旋转框后处理kernel
__global__ void decode_kernel_v11_obb(float *predict,
                                      int num_bboxes,
                                      int num_classes,
                                      int output_cdim,
                                      float confidence_threshold,
                                      float *invert_affine_matrix,
                                      float *parray,
                                      int *box_count,
                                      int max_image_boxes,
                                      int num_box_element,
                                      int start_x,
                                      int start_y,
                                      int batch_index);

// yolo nms kernel
__global__ void
fast_nms_kernel(float *bboxes, int *box_count, int max_image_boxes, float threshold, int num_box_element);

// yolo pose nms kernel
__global__ void fast_nms_pose_kernel(
    float *bboxes, int *box_count, int max_image_boxes, float threshold, int num_box_element, int num_key_point);

// yolo obb nms kernel
__global__ void
fast_nms_obb_kernel(float *bboxes, int *box_count, int max_image_boxes, float threshold, int num_box_element);

// yolo mask kernel
__global__ void decode_single_mask_kernel(int left,
                                          int top,
                                          float *mask_weights,
                                          float *mask_predict,
                                          int mask_width,
                                          int mask_height,
                                          float *mask_out,
                                          int mask_dim,
                                          int out_width,
                                          int out_height);

__global__ void decode_dfine_kernel(
    int64_t* labels, 
    float* scores, 
    float* boxes, 
    int num_bboxes,
    float confidence_threshold, 
    int *box_count, 
    int start_x,
    int start_y, 
    float* result, 
    int max_image_boxes, 
    int num_box_element
    );

} // namespace cuda

#endif