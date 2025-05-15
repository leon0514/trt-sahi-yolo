#ifndef KERNELWARP_HPP__
#define KERNELWARP_HPP__

#include "common/norm.hpp"
#include "kernels/decode.cuh"
#include "kernels/process.cuh"
#include "kernels/slice_kernel.cuh"

void warp_affine_bilinear_and_normalize_plane(uint8_t *src,
                                              int src_line_size,
                                              int src_width,
                                              int src_height,
                                              float *dst,
                                              int dst_width,
                                              int dst_height,
                                              float *matrix_2_3,
                                              uint8_t const_value,
                                              const norm_image::Norm &norm,
                                              cudaStream_t stream);

void warp_affine_bilinear_single_channel_plane(float *src,
                                               int src_line_size,
                                               int src_width,
                                               int src_height,
                                               float *dst,
                                               int dst_width,
                                               int dst_height,
                                               float *matrix_2_3,
                                               float const_value,
                                               cudaStream_t stream);

void warp_affine_bilinear_single_channel_mask_plane(float *src,
                                                    int src_line_size,
                                                    int src_width,
                                                    int src_height,
                                                    uint8_t *dst,
                                                    int dst_width,
                                                    int dst_height,
                                                    float *matrix_2_3,
                                                    uint8_t const_value,
                                                    cudaStream_t stream);

// 对 decode_kernel_v5 的包装
void decode_kernel_invoker_v5(float *predict,
                              int num_bboxes,
                              int num_classes,
                              int output_cdim,
                              float confidence_threshold,
                              float nms_threshold,
                              float *invert_affine_matrix,
                              float *parray,
                              int *box_count,
                              int max_image_boxes,
                              int num_box_element,
                              int start_x,
                              int start_y,
                              int batch_index,
                              cudaStream_t stream);

// 对 decode_kernel_v11 的包装
void decode_kernel_invoker_v11(float *predict,
                               int num_bboxes,
                               int num_classes,
                               int output_cdim,
                               float confidence_threshold,
                               float nms_threshold,
                               float *invert_affine_matrix,
                               float *parray,
                               int *box_count,
                               int max_image_boxes,
                               int num_box_element,
                               int start_x,
                               int start_y,
                               int batch_index,
                               cudaStream_t stream);

// 对 decode_kernel_v11_pose 的包装
void decode_kernel_invoker_v11_pose(float *predict,
                                    int num_bboxes,
                                    int num_classes,
                                    int output_cdim,
                                    float confidence_threshold,
                                    float nms_threshold,
                                    float *invert_affine_matrix,
                                    float *parray,
                                    int *box_count,
                                    int max_image_boxes,
                                    int num_box_element,
                                    int num_key_point,
                                    int start_x,
                                    int start_y,
                                    int batch_index,
                                    cudaStream_t stream);

// 对 decode_kernel_v11_obb 的包装
void decode_kernel_invoker_v11_obb(float *predict,
                                   int num_bboxes,
                                   int num_classes,
                                   int output_cdim,
                                   float confidence_threshold,
                                   float nms_threshold,
                                   float *invert_affine_matrix,
                                   float *parray,
                                   int *box_count,
                                   int max_image_boxes,
                                   int num_box_element,
                                   int start_x,
                                   int start_y,
                                   int batch_index,
                                   cudaStream_t stream);

// 对 decode_single_mask_kernel 的包装
void decode_single_mask_invoker(float left,
                                float top,
                                float *mask_weights,
                                float *mask_predict,
                                int mask_width,
                                int mask_height,
                                float *mask_out,
                                int mask_dim,
                                int out_width,
                                int out_height,
                                cudaStream_t stream);

// 对 fast_nms_kernel 的包装
void fast_nms_kernel_invoker(
    float *parray, int *box_count, int max_image_boxes, float nms_threshold, int num_box_element, cudaStream_t stream);

// 对 fast_nms_pose_kernel 的包装
void fast_nms_kernel_invoker_v11_pose(float *parray,
                                      int *box_count,
                                      int max_image_boxes,
                                      float nms_threshold,
                                      int num_box_element,
                                      int num_key_point,
                                      cudaStream_t stream);

// 对 fast_nms_kernel_v11_obb 的包装
void fast_nms_kernel_invoker_v11_obb(
    float *parray, int *box_count, int max_image_boxes, float nms_threshold, int num_box_element, cudaStream_t stream);

void slice_plane(const uint8_t *image,
                 uint8_t *outs,
                 int *slice_start_point,
                 const int width,
                 const int height,
                 const int slice_width,
                 const int slice_height,
                 const int slice_num_h,
                 const int slice_num_v,
                 void *stream = nullptr);

#endif