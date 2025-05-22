#ifndef DFINESAHI_HPP__
#define DFINESAHI_HPP__

#include "NvInferVersion.h"
#include "common/affine.hpp"
#include "common/image.hpp"
#include "common/memory.hpp"
#include "common/norm.hpp"
#include "trt/infer.hpp"
#include "trt/slice/slice.hpp"
#include "kernels/kernel_warp.hpp"
#include <memory>


#if NV_TENSORRT_MAJOR >= 10
#include "common/tensorrt.hpp"
namespace TensorRT = TensorRT10;
#else
#include "common/tensorrt8.hpp"
namespace TensorRT = TensorRT8;
#endif


namespace dfinesahi
{

class DFineSahiModelImpl : public InferBase
{

public:
    DFineSahiModelImpl() = default;

    std::shared_ptr<slice::SliceImage> slice_;
    // slice params
    bool auto_slice_ = false;
    int slice_width_;
    int slice_height_;
    double slice_horizontal_ratio_;
    double slice_vertical_ratio_;

    std::vector<std::shared_ptr<tensor::Memory<unsigned char>>> preprocess_images_buffers_;

    tensor::Memory<float> input_buffer_image_;
    tensor::Memory<int64_t>   input_buffer_orig_target_size_;

    tensor::Memory<float> result_;

    tensor::Memory<int> image_box_count_;

    tensor::Memory<int> start_;

    tensor::Memory<int64_t> output_labels_;
    tensor::Memory<float> output_boxes_;
    tensor::Memory<float> output_scores_;

    std::vector<std::string> class_names_;

    std::shared_ptr<TensorRT::Engine> trt_;
    tensor::Memory<float> affine_matrix_;
    int network_input_width_, network_input_height_;
    norm_image::Norm normalize_;

    std::vector<int> box_head_dims_;
    std::vector<int> label_head_dims_;
    std::vector<int> score_head_dims_;

    bool isdynamic_model_ = false;
    int max_batch_size_ = 1;
    int device_id_  = 0;

    int num_box_element_ = 9;
    int max_image_boxes_ = 1000;

    float confidence_threshold_;
    float nms_threshold_;

    bool load(const std::string &engine_file,
        const std::vector<std::string> &names,
        float confidence_threshold_,
        float nms_threshold_,
        int gpu_id,
        int max_batch_size,
        bool auto_slice,
        int slice_width,
        int slice_height,
        double slice_horizontal_ratio,
        double slice_vertical_ratio);

    void preprocess(int ibatch,
        void *stream = nullptr)
    {
        size_t input_numel  = network_input_width_ * network_input_height_ * 3;
        float *input_device = input_buffer_image_.gpu() + ibatch * input_numel;
        size_t size_image   = slice_->slice_width_ * slice_->slice_height_ * 3;

        int64_t* orig_target_sizes_cpu_ptr = input_buffer_orig_target_size_.cpu();
        orig_target_sizes_cpu_ptr[ibatch * 2 + 0] = slice_->slice_width_;  // 原始图像高度
        orig_target_sizes_cpu_ptr[ibatch * 2 + 1] = slice_->slice_height_;  // 原始图像宽度

        float *affine_matrix_device = affine_matrix_.gpu();
        uint8_t *image_device       = slice_->output_images_.gpu() + ibatch * size_image;

        // speed up
        cudaStream_t stream_ = (cudaStream_t)stream;

        warp_affine_bilinear_and_normalize_plane(image_device,
                                                slice_->slice_width_ * 3,
                                                slice_->slice_width_,
                                                slice_->slice_height_,
                                                input_device,
                                                network_input_width_,
                                                network_input_height_,
                                                affine_matrix_device,
                                                114,
                                                normalize_,
                                                stream_);
    }

    void adjust_memory(int batch_size)
    {
        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        input_buffer_image_.gpu(batch_size * input_numel);
        input_buffer_orig_target_size_.gpu(batch_size * 2);
        input_buffer_orig_target_size_.cpu(batch_size * 2);

        output_boxes_.gpu(batch_size * box_head_dims_[1] * box_head_dims_[2]);
        output_labels_.gpu(batch_size * label_head_dims_[1]);
        output_scores_.gpu(batch_size * score_head_dims_[1]);

        output_boxes_.cpu(batch_size * box_head_dims_[1] * box_head_dims_[2]);
        output_labels_.cpu(batch_size * label_head_dims_[1]);
        output_scores_.cpu(batch_size * score_head_dims_[1]);
    
        affine_matrix_.gpu(6);
        affine_matrix_.cpu(6);

        image_box_count_.gpu(1);
        image_box_count_.cpu(1);

        int num_image = slice_->slice_num_h_ * slice_->slice_num_v_;
        start_.gpu(num_image * 2);
        start_.cpu(num_image * 2);

        result_.gpu(num_image * box_head_dims_[1] * num_box_element_);
        result_.cpu(num_image * box_head_dims_[1] * num_box_element_);
    }

    void compute_affine_matrix(affine::ResizeMatrix &affine, void *stream = nullptr)
    {
        affine.compute(std::make_tuple(slice_->slice_width_, slice_->slice_height_),
        std::make_tuple(network_input_width_, network_input_height_));

        float *affine_matrix_device = affine_matrix_.gpu();
        float *affine_matrix_host   = affine_matrix_.cpu();

        cudaStream_t stream_ = (cudaStream_t)stream;
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        checkRuntime(
        cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream_));
    }

    virtual InferResult forwards(const std::vector<cv::Mat> &inputs, void *stream = nullptr);
};

std::shared_ptr<InferBase> load_dfine_sahi(const std::string &engine_file,
    const std::vector<std::string> &names,
    int gpu_id,
    float confidence_threshold,
    float nms_threshold,
    int max_batch_size,
    bool auto_slice,
    int slice_width,
    int slice_height,
    double slice_horizontal_ratio,
    double slice_vertical_ratio);


}



#endif