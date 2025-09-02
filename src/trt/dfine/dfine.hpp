#ifndef DFINE_HPP__
#define DFINE_HPP__

#include "NvInferVersion.h"
#include "common/affine.hpp"
#include "common/image.hpp"
#include "common/memory.hpp"
#include "common/norm.hpp"
#include "trt/infer.hpp"
#include "kernels/kernel_warp.hpp"
#include <memory>


#if NV_TENSORRT_MAJOR >= 10
#include "common/tensorrt.hpp"
namespace TensorRT = TensorRT10;
#else
#include "common/tensorrt8.hpp"
namespace TensorRT = TensorRT8;
#endif


namespace dfine
{

class DFineModelImpl : public InferBase
{

public:
    DFineModelImpl() = default;

    std::vector<std::shared_ptr<tensor::Memory<unsigned char>>> preprocess_images_buffers_;

    tensor::Memory<float> input_buffer_image_;
#if NV_TENSORRT_MAJOR >= 10
    tensor::Memory<int64_t>   input_buffer_orig_target_size_;
    tensor::Memory<int64_t> output_labels_;
#else
    tensor::Memory<int32_t>   input_buffer_orig_target_size_;
    tensor::Memory<int32_t> output_labels_;
#endif
    tensor::Memory<float> output_boxes_;
    tensor::Memory<float> output_scores_;

    std::vector<std::string> class_names_;

    std::shared_ptr<TensorRT::Engine> trt_;
    std::vector<std::shared_ptr<tensor::Memory<int>>> image_box_counts_;
    std::vector<std::shared_ptr<tensor::Memory<float>>> affine_matrixs_;
    int network_input_width_, network_input_height_;
    norm_image::Norm normalize_;

    std::vector<int> box_head_dims_;
    std::vector<int> label_head_dims_;
    std::vector<int> score_head_dims_;

    bool isdynamic_model_ = false;
    int max_batch_size_ = 1;
    int device_id_  = 0;

    float confidence_threshold_;
    float nms_threshold_;

    bool load(const std::string &engine_file,
        const std::vector<std::string> &names,
        float confidence_threshold_,
        float nms_threshold_,
        int gpu_id,
        int max_batch_size);

    void preprocess(int ibatch,
        const tensor::Image &image,
        std::shared_ptr<tensor::Memory<unsigned char>> preprocess_image_buffer,
        affine::ResizeMatrix &affine,
        void *stream = nullptr)
    {
        affine.compute(std::make_tuple(image.width, image.height),
                        std::make_tuple(network_input_width_, network_input_height_));
        size_t input_numel  = network_input_width_ * network_input_height_ * 3;
        float *input_device = input_buffer_image_.gpu() + ibatch * input_numel;
        size_t size_image   = image.width * image.height * 3;

#if NV_TENSORRT_MAJOR >= 10
        int64_t* orig_target_sizes_cpu_ptr = input_buffer_orig_target_size_.cpu();
#else
        int32_t* orig_target_sizes_cpu_ptr = input_buffer_orig_target_size_.cpu();
#endif
        orig_target_sizes_cpu_ptr[ibatch * 2 + 0] = image.width;  // 原始图像高度
        orig_target_sizes_cpu_ptr[ibatch * 2 + 1] = image.height;  // 原始图像宽度
        
    
        uint8_t *image_device = preprocess_image_buffer->gpu(size_image);
        uint8_t *image_host   = preprocess_image_buffer->cpu(size_image);
    
        float *affine_matrix_device = affine_matrixs_[ibatch]->gpu();
        float *affine_matrix_host   = affine_matrixs_[ibatch]->cpu();
    
        cudaStream_t stream_ = (cudaStream_t)stream;
        memcpy(image_host, image.bgrptr, size_image);
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        checkRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
        checkRuntime(
            cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream_));
    
        warp_affine_bilinear_and_normalize_plane(image_device,
                                                    image.width * 3,
                                                    image.width,
                                                    image.height,
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
    
        if ((int)preprocess_images_buffers_.size() < batch_size)
        {
            for (int i = preprocess_images_buffers_.size(); i < batch_size; ++i)
            {
                // 分配图片所需要的空间
                preprocess_images_buffers_.push_back(std::make_shared<tensor::Memory<unsigned char>>());
                image_box_counts_.push_back(std::make_shared<tensor::Memory<int>>());
                affine_matrixs_.push_back(std::make_shared<tensor::Memory<float>>());
                // 分配记录框所需要的空间
                image_box_counts_[i]->gpu(1);
                image_box_counts_[i]->cpu(1);
                // 分配仿射矩阵苏需要的空间
                affine_matrixs_[i]->gpu(6);
                affine_matrixs_[i]->cpu(6);
            }
        }
    }

    virtual InferResult forwards(const std::vector<cv::Mat> &inputs, void *stream = nullptr);
};

std::shared_ptr<InferBase> load_dfine(const std::string &engine_file,
    const std::vector<std::string> &names,
    int gpu_id,
    float confidence_threshold,
    float nms_threshold,
    int max_batch_size = 1);


}



#endif