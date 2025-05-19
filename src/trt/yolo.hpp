#ifndef YOLO_HPP__
#define YOLO_HPP__

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

namespace yolo
{

class YoloModelImpl : public InferBase
{
  protected:
    std::vector<std::shared_ptr<tensor::Memory<unsigned char>>> preprocess_buffers_;
    tensor::Memory<float> input_buffer_, bbox_predict_, output_boxarray_;
    std::vector<std::string> class_names_;
    std::shared_ptr<TensorRT::Engine> trt_;
    std::vector<std::shared_ptr<tensor::Memory<int>>> image_box_counts_;
    std::vector<std::shared_ptr<tensor::Memory<float>>> affine_matrixs_;
    int network_input_width_, network_input_height_;
    norm_image::Norm normalize_;
    std::vector<int> bbox_head_dims_;
    bool isdynamic_model_ = false;

    int max_batch_size_ = 1;

    float confidence_threshold_;
    float nms_threshold_;

    int num_classes_ = 0;
    int device_id_   = 0;

    int num_box_element_ = 9;
    int num_key_point_   = 0;
    int max_image_boxes_ = 1024;

  public:
    virtual bool load(const std::string &engine_file,
        const std::vector<std::string> &names,
        float confidence_threshold,
        float nms_threshold,
        int gpu_id,
        int max_batch_size) = 0;
    
    void preprocess(int ibatch,
        const tensor::Image &image,
        std::shared_ptr<tensor::Memory<unsigned char>> preprocess_buffer,
        affine::LetterBoxMatrix &affine,
        void *stream = nullptr)
    {
        affine.compute(std::make_tuple(image.width, image.height),
                        std::make_tuple(network_input_width_, network_input_height_));
        size_t input_numel  = network_input_width_ * network_input_height_ * 3;
        float *input_device = input_buffer_.gpu() + ibatch * input_numel;
        size_t size_image   = image.width * image.height * 3;
    
        uint8_t *image_device = preprocess_buffer->gpu(size_image);
        uint8_t *image_host   = preprocess_buffer->cpu(size_image);
    
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
        input_buffer_.gpu(batch_size * input_numel);
        bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
    
        output_boxarray_.gpu(batch_size * (max_image_boxes_ * (num_box_element_ + (num_key_point_ * 3))));
        output_boxarray_.cpu(batch_size * (max_image_boxes_ * (num_box_element_ + (num_key_point_ * 3))));
    
        if ((int)preprocess_buffers_.size() < batch_size)
        {
            for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
            {
                // 分配图片所需要的空间
                preprocess_buffers_.push_back(std::make_shared<tensor::Memory<unsigned char>>());
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


};

}

#endif