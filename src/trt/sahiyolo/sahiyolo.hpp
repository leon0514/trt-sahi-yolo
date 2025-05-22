#ifndef SAHIYOLO_HPP__
#define SAHIYOLO_HPP__

#include "NvInferVersion.h"
#include "common/affine.hpp"
#include "common/image.hpp"
#include "common/memory.hpp"
#include "common/norm.hpp"
#include "trt/slice/slice.hpp"
#include "kernels/kernel_warp.hpp"
#include "trt/infer.hpp"
#include <memory>

#if NV_TENSORRT_MAJOR >= 10
#include "common/tensorrt.hpp"
namespace TensorRT = TensorRT10;
#else
#include "common/tensorrt8.hpp"
namespace TensorRT = TensorRT8;
#endif

namespace sahiyolo
{

class YoloSahiModelImpl : public InferBase
{
  public:
    YoloSahiModelImpl() = default;
    virtual ~YoloSahiModelImpl() = default;
    
  protected:
    // for sahi crop image
    std::shared_ptr<slice::SliceImage> slice_;

    // slice params
    bool auto_slice_ = false;
    int slice_width_;
    int slice_height_;
    double slice_horizontal_ratio_;
    double slice_vertical_ratio_;

    std::vector<std::shared_ptr<tensor::Memory<unsigned char>>> preprocess_buffers_;
    tensor::Memory<float> input_buffer_, bbox_predict_, output_boxarray_;
    std::vector<std::string> class_names_;
    std::shared_ptr<TensorRT::Engine> trt_;
    tensor::Memory<int> image_box_count_;
    tensor::Memory<float> affine_matrix_;
    tensor::Memory<float> inverse_affine_matrix_;
    int network_input_width_, network_input_height_;
    norm_image::Norm normalize_;
    std::vector<int> bbox_head_dims_;
    bool isdynamic_model_ = false;

    float confidence_threshold_;
    float nms_threshold_;

    int max_batch_size_;

    int num_classes_ = 0;
    int device_id_   = 0;

    int num_box_element_ = 9;
    int num_key_point_   = 0;
    int single_image_max_boxes_ = 1024;
    int max_image_boxes_ = single_image_max_boxes_;

  public:
    virtual bool load(const std::string &engine_file,
      const std::vector<std::string> &names,
      float confidence_threshold,
      float nms_threshold,
      int gpu_id,
      int max_batch_size,
      bool auto_slice,
      int slice_width,
      int slice_height,
      double slice_horizontal_ratio,
      double slice_vertical_ratio) = 0;
    
    virtual void preprocess(int ibatch,
        void *stream = nullptr)
    {
        size_t input_numel  = network_input_width_ * network_input_height_ * 3;
        float *input_device = input_buffer_.gpu() + ibatch * input_numel;
        size_t size_image   = slice_->slice_width_ * slice_->slice_height_ * 3;

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
        input_buffer_.gpu(batch_size * input_numel);
        bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
        output_boxarray_.gpu(max_image_boxes_ * (num_box_element_ + num_key_point_ * 3));
        output_boxarray_.cpu(max_image_boxes_ * (num_box_element_ + num_key_point_ * 3));

        affine_matrix_.gpu(6);
        affine_matrix_.cpu(6);

        // 只有分割的时候用到了
        inverse_affine_matrix_.gpu(6);
        inverse_affine_matrix_.cpu(6);
    
        image_box_count_.gpu(1);
        image_box_count_.cpu(1);
    }

    void compute_affine_matrix(affine::LetterBoxMatrix &affine, void *stream = nullptr)
    {
        affine.compute(std::make_tuple(slice_->slice_width_, slice_->slice_height_),
        std::make_tuple(network_input_width_, network_input_height_));

        float *affine_matrix_device = affine_matrix_.gpu();
        float *affine_matrix_host   = affine_matrix_.cpu();

        float *inverse_affine_matrix_device = inverse_affine_matrix_.gpu();
        float *inverse_affine_matrix_host   = inverse_affine_matrix_.cpu();

        cudaStream_t stream_ = (cudaStream_t)stream;
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        checkRuntime(
        cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream_));

        memcpy(inverse_affine_matrix_host, affine.i2d, sizeof(affine.i2d));
        checkRuntime(cudaMemcpyAsync(inverse_affine_matrix_device,
                            inverse_affine_matrix_host,
                            sizeof(affine.i2d),
                            cudaMemcpyHostToDevice,
                            stream_));
    }


};

}

#endif