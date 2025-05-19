#ifndef YOLO_HPP__
#define YOLO_HPP__

#include "NvInferVersion.h"
#include "common/affine.hpp"
#include "common/image.hpp"
#include "common/memory.hpp"
#include "common/norm.hpp"
#include "trt/infer.hpp"
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
    int max_image_boxes_ = 1024;

  public:
    virtual bool load(const std::string &engine_file,
        const std::vector<std::string> &names,
        float confidence_threshold,
        float nms_threshold,
        int gpu_id,
        int max_batch_size) = 0;
    
    virtual void preprocess(int ibatch,
        const tensor::Image &image,
        std::shared_ptr<tensor::Memory<unsigned char>> preprocess_buffer,
        affine::LetterBoxMatrix &affine,
        void *stream = nullptr) = 0;

    virtual void adjust_memory(int batch_size) = 0;


};

}

#endif