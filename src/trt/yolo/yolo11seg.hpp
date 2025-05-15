#ifndef YOLO11SEG_HPP__
#define YOLO11SEG_HPP__

#include "NvInferVersion.h"
#include "common/check.hpp"
#include "common/image.hpp"
#include "common/memory.hpp"
#include "common/norm.hpp"
#include "trt/infer.hpp"
#include <cuda_runtime.h>

#if NV_TENSORRT_MAJOR >= 10
#include "common/tensorrt.hpp"
namespace TensorRT = TensorRT10;
#else
#include "common/tensorrt8.hpp"
namespace TensorRT = TensorRT8;
#endif

namespace yolo
{

class Yolo11SegModelImpl : public InferBase
{
  public:
    std::vector<std::shared_ptr<tensor::Memory<unsigned char>>> preprocess_buffers_;
    // std::vector<std::shared_ptr<tensor::Memory<float>>> affine_matrix_buffers_;
    // 模型输入、模型输出、框存储
    tensor::Memory<float> input_buffer_, bbox_predict_, segment_predict_, output_boxarray_;
    // 识别的类别列别
    std::vector<std::string> class_names_;
    // TensorRT engine
    std::shared_ptr<TensorRT::Engine> trt_;
    // 记录框的个数
    std::vector<std::shared_ptr<tensor::Memory<int>>> image_box_counts_;
    // affine matrix 保存变量
    std::vector<std::shared_ptr<tensor::Memory<float>>> affine_matrixs_;
    // inverse affine matrix 保存变量
    std::vector<std::shared_ptr<tensor::Memory<float>>> inverse_affine_matrixs_;
    // mask框的仿射矩阵
    tensor::Memory<float> mask_affine_matrix_;
    // 框的segment缓存
    tensor::Memory<float> box_segment_cache_;
    // 框的segment缓存
    tensor::Memory<unsigned char> original_box_segment_cache_;
    // 模型输入宽高
    int network_input_width_, network_input_height_;
    // 预处理参数
    norm_image::Norm normalize_;
    std::vector<int> bbox_head_dims_;
    std::vector<int> segment_head_dims_;
    // 是否为动态模型
    bool isdynamic_model_ = false;

    int max_batch_size_ = 1;

    float confidence_threshold_;
    float nms_threshold_;

    int num_classes_ = 0;
    int device_id_   = 0;

    int num_box_element_ = 9;
    int max_image_boxes_ = 1024;

  public:
    virtual InferResult forwards(const std::vector<cv::Mat> &inputs, void *stream = nullptr);

  public:
    bool load(const std::string &engine_file,
              const std::vector<std::string> &names,
              float confidence_threshold,
              float nms_threshold,
              int gpu_id,
              int max_batch_size);

  private:
    std::shared_ptr<object::SegmentMap> decode_segment(int ib, float *pbox, void *stream);
    void preprocess(int ibatch,
                    const tensor::Image &image,
                    std::shared_ptr<tensor::Memory<unsigned char>> preprocess_buffer,
                    affine::LetterBoxMatrix &affine,
                    void *stream = nullptr);
    void adjust_memory(int batch_size);
};

std::shared_ptr<InferBase> load_yolo_11_seg(const std::string &engine_file,
                                            const std::vector<std::string> &names,
                                            int gpu_id,
                                            float confidence_threshold,
                                            float nms_threshold,
                                            int max_batch_size);

} // end namespace yolo

#endif