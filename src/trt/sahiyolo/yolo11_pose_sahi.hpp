#ifndef YOLO11_POSE_SAHI_HPP
#define YOLO11_POSE_SAHI_HPP

#include "NvInferVersion.h"
#include "common/affine.hpp"
#include "common/image.hpp"
#include "common/memory.hpp"
#include "common/norm.hpp"
#include "trt/infer.hpp"
#include "trt/sahiyolo/slice/slice.hpp"

#ifdef NV_TENSORRT_MAJOR >= 10
#include "common/tensorrt.hpp"
namespace TensorRT = TensorRT10;
#else
#include "common/tensorrt8.hpp"
namespace TensorRT = TensorRT8;
#endif

namespace sahiyolo
{

class Yolo11PoseSahiModelImpl : public InferBase
{
  public:
    // for sahi crop image
    std::shared_ptr<slice::SliceImage> slice_;

    // slice params
    bool auto_slice_ = false;
    int slice_width_;
    int slice_height_;
    double slice_horizontal_ratio_;
    double slice_vertical_ratio_;

    std::vector<std::shared_ptr<tensor::Memory<unsigned char>>> preprocess_buffers_;
    // std::vector<std::shared_ptr<tensor::Memory<float>>> affine_matrix_buffers_;
    // 模型输入、模型输出、框存储
    tensor::Memory<float> input_buffer_, bbox_predict_, output_boxarray_;
    // 识别的类别列别
    std::vector<std::string> class_names_;
    // TensorRT engine
    std::shared_ptr<TensorRT::Engine> trt_;
    // 记录框的个数
    tensor::Memory<int> image_box_count_;
    // 记录仿射变换矩阵
    tensor::Memory<float> affine_matrix_;
    // 模型输入宽高
    int network_input_width_, network_input_height_;
    // 预处理参数
    norm_image::Norm normalize_;
    std::vector<int> bbox_head_dims_;
    // 是否为动态模型
    bool isdynamic_model_ = false;

    int max_batch_size_ = 1;

    float confidence_threshold_;
    float nms_threshold_;

    int num_classes_ = 0;
    int device_id_   = 0;

    int num_box_element_ = 9;
    int num_key_point_   = 17;
    int max_image_boxes_ = 1024;

  public:
    virtual InferResult forwards(const std::vector<cv::Mat> &inputs, void *stream = nullptr);

  public:
    bool load(const std::string &engine_file,
              const std::vector<std::string> &names,
              float confidence_threshold,
              float nms_threshold,
              int gpu_id,
              int max_batch_size,
              bool auto_slice,
              int slice_width,
              int slice_height,
              double slice_horizontal_ratio,
              double slice_vertical_ratio);

  private:
    void compute_affine_matrix(affine::LetterBoxMatrix &affine, void *stream = nullptr);
    void preprocess(int ibatch, void *stream);
    void adjust_memory(int batch_size);
};

std::shared_ptr<InferBase> load_yolo_11_pose_sahi(const std::string &engine_file,
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

} // namespace sahiyolo

#endif