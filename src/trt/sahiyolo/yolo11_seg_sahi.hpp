#ifndef YOLO11_SEG_SAHI_HPP
#define YOLO11_SEG_SAHI_HPP

#include "trt/sahiyolo/sahiyolo.hpp"

namespace sahiyolo
{

class Yolo11SegSahiModelImpl : public YoloSahiModelImpl
{
  public:
    tensor::Memory<float> segment_predict_;
    // TensorRT engine
    std::shared_ptr<TensorRT::Engine> trt_;
    tensor::Memory<float> mask_affine_matrix_;
    tensor::Memory<float> box_segment_cache_;
    tensor::Memory<unsigned char> original_box_segment_cache_;

    std::vector<int> segment_head_dims_;
    // 是否为动态模型
    bool isdynamic_model_ = false;
  
  public:
    virtual InferResult forwards(const std::vector<cv::Mat> &inputs, void *stream = nullptr) override;

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
      double slice_vertical_ratio) override;

    void adjust_memory(int batch_size);
  private:
    std::shared_ptr<object::SegmentMap> decode_segment(int ib, float *pbox, void *stream);
};

std::shared_ptr<InferBase> load_yolo_11_seg_sahi(const std::string &engine_file,
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