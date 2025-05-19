#ifndef YOLO11_POSE_SAHI_HPP
#define YOLO11_POSE_SAHI_HPP

#include "trt/sahiyolo/sahiyolo.hpp"

namespace sahiyolo
{

class Yolo11PoseSahiModelImpl : public YoloSahiModelImpl
{
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