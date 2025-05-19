#ifndef YOLOV5_HPP__
#define YOLOV5_HPP__

#include "trt/yolo/yolo.hpp"

namespace yolo
{

class Yolov5ModelImpl : public YoloModelImpl
{
  public:
    virtual InferResult forwards(const std::vector<cv::Mat> &inputs, void *stream = nullptr);
    virtual bool load(const std::string &engine_file,
        const std::vector<std::string> &names,
        float confidence_threshold,
        float nms_threshold,
        int gpu_id,
        int max_batch_size) override;
};

std::shared_ptr<InferBase> load_yolo_v5(const std::string &engine_file,
                                        const std::vector<std::string> &names,
                                        int gpu_id,
                                        float confidence_threshold,
                                        float nms_threshold,
                                        int max_batch_size);

} // end namespace yolo

#endif // endif YOLOV5_HPP__