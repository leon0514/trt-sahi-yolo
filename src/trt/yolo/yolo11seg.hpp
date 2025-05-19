#ifndef YOLO11SEG_HPP__
#define YOLO11SEG_HPP__

#include "trt/yolo.hpp"

namespace yolo
{

class Yolo11SegModelImpl : public YoloModelImpl
{
  public:
    tensor::Memory<float> segment_predict_;
    // mask框的仿射矩阵
    tensor::Memory<float> mask_affine_matrix_;
    std::vector<std::shared_ptr<tensor::Memory<float>>> inverse_affine_matrixs_;
    // 框的segment缓存
    tensor::Memory<float> box_segment_cache_;
    // 框的segment缓存
    tensor::Memory<unsigned char> original_box_segment_cache_;
    std::vector<int> segment_head_dims_;

  public:
    virtual InferResult forwards(const std::vector<cv::Mat> &inputs, void *stream = nullptr);
    virtual bool load(const std::string &engine_file,
        const std::vector<std::string> &names,
        float confidence_threshold,
        float nms_threshold,
        int gpu_id,
        int max_batch_size) override;

  protected:
    virtual void preprocess(int ibatch,
        const tensor::Image &image,
        std::shared_ptr<tensor::Memory<unsigned char>> preprocess_buffer,
        affine::LetterBoxMatrix &affine,
        void *stream = nullptr) override;

    virtual void adjust_memory(int batch_size) override;
  private:
    std::shared_ptr<object::SegmentMap> decode_segment(int ib, float *pbox, void *stream);
};

std::shared_ptr<InferBase> load_yolo_11_seg(const std::string &engine_file,
                                            const std::vector<std::string> &names,
                                            int gpu_id,
                                            float confidence_threshold,
                                            float nms_threshold,
                                            int max_batch_size);

} // end namespace yolo

#endif