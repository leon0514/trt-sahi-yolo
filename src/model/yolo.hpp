#ifndef YOLOV11_HPP__
#define YOLOV11_HPP__
#include "common/memory.hpp"
#include "common/image.hpp"
#include "common/check.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <iomanip>

namespace yolo
{

struct Point
{
    float x, y, vis;
    Point() = default;
    Point(float x, float y, float vis) :
        x(x), y(y), vis(vis) {}
};

struct InstanceSegmentMap 
{
    int width = 0, height = 0;      // width % 8 == 0
    unsigned char *data = nullptr;  // is width * height memory
  
    InstanceSegmentMap(int width, int height)
    {
        this->width = width;
        this->height = height;
        checkRuntime(cudaMallocHost(&this->data, width * height));
    }
    virtual ~InstanceSegmentMap()
    {
        if (this->data) 
        {
            checkRuntime(cudaFreeHost(this->data));
            this->data = nullptr;
        }
        this->width = 0;
        this->height = 0; 
    }
};

struct Box 
{
    float left, top, right, bottom, confidence;
    int class_label;
    std::vector<Point> pose;

    std::shared_ptr<InstanceSegmentMap> seg;  // valid only in segment task

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int class_label)
        : left(left),
            top(top),
            right(right),
            bottom(bottom),
            confidence(confidence),
            class_label(class_label) {}
    friend std::ostream& operator<<(std::ostream& os, const Box& box)
    {
        os << std::fixed << std::setprecision(2)  // 设置浮点数精度
           << "Box:\n"
           << "  Left:      " << std::setw(6) << box.left << "\n"
           << "  Top:       " << std::setw(6) << box.top << "\n"
           << "  Right:     " << std::setw(6) << box.right << "\n"
           << "  Bottom:    " << std::setw(6) << box.bottom << "\n"
           << "  Confidence: " << std::setw(6) << box.confidence << "\n"
           << "  Class Label: " << box.class_label;
        return os;
    }
};

enum class YoloType : int{
    YOLOV5  = 0,
    YOLOV5SEG = 1,
    YOLOV8  = 2,
    YOLOV8POSE = 3,
    YOLOV8SEG = 4,
    YOLOV11 = 5,
    YOLOV11POSE = 6,
    YOLOV11SEG = 7,
};

using BoxArray = std::vector<Box>;


class Infer {
public:
    virtual BoxArray forward(const tensor::Image &image, int slice_width, int slice_height, float overlap_width_ratio, float overlap_height_ratio, void *stream = nullptr) = 0;
    virtual BoxArray forward(const tensor::Image &image, void *stream = nullptr) = 0;
    virtual BoxArray forwards(void *stream = nullptr) = 0;
};

std::shared_ptr<Infer> load(const std::string &engine_file, YoloType yolo_type, int gpu_id = 0, float confidence_threshold=0.5f, float nms_threshold=0.45f);

}



#endif