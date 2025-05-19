#ifndef INFER_HPP__
#define INFER_HPP__

#include "common/object.hpp"
#include <iostream>
#include <variant>

enum class ModelType : int
{
    YOLOV5         = 0,
    YOLOV5SAHI     = 1,
    YOLO11         = 2,
    YOLO11SAHI     = 3,
    YOLO11POSE     = 4,
    YOLO11POSESAHI = 5,
    YOLO11SEG      = 6,
    YOLO11SEGSAHI  = 7,
    YOLO11OBB      = 8,
    YOLO11OBBSAHI  = 9,
};

// 为枚举类添加字符串转换功能
namespace ModelTypeConverter
{
// 枚举转字符串
inline std::string to_string(ModelType type)
{
    switch (type)
    {
    case ModelType::YOLOV5:
        return "YOLOV5";
    case ModelType::YOLOV5SAHI:
        return "YOLOV5SAHI";
    case ModelType::YOLO11:
        return "YOLO11";
    case ModelType::YOLO11SAHI:
        return "YOLO11SAHI";
    case ModelType::YOLO11POSE:
        return "YOLO11POSE";
    case ModelType::YOLO11POSESAHI:
        return "YOLO11POSESAHI";
    case ModelType::YOLO11SEG:
        return "YOLO11SEG";
    case ModelType::YOLO11SEGSAHI:
        return "YOLO11SEGSAHI";
    case ModelType::YOLO11OBB:
        return "YOLO11OBB";
    case ModelType::YOLO11OBBSAHI:
        return "YOLO11OBBSAHI";
    default:
        return "UNKNOWN";
    }
}

// 字符串转枚举
inline ModelType from_string(const std::string &str)
{
    static const std::unordered_map<std::string, ModelType> str2enum = {{"YOLOV5", ModelType::YOLOV5},
                                                                        {"YOLOV5SAHI", ModelType::YOLOV5SAHI},
                                                                        {"YOLO11", ModelType::YOLO11},
                                                                        {"YOLO11SAHI", ModelType::YOLO11SAHI},
                                                                        {"YOLO11POSE", ModelType::YOLO11POSE},
                                                                        {"YOLO11POSESAHI", ModelType::YOLO11POSESAHI},
                                                                        {"YOLO11SEG", ModelType::YOLO11SEG},
                                                                        {"YOLO11SEGSAHI", ModelType::YOLO11SEGSAHI},
                                                                        {"YOLO11OBB", ModelType::YOLO11OBB},
                                                                        {"YOLO11OBBSAHI", ModelType::YOLO11OBBSAHI}};

    auto it = str2enum.find(str);
    if (it != str2enum.end())
    {
        return it->second;
    }
    throw std::invalid_argument("Invalid ModelType string: " + str);
}
} // namespace ModelTypeConverter

inline std::ostream &operator<<(std::ostream &os, ModelType type)
{
    os << ModelTypeConverter::to_string(type);
    return os;
}

using InferResult = std::variant<object::DetectionResultArray,
                                 object::DetectionObbResultArray,
                                 object::PoseResultArray,
                                 object::SegmentationResultArray,
                                 std::vector<object::DetectionResultArray>,
                                 std::vector<object::DetectionObbResultArray>,
                                 std::vector<object::PoseResultArray>,
                                 std::vector<object::SegmentationResultArray>>;
class InferBase
{
  public:
    virtual InferResult forwards(const std::vector<cv::Mat> &inputs, void *stream = nullptr) = 0;

    virtual ~InferBase() = default;

};

std::shared_ptr<InferBase> load(const std::string &model_path,
                                ModelType model_type,
                                const std::vector<std::string> &names,
                                int gpu_id                    = 0,
                                float confidence_threshold    = 0.5f,
                                float nms_threshold           = 0.45f,
                                int max_batch_size            = 1,
                                bool auto_slice               = false,
                                int slice_width               = 640,
                                int slice_height              = 640,
                                double slice_horizontal_ratio = 0.3,
                                double slice_vertical_ratio   = 0.3);

#endif // INFER_HPP__