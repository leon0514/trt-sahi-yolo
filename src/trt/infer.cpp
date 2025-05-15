#include "trt/infer.hpp"
#include "trt/yolo/yolo11.hpp"
#include "trt/yolo/yolo11obb.hpp"
#include "trt/yolo/yolo11pose.hpp"
#include "trt/yolo/yolo11seg.hpp"
#include "trt/yolo/yolov5.hpp"

#include "trt/sahiyolo/yolo11_pose_sahi.hpp"
#include "trt/sahiyolo/yolo11_sahi.hpp"
#include "trt/sahiyolo/yolo11_seg_sahi.hpp"
#include "trt/sahiyolo/yolo11_obb_sahi.hpp"
#include "trt/sahiyolo/yolov5_sahi.hpp"

std::shared_ptr<InferBase> load(const std::string &model_path,
                                ModelType model_type,
                                const std::vector<std::string> &names,
                                int gpu_id,
                                float confidence_threshold,
                                float nms_threshold,
                                int max_batch_size,
                                bool auto_slice,
                                int slice_width,
                                int slice_height,
                                double slice_horizontal_ratio,
                                double slice_vertical_ratio)
{
    printf("Loading model: %s\n", model_path.c_str());
    printf("Model type: %s\n", ModelTypeConverter::to_string(model_type).c_str());
    std::shared_ptr<InferBase> instance;
    switch (model_type)
    {
    case ModelType::YOLOV5:
        instance = yolo::load_yolo_v5(model_path, names, gpu_id, confidence_threshold, nms_threshold, max_batch_size);
        break;
    case ModelType::YOLO11:
        instance = yolo::load_yolo_11(model_path, names, gpu_id, confidence_threshold, nms_threshold, max_batch_size);
        break;
    case ModelType::YOLO11SEG:
        instance =
            yolo::load_yolo_11_seg(model_path, names, gpu_id, confidence_threshold, nms_threshold, max_batch_size);
        break;
    case ModelType::YOLO11POSE:
        instance =
            yolo::load_yolo_11_pose(model_path, names, gpu_id, confidence_threshold, nms_threshold, max_batch_size);
        break;
    case ModelType::YOLO11OBB:
        instance =
            yolo::load_yolo_11_obb(model_path, names, gpu_id, confidence_threshold, nms_threshold, max_batch_size);
        break;
    case ModelType::YOLOV5SAHI:
        instance = sahiyolo::load_yolo_v5_sahi(model_path,
                                               names,
                                               gpu_id,
                                               confidence_threshold,
                                               nms_threshold,
                                               max_batch_size,
                                               auto_slice,
                                               slice_width,
                                               slice_height,
                                               slice_horizontal_ratio,
                                               slice_vertical_ratio);
        break;
    case ModelType::YOLO11SAHI:
        instance = sahiyolo::load_yolo_11_sahi(model_path,
                                               names,
                                               gpu_id,
                                               confidence_threshold,
                                               nms_threshold,
                                               max_batch_size,
                                               auto_slice,
                                               slice_width,
                                               slice_height,
                                               slice_horizontal_ratio,
                                               slice_vertical_ratio);
        break;
    case ModelType::YOLO11POSESAHI:
        instance = sahiyolo::load_yolo_11_pose_sahi(model_path,
                                                    names,
                                                    gpu_id,
                                                    confidence_threshold,
                                                    nms_threshold,
                                                    max_batch_size,
                                                    auto_slice,
                                                    slice_width,
                                                    slice_height,
                                                    slice_horizontal_ratio,
                                                    slice_vertical_ratio);
        break;
    case ModelType::YOLO11SEGSAHI:
        instance = sahiyolo::load_yolo_11_seg_sahi(model_path,
                                                   names,
                                                   gpu_id,
                                                   confidence_threshold,
                                                   nms_threshold,
                                                   max_batch_size,
                                                   auto_slice,
                                                   slice_width,
                                                   slice_height,
                                                   slice_horizontal_ratio,
                                                   slice_vertical_ratio);
        break;
    case ModelType::YOLO11OBBSAHI:
        instance = sahiyolo::load_yolo_11_obb_sahi(model_path,
                                                   names,
                                                   gpu_id,
                                                   confidence_threshold,
                                                   nms_threshold,
                                                   max_batch_size,
                                                   auto_slice,
                                                   slice_width,
                                                   slice_height,
                                                   slice_horizontal_ratio,
                                                   slice_vertical_ratio);
        break;
    default:
        instance = nullptr;
        break;
    }
    return instance;
}