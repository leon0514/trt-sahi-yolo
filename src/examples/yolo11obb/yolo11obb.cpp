#include "trt/infer.hpp"
#include "osd/osd.hpp"
#include "common/object.hpp"

static std::vector<std::string> classes_names = {
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
    "container crane"
};

void run_yolo11obb()
{
    std::shared_ptr<InferBase> model_ = load("models/engine/yolo11m-obb.transd.engine",
        ModelType::YOLO11OBB,
        classes_names,
        0,
        0.7f,
        0.45f,
        1,
        false,
        0,
        0,
        0.0,
        0.0);
    cv::Mat image = cv::imread("inference/car.jpg");
    std::vector<cv::Mat> images = {image};
    auto det = model_->forwards(images);
    std::visit(
        [&images](auto &&result)
        {
            int batch_size = images.size();
            using T        = std::decay_t<decltype(result)>;
            if constexpr (std::is_same_v<T, std::vector<object::DetectionObbResultArray>>)
            {
                for (int i = 0; i < batch_size; i++)
                {
                    printf("Batch %d: size : %d\n", i, result[i].size());
                    osd_obb(images[i], result[i]);
                    cv::imwrite("result/yolo11obb.jpg", images[i]);
                }
                
            }
        },
        det);
}

void run_yolo11obb_sahi()
{
    std::shared_ptr<InferBase> model_ = load("models/engine/yolo11m-obb.transd.engine",
        ModelType::YOLO11OBBSAHI,
        classes_names,
        0,
        0.7f,
        0.45f,
        32,
        true,
        640,
        640,
        0.3,
        0.3);
    cv::Mat image = cv::imread("inference/car.jpg");
    std::vector<cv::Mat> images = {image};
    auto det = model_->forwards(images);
    std::visit(
        [&images](auto &&result)
        {
            int batch_size = images.size();
            using T        = std::decay_t<decltype(result)>;
            if constexpr (std::is_same_v<T, std::vector<object::DetectionObbResultArray>>)
            {
                for (int i = 0; i < batch_size; i++)
                {
                    printf("Batch %d: size : %d\n", i, result[i].size());
                    osd_obb(images[i], result[i]);
                    cv::imwrite("result/yolo11obbsahi.jpg", images[i]);
                }

            }
        },
        det);
}