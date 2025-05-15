#include "trt/infer.hpp"
#include "osd/osd.hpp"
#include "common/object.hpp"

static std::vector<std::string> classes_names = {
    "person"
};

void run_yolo11pose()
{
    std::shared_ptr<InferBase> model_ = load("models/yolo11l-pose.transd.engine",
        ModelType::YOLO11POSE,
        classes_names,
        0,
        0.5f,
        0.45f,
        1,
        false,
        0,
        0,
        0.0,
        0.0);
    cv::Mat image = cv::imread("inference/persons.jpg");
    std::vector<cv::Mat> images = {image};
    auto det = model_->forwards(images);
    std::visit(
        [&images](auto &&result)
        {
            int batch_size = images.size();
            using T        = std::decay_t<decltype(result)>;
            if constexpr (std::is_same_v<T, std::vector<object::PoseResultArray>>)
            {
                for (int i = 0; i < batch_size; i++)
                {
                    printf("Batch %d: size : %d\n", i, result[i].size());
                    osd_pose(images[i], result[i]);
                    cv::imwrite("result/yolo11pose.jpg", images[i]);
                }
                
            }
        },
        det);
}

void run_yolo11pose_sahi()
{
    std::shared_ptr<InferBase> model_ = load("models/yolo11l-pose.transd.engine",
        ModelType::YOLO11POSESAHI,
        classes_names,
        0,
        0.5f,
        0.45f,
        32,
        true,
        640,
        640,
        0.3,
        0.3);
    cv::Mat image = cv::imread("inference/persons.jpg");
    std::vector<cv::Mat> images = {image};
    auto det = model_->forwards(images);
    std::visit(
        [&images](auto &&result)
        {
            int batch_size = images.size();
            using T        = std::decay_t<decltype(result)>;
            if constexpr (std::is_same_v<T, std::vector<object::PoseResultArray>>)
            {
                for (int i = 0; i < batch_size; i++)
                {
                    printf("Batch %d: size : %d\n", i, result[i].size());
                    osd_pose(images[i], result[i]);
                    cv::imwrite("result/yolo11posesahi.jpg", images[i]);
                }

            }
        },
        det);
}