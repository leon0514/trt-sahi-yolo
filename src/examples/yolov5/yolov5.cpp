#include "trt/infer.hpp"
#include "osd/osd.hpp"
#include "common/object.hpp"

static std::vector<std::string> classes_names = {
    "person",
    "helmet"
};

void run_yolov5()
{
    std::shared_ptr<InferBase> model_ = load("models/helmet.engine",
        ModelType::YOLOV5,
        classes_names,
        0,
        0.5f,
        0.4f,
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
            if constexpr (std::is_same_v<T, std::vector<object::DetectionResultArray>>)
            {
                for (int i = 0; i < batch_size; i++)
                {
                    printf("Batch %d: size : %d\n", i, result[i].size());
                    osd_detection(images[i], result[i]);
                    cv::imwrite("result/yolov5.jpg", images[i]);
                }
                
            }
        },
        det);
}

void run_yolov5_sahi()
{
    std::shared_ptr<InferBase> model_ = load("models/helmet.engine",
        ModelType::YOLOV5SAHI,
        classes_names,
        0,
        0.5f,
        0.4f,
        32,
        true,
        1000,
        1000,
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
            if constexpr (std::is_same_v<T, std::vector<object::DetectionResultArray>>)
            {
                for (int i = 0; i < batch_size; i++)
                {
                    printf("Batch %d: size : %d\n", i, result[i].size());
                    osd_detection(images[i], result[i]);
                    cv::imwrite("result/yolov5sahi.jpg", images[i]);
                }

            }
        },
        det);
}