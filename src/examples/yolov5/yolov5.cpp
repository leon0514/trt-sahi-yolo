#include "trt/infer.hpp"
#include "osd/osd.hpp"
#include "common/object.hpp"

static std::vector<std::string> classes_names = {
    "person",
    "helmet"
};

void run_yolov5()
{
    std::shared_ptr<InferBase> model_ = load("models/engine/helmet.engine",
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
    for (int i = 0; i < images.size(); i++)
    {
        printf("Batch %d: size : %d\n", i, det[i].size());
        osd(images[i], det[i]);
        cv::imwrite("result/run_yolov5.jpg", images[i]);
    }
}

void run_yolov5_sahi()
{
    std::shared_ptr<InferBase> model_ = load("models/engine/helmet.engine",
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
    for (int i = 0; i < images.size(); i++)
    {
        printf("Batch %d: size : %d\n", i, det[i].size());
        osd(images[i], det[i]);
        cv::imwrite("result/run_yolov5_sahi.jpg", images[i]);
    }
}