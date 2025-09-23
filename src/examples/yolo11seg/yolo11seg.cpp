#include "trt/infer.hpp"
#include "osd/osd.hpp"
#include "common/object.hpp"

static std::vector<std::string> classes_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

void run_yolo11seg()
{
    std::shared_ptr<InferBase> model_ = load("models/engine/yoloe-v8l-seg.transd.engine",
        ModelType::YOLO11SEG,
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
    for (int i = 0; i < images.size(); i++)
    {
        printf("Batch %d: size : %d\n", i, det[i].size());
        osd(images[i], det[i]);
        cv::imwrite("result/run_yolo11seg.jpg", images[i]);
    }
}

void run_yolo11seg_sahi()
{
    std::shared_ptr<InferBase> model_ = load("models/engine/yoloe-v8l-seg.transd.engine",
        ModelType::YOLO11SEGSAHI,
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
    for (int i = 0; i < images.size(); i++)
    {
        printf("Batch %d: size : %d\n", i, det[i].size());
        osd(images[i], det[i]);
        cv::imwrite("result/run_yolo11seg_sahi.jpg", images[i]);
    }
}