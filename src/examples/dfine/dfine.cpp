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

void run_dfine()
{
    std::shared_ptr<InferBase> model_ = load("models/engine/dfine_l_obj2coco.engine",
        ModelType::DFINE,
        classes_names,
        0,
        0.5f,
        0.45f,
        16,
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
                    cv::imwrite("result/dfine.jpg", images[i]);
                }
                
            }
        },
        det);
}

void run_dfine_sahi()
{
    std::shared_ptr<InferBase> model_ = load("models/engine/dfine_l_obj2coco.engine",
        ModelType::DFINESAHI,
        classes_names,
        0,
        0.5f,
        0.45f,
        16,
        true,
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
                    cv::imwrite("result/dfinesahi.jpg", images[i]);
                }
                
            }
        },
        det);
}