#include "model/yolo.hpp"
#include "common/timer.hpp"
#include "common/image.hpp"
#include "common/position.hpp"

static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) 
{
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) 
    {
        case 0:
            r = v, g = t, b = p;
            break;
        case 1:
            r = q, g = v, b = p;
            break;
        case 2:
            r = p, g = v, b = t;
            break;
        case 3:
            r = p, g = q, b = v;
            break;
        case 4:
            r = t, g = p, b = v;
            break;
        case 5:
            r = v, g = p, b = q;
            break;
        default:
            r = 1, g = 1, b = 1;
            break;
    }
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255),
                        static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) 
{
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

static std::tuple<int, int, int> getFontSize(const std::string& text)
{
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
    return std::make_tuple(textSize.width, textSize.height, baseline);
}

static const char *cocolabels[] = {"person",        "bicycle",      "car",
                                   "motorcycle",    "airplane",     "bus",
                                   "train",         "truck",        "boat",
                                   "traffic light", "fire hydrant", "stop sign",
                                   "parking meter", "bench",        "bird",
                                   "cat",           "dog",          "horse",
                                   "sheep",         "cow",          "elephant",
                                   "bear",          "zebra",        "giraffe",
                                   "backpack",      "umbrella",     "handbag",
                                   "tie",           "suitcase",     "frisbee",
                                   "skis",          "snowboard",    "sports ball",
                                   "kite",          "baseball bat", "baseball glove",
                                   "skateboard",    "surfboard",    "tennis racket",
                                   "bottle",        "wine glass",   "cup",
                                   "fork",          "knife",        "spoon",
                                   "bowl",          "banana",       "apple",
                                   "sandwich",      "orange",       "broccoli",
                                   "carrot",        "hot dog",      "pizza",
                                   "donut",         "cake",         "chair",
                                   "couch",         "potted plant", "bed",
                                   "dining table",  "toilet",       "tv",
                                   "laptop",        "mouse",        "remote",
                                   "keyboard",      "cell phone",   "microwave",
                                   "oven",          "toaster",      "sink",
                                   "refrigerator",  "book",         "clock",
                                   "vase",          "scissors",     "teddy bear",
                                   "hair drier",    "toothbrush"};

void v11SlicedInfer()
{
    cv::Mat image = cv::imread("inference/persons.jpg");
    auto yolo = yolo::load("yolov8n.transd.engine", yolo::YoloType::YOLOV8);
    if (yolo == nullptr) return;
    auto objs = yolo->forward(tensor::cvimg(image));
    printf("objs size : %d\n", objs.size());
    PositionManager<int> pm(getFontSize);
    for (auto &obj : objs) 
    {
        // std::cout << obj << std::endl;
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                    cv::Scalar(b, g, r), 5);
    }
    for (auto &obj : objs) 
    {
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);
        auto name = cocolabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);

        std::tuple<int, int, int, int> box = std::make_tuple((int)obj.left, (int)obj.top, (int)obj.right, (int)obj.bottom);
        int x, y;
        std::tie(x, y) = pm.selectOptimalPosition(box, image.cols, image.rows, caption);
        std::tuple<int, int, int, int> curPos = pm.getCurrentPosition();
        int left, top, right, bottom;
        std::tie(left, top, right, bottom) = curPos;
        cv::rectangle(image, cv::Point(left, top),
                    cv::Point(right, bottom), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(x, y), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    printf("Save result to result/v11SlicedInfer.jpg, %d objects\n", (int)objs.size());
    cv::imwrite("result/v11SlicedInfer.jpg", image);

}

void v11NoSlicedInfer()
{
    cv::Mat image = cv::imread("inference/persons.jpg");
    auto yolo = yolo::load("yolov8n.transd.engine", yolo::YoloType::YOLOV11);
    if (yolo == nullptr) return;
    auto objs = yolo->forward(tensor::cvimg(image), image.cols, image.rows, 0.0f, 0.0f);
    printf("objs size : %d\n", objs.size());
    PositionManager<int> pm(getFontSize);
    for (auto &obj : objs) 
    {
        // obj.dump();
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                    cv::Scalar(b, g, r), 5);
    }
    for (auto &obj : objs) 
    {
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);        
        auto name = cocolabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);
        std::tuple<int, int, int, int> box = std::make_tuple((int)obj.left, (int)obj.top, (int)obj.right, (int)obj.bottom);
        int x, y;
        std::tie(x, y) = pm.selectOptimalPosition(box, image.cols, image.rows, caption);
        std::tuple<int, int, int, int> curPos = pm.getCurrentPosition();
        int left, top, right, bottom;
        std::tie(left, top, right, bottom) = curPos;
        cv::rectangle(image, cv::Point(left, top),
                    cv::Point(right, bottom), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(x, y), 0, 1, cv::Scalar::all(0), 2, 16);
        
    }
    printf("Save result to result/v11NoSlicedInfer.jpg, %d objects\n", (int)objs.size());
    cv::imwrite("result/v11NoSlicedInfer.jpg", image);

}

void v5SlicedInfer()
{
    cv::Mat image = cv::imread("inference/persons.jpg");
    auto yolo = yolo::load("helmetv5.engine", yolo::YoloType::YOLOV5);
    if (yolo == nullptr) return;
    auto objs = yolo->forward(tensor::cvimg(image));
    printf("objs size : %d\n", objs.size());
    PositionManager<int> pm(getFontSize);
    for (auto &obj : objs) 
    {
        // std::cout << obj << std::endl;
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                    cv::Scalar(b, g, r), 5);
    }
    for (auto &obj : objs) 
    {
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);
        auto name = cocolabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);

        std::tuple<int, int, int, int> box = std::make_tuple((int)obj.left, (int)obj.top, (int)obj.right, (int)obj.bottom);
        int x, y;
        std::tie(x, y) = pm.selectOptimalPosition(box, image.cols, image.rows, caption);
        std::tuple<int, int, int, int> curPos = pm.getCurrentPosition();
        int left, top, right, bottom;
        std::tie(left, top, right, bottom) = curPos;
        cv::rectangle(image, cv::Point(left, top),
                    cv::Point(right, bottom), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(x, y), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    printf("Save result to result/v5SlicedInfer.jpg, %d objects\n", (int)objs.size());
    cv::imwrite("result/v5SlicedInfer.jpg", image);

}

void v5NoSlicedInfer()
{
    cv::Mat image = cv::imread("inference/persons.jpg");
    auto yolo = yolo::load("helmetv5.engine", yolo::YoloType::YOLOV5);
    if (yolo == nullptr) return;
    auto objs = yolo->forward(tensor::cvimg(image), image.cols, image.rows, 0.0f, 0.0f);
    printf("objs size : %d\n", objs.size());
    PositionManager<int> pm(getFontSize);
    for (auto &obj : objs) 
    {
        // obj.dump();
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                    cv::Scalar(b, g, r), 5);
    }
    for (auto &obj : objs) 
    {
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);        
        auto name = cocolabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);
        std::tuple<int, int, int, int> box = std::make_tuple((int)obj.left, (int)obj.top, (int)obj.right, (int)obj.bottom);
        int x, y;
        std::tie(x, y) = pm.selectOptimalPosition(box, image.cols, image.rows, caption);
        std::tuple<int, int, int, int> curPos = pm.getCurrentPosition();
        int left, top, right, bottom;
        std::tie(left, top, right, bottom) = curPos;
        cv::rectangle(image, cv::Point(left, top),
                    cv::Point(right, bottom), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(x, y), 0, 1, cv::Scalar::all(0), 2, 16);
        
    }
    printf("Save result to result/v11NoSlicedInfer.jpg, %d objects\n", (int)objs.size());
    cv::imwrite("result/v11NoSlicedInfer.jpg", image);

}


void yolo11poseSlicedInfer()
{
    const std::vector<std::pair<int, int>> coco_pairs = {
        {0, 1}, {0, 2}, {0, 11}, {0, 12}, {1, 3}, {2, 4},
        {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
        {11, 12}, {5, 11}, {6, 12},
        {11, 13}, {13, 15}, {12, 14}, {14, 16}
    };


    cv::Mat image = cv::imread("inference/persons.jpg");
    auto yolo = yolo::load("yolo11s-pose.transd.engine", yolo::YoloType::YOLOV11POSE);
    if (yolo == nullptr) return;
    // auto objs = yolo->forward(tensor::cvimg(image), image.cols, image.rows, 0.0f, 0.0f);
    auto objs = yolo->forward(tensor::cvimg(image));
    printf("objs size : %d\n", objs.size());
    PositionManager<int> pm(getFontSize);
    for (auto &obj : objs) 
    {
        // obj.dump();
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
            cv::Scalar(b, g, r), 5);

        for (const auto& point : obj.pose)
        {
            int x = (int)point.x;
            int y = (int)point.y;
            cv::circle(image, cv::Point(x, y), 6, cv::Scalar(b, g, r), -1);
        }
        for (const auto& pair : coco_pairs) 
        {
            int startIdx = pair.first;
            int endIdx = pair.second;

            if (startIdx < obj.pose.size() && endIdx < obj.pose.size()) 
            {
                int x1 = (int)obj.pose[startIdx].x;
                int y1 = (int)obj.pose[startIdx].y;
                int x2 = (int)obj.pose[endIdx].x;
                int y2 = (int)obj.pose[endIdx].y;

                cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
            }
        }
    }
    
    printf("Save result to result/yolo11poseSlicedInfer.jpg, %d objects\n", (int)objs.size());
    cv::imwrite("result/yolo11poseSlicedInfer.jpg", image);

}

void yolo11poseNoSlicedInfer()
{
    const std::vector<std::pair<int, int>> coco_pairs = {
        {0, 1}, {0, 2}, {0, 11}, {0, 12}, {1, 3}, {2, 4},
        {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
        {11, 12}, {5, 11}, {6, 12},
        {11, 13}, {13, 15}, {12, 14}, {14, 16}
    };


    cv::Mat image = cv::imread("inference/persons.jpg");
    auto yolo = yolo::load("yolo11s-pose.transd.engine", yolo::YoloType::YOLOV11POSE);
    if (yolo == nullptr) return;
    auto objs = yolo->forward(tensor::cvimg(image), image.cols, image.rows, 0.0f, 0.0f);
    // auto objs = yolo->forward(tensor::cvimg(image));
    printf("objs size : %d\n", objs.size());
    PositionManager<int> pm(getFontSize);
    for (auto &obj : objs) 
    {
        // obj.dump();
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
            cv::Scalar(b, g, r), 5);

        for (const auto& point : obj.pose)
        {
            int x = (int)point.x;
            int y = (int)point.y;
            cv::circle(image, cv::Point(x, y), 6, cv::Scalar(b, g, r), -1);
        }
        for (const auto& pair : coco_pairs) 
        {
            int startIdx = pair.first;
            int endIdx = pair.second;

            if (startIdx < obj.pose.size() && endIdx < obj.pose.size()) 
            {
                int x1 = (int)obj.pose[startIdx].x;
                int y1 = (int)obj.pose[startIdx].y;
                int x2 = (int)obj.pose[endIdx].x;
                int y2 = (int)obj.pose[endIdx].y;

                cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
            }
        }
    }
    
    printf("Save result to result/yolo11poseNoSlicedInfer.jpg, %d objects\n", (int)objs.size());
    cv::imwrite("result/yolo11poseNoSlicedInfer.jpg", image);

}


