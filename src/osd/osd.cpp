#include "osd/osd.hpp"
#include "common/format.hpp"
#include "osd/position.hpp"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <tuple>

const std::vector<std::pair<int, int>> coco_pairs = {{0, 1},
                                                     {0, 2},
                                                     {0, 11},
                                                     {0, 12},
                                                     {1, 3},
                                                     {2, 4},
                                                     {5, 6},
                                                     {5, 7},
                                                     {7, 9},
                                                     {6, 8},
                                                     {8, 10},
                                                     {11, 12},
                                                     {5, 11},
                                                     {6, 12},
                                                     {11, 13},
                                                     {13, 15},
                                                     {12, 14},
                                                     {14, 16}};

static std::tuple<int, int, int> getFontSize(const std::string &text)
{
    int baseline      = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);

    return std::make_tuple(textSize.width, textSize.height, baseline);
}

static void
overlay_mask(cv::Mat &image, const cv::Mat &smallMask, int roiX, int roiY, const cv::Scalar &color, double alpha)
{
    if (image.empty() || smallMask.empty() || image.type() != CV_8UC3 || smallMask.type() != CV_8UC1)
    {
        return;
    }
    alpha = std::max(0.0, std::min(1.0, alpha));

    cv::Rect roiRect(roiX, roiY, smallMask.cols, smallMask.rows);

    cv::Rect imageRect(0, 0, image.cols, image.rows);
    cv::Rect intersectionRect = roiRect & imageRect; // 使用 & 操作符计算交集

    if (intersectionRect.width <= 0 || intersectionRect.height <= 0)
    {
        return;
    }

    cv::Mat originalROI = image(intersectionRect); // ROI 指向 image 的数据

    int maskStartX = intersectionRect.x - roiX;
    int maskStartY = intersectionRect.y - roiY;
    cv::Rect maskIntersectionRect(maskStartX, maskStartY, intersectionRect.width, intersectionRect.height);
    cv::Mat smallMaskROI = smallMask(maskIntersectionRect);

    cv::Mat colorPatchROI(intersectionRect.size(), image.type(), color);

    cv::Mat tempColoredROI = originalROI.clone(); // 需要一个临时区域进行覆盖
    colorPatchROI.copyTo(tempColoredROI, smallMaskROI);

    cv::addWeighted(originalROI, 1.0 - alpha, tempColoredROI, alpha, 0.0, originalROI);
}

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
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id)
{
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

static void osd_box(cv::Mat &image, const object::Box &box, PositionManager<float> &pm)
{
    int id     = box.class_id;
    auto color = random_color(id);
    cv::Scalar bgr_color(std::get<0>(color), std::get<1>(color), std::get<2>(color));
    cv::rectangle(image, box.getRect(), bgr_color, 2);
    int x, y;
    std::string text = fmt::str_format("%s %.2f", box.class_name.c_str(), box.score);
    std::tie(x, y)   = pm.selectOptimalPosition(std::make_tuple(box.left, box.top, box.right, box.bottom),
                                              image.cols,
                                              image.rows,
                                              text);
    cv::putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 1.0, bgr_color, 2);
}

static void osd_obbox(cv::Mat &image, const object::OBBox &box)
{
    int id     = box.class_id;
    auto color = random_color(id);
    cv::Scalar bgr_color(std::get<0>(color), std::get<1>(color), std::get<2>(color));

    // Draw the oriented bounding box
    std::vector<cv::Point> vertices(4);
    for (int i = 0; i < 4; ++i)
    {
        float x, y;
        std::tie(x, y) = box.point(i);
        vertices[i]    = cv::Point(static_cast<int>(x), static_cast<int>(y));
    }
    cv::polylines(image, std::vector<std::vector<cv::Point>>{vertices}, true, bgr_color, 2);

    // Draw the label and score directly above the box
    std::string text = fmt::str_format("%s %.2f", box.class_name.c_str(), box.score);
    float left, top;
    std::tie(left, top) = box.left_top();
    // Find center point
    cv::Point center = vertices[0];
    for (int i = 1; i < 4; ++i)
    {
        center += vertices[i];
    }
    center.x /= 4;
    center.y /= 4;

    // Draw text at center position
    // cv::putText(image, text, cv::Point(center.x - 20, center.y), cv::FONT_HERSHEY_SIMPLEX, 1.0, bgr_color, 2);
}

void osd_detection(cv::Mat &image, const object::DetectionResultArray &detection_results)
{
    PositionManager<float> pm(getFontSize);
    for (const auto &box : detection_results)
    {
        osd_box(image, box, pm);
    }
}

void osd_obb(cv::Mat &image, const object::DetectionObbResultArray &detection_obb_results)
{
    for (const auto &obbox : detection_obb_results)
    {
        osd_obbox(image, obbox);
    }
}

void osd_pose(cv::Mat &image, const object::PoseResultArray &pose_results)
{
    PositionManager<float> pm(getFontSize);
    for (const auto &pose : pose_results)
    {
        int id     = pose.box.class_id;
        auto color = random_color(id);
        cv::Scalar bgr_color(std::get<0>(color), std::get<1>(color), std::get<2>(color));
        osd_box(image, pose.box, pm);
        for (const auto &pair : coco_pairs)
        {
            auto first  = pose.keypoints[pair.first].to_cv_point();
            auto second = pose.keypoints[pair.second].to_cv_point();
            cv::line(image, first, second, bgr_color, 2);
        }
        for (int i = 0; i < pose.keypoints.size(); ++i)
        {
            cv::circle(image, pose.keypoints[i].to_cv_point(), 3, bgr_color, -1);
        }
    }
}

void osd_segmentation(cv::Mat &image, const object::SegmentationResultArray &segmentation_results)
{
    PositionManager<float> pm(getFontSize);
    for (const auto &segment : segmentation_results)
    {
        osd_box(image, segment.box, pm);
        int id     = segment.box.class_id;
        auto color = random_color(id);
        cv::Scalar bgr_color(std::get<0>(color), std::get<1>(color), std::get<2>(color));
        if (segment.seg == nullptr)
        {
            continue;
        }
        cv::Mat mask(segment.seg->height, segment.seg->width, CV_8UC1, segment.seg->data);
        overlay_mask(image, mask, segment.box.left, segment.box.top, bgr_color, 0.6);
    }
}

