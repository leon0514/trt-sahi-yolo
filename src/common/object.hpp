#ifndef OBJECT_HPP__
#define OBJECT_HPP__

#include "common/check.hpp"
#include "opencv2/opencv.hpp"
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace object
{

struct OBBox
{
    float cx     = 0.0f;
    float cy     = 0.0f;
    float width  = 0.0f;
    float height = 0.0f;
    float angle  = 0.0f;
    float score  = 0.0f;
    int class_id = -1;
    std::string class_name;

    OBBox() = default;
    OBBox(float x, float y, float w, float h, float a, float s, const std::string &name = "")
        : cx(x), cy(y), width(w), height(h), angle(a), score(s), class_id(-1), class_name(name)
    {
    }
    OBBox(float x, float y, float w, float h, float a, float s, int id, const std::string &name = "")
        : cx(x), cy(y), width(w), height(h), angle(a), score(s), class_id(id), class_name(name)
    {
    }
    OBBox(const OBBox &)            = default;
    OBBox &operator=(const OBBox &) = default;

    std::tuple<float, float> left_top() const
    {
        auto p0 = point(0);
        auto p1 = point(1);
        auto p2 = point(2);
        auto p3 = point(3);

        float min_x = std::min({std::get<0>(p0), std::get<0>(p1), std::get<0>(p2), std::get<0>(p3)});
        float min_y = std::min({std::get<1>(p0), std::get<1>(p1), std::get<1>(p2), std::get<1>(p3)});

        return std::make_tuple(min_x, min_y);
    }

    std::tuple<float, float> right_bottom() const
    {
        auto p0 = point(0);
        auto p1 = point(1);
        auto p2 = point(2);
        auto p3 = point(3);

        float max_x = std::max({std::get<0>(p0), std::get<0>(p1), std::get<0>(p2), std::get<0>(p3)});
        float max_y = std::max({std::get<1>(p0), std::get<1>(p1), std::get<1>(p2), std::get<1>(p3)});

        return std::make_tuple(max_x, max_y);
    }

    // id 0 -3
    std::tuple<float, float> point(int id) const
    {
        float dx, dy;
        switch (id % 4)
        {
        case 0:
            dx = -width / 2;
            dy = -height / 2;
            break; // 原左上
        case 1:
            dx = width / 2;
            dy = -height / 2;
            break; // 原右上
        case 2:
            dx = width / 2;
            dy = height / 2;
            break; // 原右下
        case 3:
            dx = -width / 2;
            dy = height / 2;
            break; // 原左下
        default:
            return {cx, cy}; // 无效id返回中心
        }

        // 应用旋转矩阵（绕中心点旋转）
        float cosa      = cos(angle);
        float sina      = sin(angle);
        float rotated_x = dx * cosa - dy * sina;
        float rotated_y = dx * sina + dy * cosa;

        // 转换为全局坐标
        return std::make_tuple(cx + rotated_x, cy + rotated_y);
    };
};

struct Box
{
    float left   = 0.0f;
    float top    = 0.0f;
    float right  = 0.0f;
    float bottom = 0.0f;
    float score  = 0.0f;
    int class_id = -1;
    std::string class_name;

    Box() = default;
    Box(float l, float t, float r, float b, float s, const std::string &name = "")
        : left(l), top(t), right(r), bottom(b), score(s), class_id(-1), class_name(name)
    {
    }
    Box(float l, float t, float r, float b, float s, int id, const std::string &name = "")
        : left(l), top(t), right(r), bottom(b), score(s), class_id(id), class_name(name)
    {
    }

    Box(const Box &)            = default;
    Box &operator=(const Box &) = default;

    float width() const { return right - left; }
    float height() const { return bottom - top; }
    cv::Rect getRect() const { return cv::Rect(cv::Point((int)left, (int)top), cv::Point((int)right, (int)bottom)); }
};

struct KeyPoint
{
    float x     = 0.0f;
    float y     = 0.0f;
    float score = 0.0f;

    cv::Point to_cv_point() const { return cv::Point((int)x, (int)y); }

    KeyPoint() = default;
    KeyPoint(float x, float y, float score) : x(x), y(y), score(score) {}

    KeyPoint(const KeyPoint &)            = default;
    KeyPoint &operator=(const KeyPoint &) = default;
};

struct PoseInstance
{
    Box box;
    std::vector<KeyPoint> keypoints;

    PoseInstance() = default;
    PoseInstance(const Box &b, const std::vector<KeyPoint> &k) : box(b), keypoints(k) {}

    PoseInstance(PoseInstance &&other) noexcept
        : box(std::move(other.box)),            // Move the Box
          keypoints(std::move(other.keypoints)) // Move the vector
    {
    }

    PoseInstance &operator=(PoseInstance &&other) noexcept
    {
        if (this != &other)
        {
            box       = std::move(other.box);       // Move the Box
            keypoints = std::move(other.keypoints); // Move the vector
        }
        return *this;
    }
};

struct SegmentMap
{
    int width = 0, height = 0;     // width % 8 == 0
    unsigned char *data = nullptr; // is width * height memory

    SegmentMap(int width, int height)
    {
        this->width  = width;
        this->height = height;
        checkRuntime(cudaMallocHost(&this->data, width * height));
    }
    virtual ~SegmentMap()
    {
        if (this->data)
        {
            checkRuntime(cudaFreeHost(this->data));
            this->data = nullptr;
        }
        this->width  = 0;
        this->height = 0;
    }

    // 1. Delete Copy Constructor
    SegmentMap(const SegmentMap &) = delete;

    // 2. Delete Copy Assignment Operator
    SegmentMap &operator=(const SegmentMap &) = delete;

    // 3. Move Constructor
    SegmentMap(SegmentMap &&other) noexcept
        : width(std::exchange(other.width, 0)),    // Transfer ownership and reset source
          height(std::exchange(other.height, 0)),  // Transfer ownership and reset source
          data(std::exchange(other.data, nullptr)) // Transfer ownership and reset source
    {
        // The moved-from object 'other' is now in a valid, empty state
    }

    // 4. Move Assignment Operator
    SegmentMap &operator=(SegmentMap &&other) noexcept
    {
        // Prevent self-assignment (though unlikely with &&)
        if (this != &other)
        {
            // Free existing resource first
            if (this->data)
            {
                checkRuntime(cudaFreeHost(this->data));
            }

            // Transfer ownership from 'other'
            width  = std::exchange(other.width, 0);
            height = std::exchange(other.height, 0);
            data   = std::exchange(other.data, nullptr);
        }
        return *this;
    }
};

struct SegmentationInstance
{
    Box box;
    std::shared_ptr<SegmentMap> seg;

    SegmentationInstance() = default;
    SegmentationInstance(const Box &b, std::shared_ptr<SegmentMap> m) : box(b), seg(std::move(m)) {}

    SegmentationInstance(const SegmentationInstance &) = delete;

    SegmentationInstance &operator=(const SegmentationInstance &) = delete;

    SegmentationInstance(SegmentationInstance &&other) noexcept
        : box(std::move(other.box)), seg(std::move(other.seg)) // relies on SegmentMap's move constructor
    {
    }

    SegmentationInstance &operator=(SegmentationInstance &&other) noexcept
    {
        if (this != &other)
        {
            box = std::move(other.box); // Move the Box
            seg = std::move(other.seg); // Move the SegmentMap
        }
        return *this;
    }
};


using DetectionResultArray    = std::vector<Box>;                  // 检测结果数组
using DetectionObbResultArray = std::vector<OBBox>;                // 检测旋转框数组
using SegmentationResultArray = std::vector<SegmentationInstance>; // 分割结果数组
using PoseResultArray         = std::vector<PoseInstance>;         // 姿态估计结果数组

} // namespace object

#endif // OBJECT_HPP__