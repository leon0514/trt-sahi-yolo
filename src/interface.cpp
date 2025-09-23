#include <sstream>
#include <iostream>
#include <optional>
#include <vector>
#include <string>
#include <tuple>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // 自动转换 std::vector, std::optional 等
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>    // 用于 cv::Mat 和 numpy 的转换

#include "opencv2/opencv.hpp"
#include "trt/infer.hpp"      // 您的推理引擎头文件
#include "common/object.hpp"         // 您的数据结构定义头文件

#define UNUSED(expr) do { (void)(expr); } while (0)

namespace py = pybind11;

// -------------------------------------------------------------------
// 1. cv::Mat <-> numpy.ndarray 类型转换器 (与您提供的一致)
// -------------------------------------------------------------------
namespace pybind11 { namespace detail {
template<>
struct type_caster<cv::Mat>
{
public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    // 从 numpy.ndarray 转换为 cv::Mat
    bool load(handle obj, bool)
    {
        array b = reinterpret_borrow<array>(obj);
        buffer_info info = b.request();

        int ndims = info.ndim;
        int nh = 1, nw = 1, nc = 1;
        
        if (ndims == 2) {
            nh = info.shape[0];
            nw = info.shape[1];
        } else if (ndims == 3) {
            nh = info.shape[0];
            nw = info.shape[1];
            nc = info.shape[2];
        } else {
            throw std::logic_error("不支持的维度数量，仅支持 2D 或 3D 数组。");
            return false;
        }

        int dtype;
        if (info.format == format_descriptor<unsigned char>::format()) {
            dtype = CV_8UC(nc);
        } else if (info.format == format_descriptor<int>::format()) {
            dtype = CV_32SC(nc);
        } else if (info.format == format_descriptor<float>::format()) {
            dtype = CV_32FC(nc);
        } else {
            throw std::logic_error("不支持的 numpy 数据类型，仅支持 uchar, int32, 和 float32。");
            return false;
        }
        
        value = cv::Mat(nh, nw, dtype, info.ptr);
        return true;
    }

    // 从 cv::Mat 转换为 numpy.ndarray
    static handle cast(const cv::Mat& mat, return_value_policy, handle defval){
        UNUSED(defval);
        std::string format;
        size_t elemsize;
        int depth = mat.depth();

        switch (depth) {
            case CV_8U:  format = format_descriptor<unsigned char>::format(); elemsize = sizeof(unsigned char); break;
            case CV_32S: format = format_descriptor<int>::format();           elemsize = sizeof(int);           break;
            case CV_32F: format = format_descriptor<float>::format();         elemsize = sizeof(float);         break;
            default: throw std::logic_error("不支持的 cv::Mat 深度，仅支持 CV_8U, CV_32S, 和 CV_32F。");
        }

        int nc = mat.channels();
        std::vector<size_t> bufferdim;
        std::vector<size_t> strides;

        if (nc == 1 || mat.dims == 2) { // 2D 数组
            bufferdim = {(size_t)mat.rows, (size_t)mat.cols};
            strides = {elemsize * (size_t)mat.cols, elemsize};
        } else { // 3D 数组
            bufferdim = {(size_t)mat.rows, (size_t)mat.cols, (size_t)nc};
            strides = {elemsize * (size_t)mat.cols * (size_t)nc, elemsize * (size_t)nc, elemsize};
        }

        return array(buffer_info(mat.data, elemsize, format, bufferdim.size(), bufferdim, strides)).release();
    }
};
}} // namespace pybind11::detail


// -------------------------------------------------------------------
// 2. 重构后的 TrtSahi C++ 类
// -------------------------------------------------------------------
class TrtSahi {
public:
    // 构造函数保持不变，它负责通过 trt::load 创建正确的 InferBase 实现
    TrtSahi(
        const std::string &model_path, ModelType model_type,
        const std::vector<std::string> &names, int gpu_id = 0,
        float confidence_threshold = 0.5f, float nms_threshold = 0.45f,
        int max_batch_size = 32, bool auto_slice = true, int slice_width = 640,
        int slice_height = 640, double slice_horizontal_ratio = 0.3,
        double slice_vertical_ratio = 0.3)
    {
        instance_ = load(model_path, model_type, names, gpu_id, confidence_threshold, nms_threshold, max_batch_size, auto_slice, slice_width, slice_height, slice_horizontal_ratio, slice_vertical_ratio);
    }

    // forwards 方法现在变得极其简单！
    // 它只是一个直接的方法调用转发。
    std::vector<object::DetectionBoxArray> forwards(const std::vector<cv::Mat>& images)
    {
        // 直接调用并返回底层实现的结果，不再需要任何转换。
        return instance_->forwards(images);
    }

    bool valid() { return instance_ != nullptr; }

private:
    std::shared_ptr<InferBase> instance_;
};



// -------------------------------------------------------------------
// 3. 定义 Python 模块和所有绑定
// -------------------------------------------------------------------
PYBIND11_MODULE(trtsahi, m) {
    m.doc() = "Python bindings for TensorRT SAHI inference library";

    // --- 绑定您的模型类型 Enum ---
    py::enum_<ModelType>(m, "ModelType")
        .value("YOLOV5", ModelType::YOLOV5)
        .value("YOLO11", ModelType::YOLO11)
        .value("YOLO11POSE", ModelType::YOLO11POSE)
        .value("YOLO11SEG", ModelType::YOLO11SEG)
        .value("YOLO11OBB", ModelType::YOLO11OBB)
        .value("YOLOV5SAHI", ModelType::YOLOV5SAHI)
        .value("YOLO11SAHI", ModelType::YOLO11SAHI)
        .value("YOLO11POSESAHI", ModelType::YOLO11POSESAHI)
        .value("YOLO11SEGSAHI", ModelType::YOLO11SEGSAHI)
        .value("YOLO11OBBSAHI", ModelType::YOLO11OBBSAHI)
        .export_values();
    
    // --- 绑定 object.hpp 中的所有数据结构 ---
    // 这是让 TrtSahi 类能被正确绑定的前提
    
    py::enum_<object::ObjectType>(m, "ObjectType")
        .value("UNKNOW", object::ObjectType::UNKNOW)
        .value("DETECTION", object::ObjectType::DETECTION)
        .value("POSE", object::ObjectType::POSE)
        .value("OBB", object::ObjectType::OBB)
        .value("SEGMENTATION", object::ObjectType::SEGMENTATION)
        // ... (绑定其他枚举值)
        .export_values();

    py::class_<object::Box>(m, "Box")
        .def(py::init<float, float, float, float>(), py::arg("left")=0, py::arg("top")=0, py::arg("right")=0, py::arg("bottom")=0)
        .def_readwrite("left", &object::Box::left)
        .def_readwrite("top", &object::Box::top)
        .def_readwrite("right", &object::Box::right)
        .def_readwrite("bottom", &object::Box::bottom)
        .def("__repr__", [](const object::Box &b) {
            return "<Box l=" + std::to_string(b.left) + ", t=" + std::to_string(b.top) + ", r=" + std::to_string(b.right) + ", b=" + std::to_string(b.bottom) + ">";
        });

    py::class_<object::PosePoint>(m, "PosePoint")
        .def(py::init<float, float, float>(), py::arg("x")=0, py::arg("y")=0, py::arg("vis")=0)
        .def_readwrite("x", &object::PosePoint::x)
        .def_readwrite("y", &object::PosePoint::y)
        .def_readwrite("vis", &object::PosePoint::vis)
        .def("__repr__", [](const object::PosePoint &p) {
            return "<PosePoint x=" + std::to_string(p.x) + ", y=" + std::to_string(p.y) + ", vis=" + std::to_string(p.vis) + ">";
        });

    py::class_<object::Pose>(m, "Pose")
        .def(py::init<>())
        .def_readwrite("points", &object::Pose::points);

    py::class_<object::Obb>(m, "Obb")
        .def(py::init<float, float, float, float, float>(), py::arg("cx")=0, py::arg("cy")=0, py::arg("width")=0, py::arg("height")=0, py::arg("angle")=0)
        .def_readwrite("cx", &object::Obb::cx)
        .def_readwrite("cy", &object::Obb::cy)
        .def_readwrite("width", &object::Obb::w)
        .def_readwrite("height", &object::Obb::h)
        .def_readwrite("angle", &object::Obb::angle);
        
    py::class_<object::Segmentation>(m, "Segmentation")
        .def(py::init<>())
        .def_readwrite("mask", &object::Segmentation::mask);

    // ... (您可以继续绑定 Depth, Track 等其他结构) ...
    
    // 绑定核心的 DetectionBox 结构
    py::class_<object::DetectionBox>(m, "DetectionBox")
        .def(py::init<>())
        .def_readwrite("type", &object::DetectionBox::type)
        .def_readwrite("box", &object::DetectionBox::box)
        .def_readwrite("score", &object::DetectionBox::score)
        .def_readwrite("class_id", &object::DetectionBox::class_id)
        .def_readwrite("class_name", &object::DetectionBox::class_name)
        .def_readwrite("pose", &object::DetectionBox::pose)
        .def_readwrite("obb", &object::DetectionBox::obb)
        .def_readwrite("segmentation", &object::DetectionBox::segmentation)
        .def("__repr__", [](const object::DetectionBox &d) {
            return "<DetectionBox class='" + d.class_name + "' score=" + std::to_string(d.score) + ">";
        });

    // 为 std::vector<DetectionBox> 绑定一个 Python 类型，通常命名为 ...Array
    py::bind_vector<std::vector<object::DetectionBox>>(m, "DetectionBoxArray");

    // --- 最后，绑定 TrtSahi 类 ---
    py::class_<TrtSahi>(m, "TrtSahi")
        .def(py::init<const std::string&, ModelType, const std::vector<std::string>&, int, float, float, int, bool, int, int, double, double>(),
            // 为构造函数参数提供名称，使其在 Python 中可以作为关键字参数使用
            py::arg("model_path"),
            py::arg("model_type"),
            py::arg("names"),
            py::arg("gpu_id") = 0,
            py::arg("confidence_threshold") = 0.5f,
            py::arg("nms_threshold") = 0.45f,
            py::arg("max_batch_size") = 32,
            py::arg("auto_slice") = true,
            py::arg("slice_width") = 640,
            py::arg("slice_height") = 640,
            py::arg("slice_horizontal_ratio") = 0.3,
            py::arg("slice_vertical_ratio") = 0.3)
    .def_property_readonly("valid", &TrtSahi::valid, "检查推理引擎是否有效")
    .def("forwards", &TrtSahi::forwards, py::arg("images"), "对一批图像执行推理，返回检测结果列表的列表");
}