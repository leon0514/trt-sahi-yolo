#include <sstream>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "opencv2/opencv.hpp"
#include "model/yolo.hpp"

#define UNUSED(expr) do { (void)(expr); } while (0)

using namespace std;

namespace py=pybind11;

namespace pybind11 { namespace detail {
template<>
struct type_caster<cv::Mat>
{
public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    //! 1. cast numpy.ndarray to cv::Mat
    bool load(handle obj, bool)
    {
        array b = reinterpret_borrow<array>(obj);
        buffer_info info = b.request();

        //const int ndims = (int)info.ndim;
        int nh = 1;
        int nw = 1;
        int nc = 1;
        int ndims = info.ndim;
        if(ndims == 2)
	{
            nh = info.shape[0];
            nw = info.shape[1];
        } 
	else if(ndims == 3)
	{
            nh = info.shape[0];
            nw = info.shape[1];
            nc = info.shape[2];
        }
	else
	{
            char msg[64];
            std::sprintf(msg, "Unsupported dim %d, only support 2d, or 3-d", ndims);
            throw std::logic_error(msg);
            return false;
        }

        int dtype;
        if(info.format == format_descriptor<unsigned char>::format())
	{
            dtype = CV_8UC(nc);
        }
	else if (info.format == format_descriptor<int>::format())
	{
            dtype = CV_32SC(nc);
        }
	else if (info.format == format_descriptor<float>::format())
	{
            dtype = CV_32FC(nc);
        }
	else
	{
            throw std::logic_error("Unsupported type, only support uchar, int32, float");
            return false;
        }
        value = cv::Mat(nh, nw, dtype, info.ptr);
        return true;
    }

    //! 2. cast cv::Mat to numpy.ndarray
    static handle cast(const cv::Mat& mat, return_value_policy, handle defval){
        UNUSED(defval);

        std::string format = format_descriptor<unsigned char>::format();
        size_t elemsize = sizeof(unsigned char);
        int nw = mat.cols;
        int nh = mat.rows;
        int nc = mat.channels();
        int depth = mat.depth();
        int type = mat.type();
        int dim = (depth == type)? 2 : 3;

        if(depth == CV_8U)
        {
            format = format_descriptor<unsigned char>::format();
            elemsize = sizeof(unsigned char);
        }
	else if(depth == CV_32S)
        {
            format = format_descriptor<int>::format();
            elemsize = sizeof(int);
        }
	else if(depth == CV_32F)
	{
            format = format_descriptor<float>::format();
            elemsize = sizeof(float);
        }
	else
	{
            throw std::logic_error("Unsupport type, only support uchar, int32, float");
        }

        std::vector<size_t> bufferdim;
        std::vector<size_t> strides;
        if (dim == 2) 
	{
            bufferdim = {(size_t) nh, (size_t) nw};
            strides = {elemsize * (size_t) nw, elemsize};
        } 
	else if (dim == 3) 
	{
            bufferdim = {(size_t) nh, (size_t) nw, (size_t) nc};
            strides = {(size_t) elemsize * nw * nc, (size_t) elemsize * nc, (size_t) elemsize};
        }
        return array(buffer_info( mat.data,  elemsize,  format, dim, bufferdim, strides )).release();
    }
};
}}//! end namespace pybind11::detail

class TrtSahiYolo{
public:
    TrtSahiYolo(std::string model_path, yolo::YoloType yolo_type, int gpu_id, float confidence_threshold, float nms_threshold)
    {
        instance_ = yolo::load(model_path, yolo_type, gpu_id, confidence_threshold, nms_threshold);
    }

    yolo::BoxArray autoSliceForward(const cv::Mat& image)
    {
        return instance_->forward(tensor::cvimg(image));
    }

    yolo::BoxArray manualSliceForward(const cv::Mat& image, int width, int height, float xratio, float yratio)
    {
        return instance_->forward(tensor::cvimg(image), width, height, xratio, yratio);
    }


    bool valid()
    {
        return instance_ != nullptr;
    }

private:
    std::shared_ptr<yolo::Infer> instance_;

};


PYBIND11_MODULE(trtsahiyolo, m){
    py::enum_<yolo::YoloType>(m, "YoloType")
        .value("YOLOV5", yolo::YoloType::YOLOV5)
        .value("YOLOV8", yolo::YoloType::YOLOV8)
        .value("YOLOV11", yolo::YoloType::YOLOV11)
        .value("YOLOV11POSE", yolo::YoloType::YOLOV11POSE)
        .export_values();

    py::class_<yolo::Box>(m, "Box")
        .def_readwrite("left", &yolo::Box::left)
        .def_readwrite("top", &yolo::Box::top)
        .def_readwrite("right", &yolo::Box::right)
        .def_readwrite("bottom", &yolo::Box::bottom)
        .def_readwrite("confidence", &yolo::Box::confidence)
        .def_readwrite("class_label", &yolo::Box::class_label)
        .def_readwrite("pose", &yolo::Box::pose)
        .def("__repr__", [](const yolo::Box &box) {
            std::ostringstream oss;
            oss << "Box(left: " << box.left
                << ", top: " << box.top
                << ", right: " << box.right
                << ", bottom: " << box.bottom
                << ", confidence: " << box.confidence
                << ", id: " << box.class_label
                << ")";
            return oss.str();
        });

    py::class_<TrtSahiYolo>(m, "TrtSahiYolo")
	.def(py::init<string, yolo::YoloType, int, float, float>(), 
        py::arg("model_path"), 
        py::arg("yolo_type"),
        py::arg("gpu_id"), 
        py::arg("confidence_threshold"),
        py::arg("nms_threshold"))
	.def_property_readonly("valid", &TrtSahiYolo::valid)
	.def("autoSliceForward", &TrtSahiYolo::autoSliceForward, py::arg("image"))
	.def("manualSliceForward", &TrtSahiYolo::manualSliceForward, 
			py::arg("image"), 
			py::arg("width"), 
			py::arg("height"), 
			py::arg("xratio"), 
			py::arg("yratio"));
};
