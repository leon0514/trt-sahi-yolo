#include <sstream>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "opencv2/opencv.hpp"
#include "trt/infer.hpp"

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
            throw std::logic_error("Unsupported type, only support uchar, int32, float");
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

struct DetResult
{
    object::Box box;
    std::vector<object::KeyPoint> keypoints;
    cv::Mat seg;
    object::OBBox obb;
};

class TrtSahi{
public:
    TrtSahi(
        const std::string &model_path,
        ModelType model_type,
        const std::vector<std::string> &names,
        int gpu_id                    = 0,
        float confidence_threshold    = 0.5f,
        float nms_threshold           = 0.45f,
        int max_batch_size            = 32,
        bool auto_slice               = true,
        int slice_width               = 640,
        int slice_height              = 640,
        double slice_horizontal_ratio = 0.3,
        double slice_vertical_ratio   = 0.3)
    {
        instance_ = load(model_path, model_type, names, gpu_id, confidence_threshold, nms_threshold, max_batch_size, auto_slice, slice_width, slice_height, slice_horizontal_ratio, slice_vertical_ratio);
    }

    std::vector<std::vector<DetResult>> forwards(const std::vector<cv::Mat>& images)
    {
        InferResult variant_batch_results = instance_->forwards(images);

        std::vector<std::vector<DetResult>> final_det_results;

        std::visit(
            // Lambda to process the actual data from the variant
            [&images, &final_det_results](auto&& arg_batch_results) 
            {
                // arg_batch_results is one of the types in InferResult, e.g., std::vector<object::DetectionObbResultArray>&
                int num_images_in_batch = images.size();
                
                // Resize the outer vector to hold results for each image
                final_det_results.resize(num_images_in_batch);

                // Type of the current batch result (e.g., std::vector<object::PoseResultArray>)
                using CurrentBatchResultType = std::decay_t<decltype(arg_batch_results)>;

                if constexpr (std::is_same_v<CurrentBatchResultType, std::vector<object::PoseResultArray>>)
                {
                    // arg_batch_results is std::vector<object::PoseResultArray>
                    // Each element arg_batch_results[i] is an object::PoseResultArray (i.e., std::vector<object::PoseInstance>)
                    for (int i = 0; i < num_images_in_batch; ++i)
                    {
                        if (i >= arg_batch_results.size()) continue; // Safety check

                        for (const auto& pose_instance : arg_batch_results[i]) // pose_instance is object::PoseInstance
                        {
                            DetResult det_item;
                            det_item.box = pose_instance.box;
                            det_item.keypoints = pose_instance.keypoints;
                            final_det_results[i].push_back(det_item);
                        }
                    }
                }
                else if constexpr (std::is_same_v<CurrentBatchResultType, std::vector<object::DetectionResultArray>>)
                {
                    // arg_batch_results is std::vector<object::DetectionResultArray>
                    // arg_batch_results[i] is object::DetectionResultArray (i.e., std::vector<object::Box>)
                    for (int i = 0; i < num_images_in_batch; ++i)
                    {
                        if (i >= arg_batch_results.size()) continue;

                        for (const auto& box_instance : arg_batch_results[i]) // box_instance is object::Box
                        {
                            DetResult det_item;
                            det_item.box = box_instance;
                            final_det_results[i].push_back(det_item);
                        }
                    }
                }
                else if constexpr (std::is_same_v<CurrentBatchResultType, std::vector<object::DetectionObbResultArray>>)
                {
                    // arg_batch_results is std::vector<object::DetectionObbResultArray>
                    // arg_batch_results[i] is object::DetectionObbResultArray (i.e., std::vector<object::OBBox>)
                    for (int i = 0; i < num_images_in_batch; ++i)
                    {
                        if (i >= arg_batch_results.size()) continue;

                       
                        for (const auto& obb_instance : arg_batch_results[i]) // obb_instance is object::OBBox
                        {
                            DetResult det_item;
                            det_item.obb = obb_instance;
                            final_det_results[i].push_back(det_item);
                        }
                    }
                }
                else if constexpr (std::is_same_v<CurrentBatchResultType, std::vector<object::SegmentationResultArray>>)
                {
                    // arg_batch_results is std::vector<object::SegmentationResultArray>
                    // arg_batch_results[i] is object::SegmentationResultArray (i.e., std::vector<object::SegmentationInstance>)
                    for (int i = 0; i < num_images_in_batch; ++i)
                    {
                        if (i >= arg_batch_results.size()) continue;

                        for (const auto& seg_instance : arg_batch_results[i]) // seg_instance is object::SegmentationInstance
                        {
                            DetResult det_item;
                            det_item.box = seg_instance.box; // Populate bounding box

                            // Convert object::SegmentMap to cv::Mat for DetResult::seg
                            if (seg_instance.seg && seg_instance.seg->data && seg_instance.seg->width > 0 && seg_instance.seg->height > 0)
                            {
                                // Create a cv::Mat wrapper. Assuming CV_8UC1 for mask data.
                                cv::Mat mask_wrapper_mat(seg_instance.seg->height,
                                                         seg_instance.seg->width,
                                                         CV_8UC1, // unsigned char, 1 channel
                                                         seg_instance.seg->data);
                                
                                // IMPORTANT: Clone the data. The mask_wrapper_mat only points to
                                // SegmentMap's data. Cloning creates a new cv::Mat with its own copy
                                // of the data, ensuring its lifetime is independent of SegmentMap.
                                det_item.seg = mask_wrapper_mat.clone();
                            }
                            // else: det_item.seg will be an empty cv::Mat by default.
                            final_det_results[i].push_back(det_item);
                        }
                    }
                }
                // else { /* Handle other types or std::monostate if present in InferResult */ }
            },
            variant_batch_results
        );

        return final_det_results;
    }

    bool valid()
    {
        return instance_ != nullptr;
    }

private:
    std::shared_ptr<InferBase> instance_;

};


PYBIND11_MODULE(trtsahi, m){
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

    py::class_<object::Box>(m, "Box")
        .def_readwrite("left", &object::Box::left)
        .def_readwrite("top", &object::Box::top)
        .def_readwrite("right", &object::Box::right)
        .def_readwrite("bottom", &object::Box::bottom)
        .def_readwrite("score", &object::Box::score)
        .def_readwrite("class_name", &object::Box::class_name)
        .def("__repr__", [](const object::Box &box) {
            std::ostringstream oss;
            oss << "Box(left: " << box.left
                << ", top: " << box.top
                << ", right: " << box.right
                << ", bottom: " << box.bottom
                << ", score: " << box.score
                << ", class_name: " << box.class_name
                << ")";
            return oss.str();
        });
    py::class_<object::KeyPoint>(m, "KeyPoint")
        .def_readwrite("x", &object::KeyPoint::x)
        .def_readwrite("y", &object::KeyPoint::y)
        .def_readwrite("vis", &object::KeyPoint::score)
        .def("__repr__", [](const object::KeyPoint &Key_point) {
            std::ostringstream oss;
            oss << "KeyPoint(x: " << Key_point.x
                << ", y: " << Key_point.y
                << ", vis: " << Key_point.score
                << ")";
            return oss.str();
        });
    py::class_<object::OBBox>(m, "OBBox")
        .def_readwrite("cx", &object::OBBox::cx)
        .def_readwrite("cy", &object::OBBox::cy)
        .def_readwrite("width", &object::OBBox::width)
        .def_readwrite("height", &object::OBBox::height)
        .def_readwrite("angle", &object::OBBox::angle)
        .def_readwrite("score", &object::OBBox::score)
        .def_readwrite("class_name", &object::OBBox::class_name)
        .def("__repr__", [](const object::OBBox &obbox) {
            std::ostringstream oss;
            oss << "OBBox(cx: " << obbox.cx
                << ", cy: " << obbox.cy
                << ", width: " << obbox.width
                << ", height: " << obbox.height
                << ", angle: " << obbox.angle
                << ", score: " << obbox.score
                << ", class_name: " << obbox.class_name
                << ")";
            return oss.str();
        });
    py::class_<DetResult>(m, "DetResult")
        .def_readwrite("box", &DetResult::box)
        .def_readwrite("keypoints", &DetResult::keypoints)
        .def_readwrite("seg", &DetResult::seg)
        .def_readwrite("obb", &DetResult::obb);

    py::class_<TrtSahi>(m, "TrtSahi")
        .def(py::init<string, ModelType, vector<string>, int, float, float, int, bool, int, int, double, double>(),
            py::arg("model_path"),
            py::arg("model_type"),
            py::arg("names"),
            py::arg("gpu_id"),
            py::arg("confidence_threshold"),
            py::arg("nms_threshold"),
            py::arg("max_batch_size"),
            py::arg("auto_slice"),
            py::arg("slice_width"),
            py::arg("slice_height"),
            py::arg("slice_horizontal_ratio"),
            py::arg("slice_vertical_ratio"))
    .def_property_readonly("valid", &TrtSahi::valid)
    .def("forwards", &TrtSahi::forwards, py::arg("images"));
};
