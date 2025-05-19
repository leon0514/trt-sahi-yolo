#include "kernels/kernel_warp.hpp"
#include "trt/sahiyolo/yolo11_pose_sahi.hpp"
namespace sahiyolo
{

bool Yolo11PoseSahiModelImpl::load(const std::string &engine_file,
                                   const std::vector<std::string> &names,
                                   float confidence_threshold,
                                   float nms_threshold,
                                   int gpu_id,
                                   int max_batch_size,
                                   bool auto_slice,
                                   int slice_width,
                                   int slice_height,
                                   double slice_horizontal_ratio,
                                   double slice_vertical_ratio)
{
    trt_       = TensorRT::load(engine_file);
    device_id_ = gpu_id;
    if (trt_ == nullptr)
        return false;

    trt_->print();

    this->slice_ = std::make_shared<slice::SliceImage>();

    this->num_key_point_ = 17;
    this->confidence_threshold_   = confidence_threshold;
    this->nms_threshold_          = nms_threshold;
    this->class_names_            = names;
    this->max_batch_size_         = max_batch_size;
    this->auto_slice_             = auto_slice;
    this->slice_width_            = slice_width;
    this->slice_height_           = slice_height;
    this->slice_horizontal_ratio_ = slice_horizontal_ratio;
    this->slice_vertical_ratio_   = slice_vertical_ratio;

    auto input_dim  = trt_->static_dims(0);
    bbox_head_dims_ = trt_->static_dims(1);

    network_input_width_  = input_dim[3];
    network_input_height_ = input_dim[2];
    isdynamic_model_      = trt_->has_dynamic_dim();

    normalize_   = norm_image::Norm::alpha_beta(1 / 255.0f, 0.0f, norm_image::ChannelType::SwapRB);
    num_classes_ = bbox_head_dims_[2] - 4 - num_key_point_ * 3;
    return true;
}



InferResult Yolo11PoseSahiModelImpl::forwards(const std::vector<cv::Mat> &inputs, void *stream)
{
    assert(inputs.size() == 1);

    if (auto_slice_)
    {
        slice_->autoSlice(tensor::Image(inputs[0].data, inputs[0].cols, inputs[0].rows));
    }
    else
    {
        slice_->slice(tensor::Image(inputs[0].data, inputs[0].cols, inputs[0].rows),
                      slice_width_,
                      slice_height_,
                      slice_horizontal_ratio_,
                      slice_vertical_ratio_,
                      stream);
    }

    int num_image          = slice_->slice_num_h_ * slice_->slice_num_v_;
    this->max_image_boxes_ = single_image_max_boxes_ * num_image;
    auto input_dims        = trt_->static_dims(0);
    int infer_batch_size   = input_dims[0];
    if (infer_batch_size  != num_image)
    {
        if (isdynamic_model_)
        {
            assert(num_image <= max_batch_size_);
            infer_batch_size = num_image;
            input_dims[0]    = num_image;
            if (!trt_->set_run_dims(0, input_dims))
            {
                printf("Fail to set run dims\n");
                return {};
            }
        }
        else
        {
            if (infer_batch_size < num_image)
            {
                printf("When using static shape model, number of images[%d] must be "
                       "less than or equal to the maximum batch[%d].",
                       num_image,
                       infer_batch_size);
                return {};
            }
        }
    }
    adjust_memory(infer_batch_size);

    // 每一张小图的尺寸都是一致的，所以只需要取计算一次仿射矩阵
    affine::LetterBoxMatrix affine_matrix;
    cudaStream_t stream_ = (cudaStream_t)stream;
    compute_affine_matrix(affine_matrix, stream_);
    for (int i = 0; i < num_image; ++i)
    {
        preprocess(i, stream);
    }

    float *bbox_output_device = bbox_predict_.gpu();
#if NV_TENSORRT_MAJOR >= 10
    // yolov5 模型推理
    // TensorRT10需要指定输入输出名字，这里的输入输出分别是images, output0
    std::unordered_map<std::string, const void *> bindings = {{"images", input_buffer_.gpu()},
                                                              {"output0", bbox_output_device}};
    if (!trt_->forward(bindings, stream_))
    {
        printf("Failed to tensorRT forward.\n");
        return {};
    }
#else
    std::vector<void *> bindings{input_buffer_.gpu(), bbox_output_device};
    if (!trt_->forward(bindings, stream_))
    {
        printf("Failed to tensorRT forward.\n");
        return {};
    }
#endif

    int *box_count = image_box_count_.gpu();
    checkRuntime(cudaMemsetAsync(box_count, 0, sizeof(int), stream_));

    for (int ib = 0; ib < num_image; ++ib)
    {
        int start_x                    = slice_->slice_start_point_.cpu()[ib * 2];
        int start_y                    = slice_->slice_start_point_.cpu()[ib * 2 + 1];
        float *boxarray_device         = output_boxarray_.gpu();
        float *affine_matrix_device    = affine_matrix_.gpu();
        float *image_based_bbox_output = bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
        decode_kernel_invoker_v11_pose(image_based_bbox_output,
                                       bbox_head_dims_[1],
                                       num_classes_,
                                       bbox_head_dims_[2],
                                       confidence_threshold_,
                                       nms_threshold_,
                                       affine_matrix_device,
                                       boxarray_device,
                                       box_count,
                                       max_image_boxes_,
                                       num_box_element_,
                                       num_key_point_,
                                       start_x,
                                       start_y,
                                       ib,
                                       stream_);
    }
    checkRuntime(cudaStreamSynchronize(stream_));
    float *boxarray_device = output_boxarray_.gpu();
    fast_nms_kernel_invoker_v11_pose(boxarray_device,
                                     box_count,
                                     max_image_boxes_,
                                     nms_threshold_,
                                     num_box_element_,
                                     num_key_point_,
                                     stream_);

    checkRuntime(cudaMemcpyAsync(output_boxarray_.cpu(),
                                 output_boxarray_.gpu(),
                                 output_boxarray_.gpu_bytes(),
                                 cudaMemcpyDeviceToHost,
                                 stream_));
    checkRuntime(cudaMemcpyAsync(image_box_count_.cpu(),
                                 image_box_count_.gpu(),
                                 image_box_count_.gpu_bytes(),
                                 cudaMemcpyDeviceToHost,
                                 stream_));
    checkRuntime(cudaStreamSynchronize(stream_));

    std::vector<object::PoseResultArray> arrout(1);
    for (int ib = 0; ib < 1; ++ib)
    {
        float *parray = output_boxarray_.cpu();
        int count     = min(max_image_boxes_, *(image_box_count_.cpu()));

        object::PoseResultArray &output = arrout[ib];
        for (int i = 0; i < count; ++i)
        {
            float *pbox  = parray + i * (num_box_element_ + num_key_point_ * 3);
            int label    = pbox[5];
            int keepflag = pbox[6];
            // printf("keepflag : %d\n", keepflag);
            if (keepflag == 1)
            {
                std::vector<object::KeyPoint> points;
                points.reserve(num_key_point_);
                for (int j = 0; j < num_key_point_; j++)
                {
                    float x   = pbox[num_box_element_ + j * 3];
                    float y   = pbox[num_box_element_ + j * 3 + 1];
                    float vis = pbox[num_box_element_ + j * 3 + 2];
                    points.push_back(std::move(object::KeyPoint(x, y, vis)));
                }
                std::string name = class_names_[label];
                object::PoseInstance result_object_box(
                    object::Box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label, name),
                    points);
                output.emplace_back(std::move(result_object_box));
            }
        }
    }
    return arrout;
}

static InferBase *loadraw(const std::string &engine_file,
                          const std::vector<std::string> &names,
                          float confidence_threshold,
                          float nms_threshold,
                          int gpu_id,
                          int max_batch_size,
                          bool auto_slice,
                          int slice_width,
                          int slice_height,
                          double slice_horizontal_ratio,
                          double slice_vertical_ratio)
{
    Yolo11PoseSahiModelImpl *impl = new Yolo11PoseSahiModelImpl();
    if (!impl->load(engine_file,
                    names,
                    confidence_threshold,
                    nms_threshold,
                    gpu_id,
                    max_batch_size,
                    auto_slice,
                    slice_width,
                    slice_height,
                    slice_horizontal_ratio,
                    slice_vertical_ratio))
    {
        delete impl;
        impl = nullptr;
    }
    return impl;
}

std::shared_ptr<InferBase> load_yolo_11_pose_sahi(const std::string &engine_file,
                                                  const std::vector<std::string> &names,
                                                  int gpu_id,
                                                  float confidence_threshold,
                                                  float nms_threshold,
                                                  int max_batch_size,
                                                  bool auto_slice,
                                                  int slice_width,
                                                  int slice_height,
                                                  double slice_horizontal_ratio,
                                                  double slice_vertical_ratio)
{
    try
    {
        checkRuntime(cudaSetDevice(gpu_id));
        return std::shared_ptr<Yolo11PoseSahiModelImpl>((Yolo11PoseSahiModelImpl *)loadraw(engine_file,
                                                                                           names,
                                                                                           confidence_threshold,
                                                                                           nms_threshold,
                                                                                           gpu_id,
                                                                                           max_batch_size,
                                                                                           auto_slice,
                                                                                           slice_width,
                                                                                           slice_height,
                                                                                           slice_horizontal_ratio,
                                                                                           slice_vertical_ratio),
                                                        [](InferBase *impl) { delete impl; });
    }
    catch (const std::exception &ex)
    {
        return nullptr;
    }
}

} // namespace sahiyolo