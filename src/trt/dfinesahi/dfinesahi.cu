#include "trt/dfinesahi/dfinesahi.hpp"

namespace dfinesahi
{

bool DFineSahiModelImpl::load(const std::string &engine_file,
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

    label_head_dims_ = trt_->static_dims(2);
    box_head_dims_ = trt_->static_dims(3);
    score_head_dims_ = trt_->static_dims(4);

    network_input_width_  = input_dim[3];
    network_input_height_ = input_dim[2];
    isdynamic_model_      = trt_->has_dynamic_dim();

    normalize_   = norm_image::Norm::alpha_beta(1 / 255.0f, 0.0f, norm_image::ChannelType::SwapRB);

    return true;
}


InferResult DFineSahiModelImpl::forwards(const std::vector<cv::Mat> &inputs, void *stream)
{
    assert(inputs.size() == 1);
    if (auto_slice_)
    {
        slice_->autoSlice(tensor::Image(inputs[0].data, inputs[0].cols, inputs[0].rows), stream);
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

    int num_image  = slice_->slice_num_h_ * slice_->slice_num_v_;
    assert(num_image <= max_batch_size_);
    auto input_dims0      = trt_->static_dims(0);
    int infer_batch_size = input_dims0[0];
    auto input_dims1      = trt_->static_dims(1);

    if (infer_batch_size != num_image)
    {
        if (isdynamic_model_)
        {
            infer_batch_size = num_image;
            input_dims0[0]    = num_image;
            input_dims1[0]    = num_image;
            if (!trt_->set_run_dims(0, input_dims0))
            {
                printf("Fail to set run dims\n");
                return {};
            }
            if (!trt_->set_run_dims(1, input_dims1))
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
                       "less than or equal to the maximum batch[%d].\n",
                       num_image,
                       infer_batch_size);
                return {};
            }
        }
    }

    adjust_memory(infer_batch_size);
    cudaStream_t stream_ = (cudaStream_t)stream;
    affine::ResizeMatrix affine_matrix;
    compute_affine_matrix(affine_matrix, stream_);
    for (int i = 0; i < num_image; ++i)
    {
        preprocess(i, stream);
    }
    checkRuntime(cudaMemcpyAsync(input_buffer_orig_target_size_.gpu(),
                                 input_buffer_orig_target_size_.cpu(),
                                 input_buffer_orig_target_size_.gpu_bytes(), // 或者直接用 input_buffer_orig_target_size_.cpu_bytes()
                                 cudaMemcpyHostToDevice,
                                 stream_));

    #ifdef NV_TENSORRT_MAJOR >= 10
        std::unordered_map<std::string, const void *> bindings = {
            {"images", input_buffer_image_.gpu()},
            {"orig_target_sizes", input_buffer_orig_target_size_.gpu()},
            {"labels", output_labels_.gpu()},
            {"boxes", output_boxes_.gpu()},
            {"scores", output_scores_.gpu()}};
        if (!trt_->forward(bindings, stream_))
        {
            printf("Failed to tensorRT forward.\n");
            return {};
        }
    #else
        std::vector<void *> bindings{
            input_buffer_image_.gpu(), 
            input_buffer_orig_target_size_.gpu(),
            output_scores_.gpu(),
            output_labels_.gpu(),
            output_boxes_.gpu()};
        if (!trt_->forward(bindings, stream_))
        {
            printf("Failed to tensorRT forward.\n");
            return {};
        }
    #endif

    checkRuntime(cudaMemcpyAsync(start_.gpu(),
                                 slice_->slice_start_point_.cpu(),
                                 slice_->slice_start_point_.cpu_bytes(), // 或者直接用 input_buffer_orig_target_size_.cpu_bytes()
                                 cudaMemcpyHostToDevice,
                                 stream_));

    int *box_count = image_box_count_.gpu();
    checkRuntime(cudaMemsetAsync(box_count, 0, sizeof(int), stream_));
    

    for (int ib = 0; ib < num_image; ++ib)
    {
        int start_x                    = slice_->slice_start_point_.cpu()[ib * 2];
        int start_y                    = slice_->slice_start_point_.cpu()[ib * 2 + 1];
        float *boxarray_device         = result_.gpu();
#if NV_TENSORRT_MAJOR >= 10
        int64_t* output_labels_device  = output_labels_.gpu() + ib * (box_head_dims_[1]);
#else
        int32_t* output_labels_device  = output_labels_.gpu() + ib * (box_head_dims_[1]);
#endif
        float* output_scores_device    = output_scores_.gpu() + ib * (box_head_dims_[1]);
        float* output_boxes_device     = output_boxes_.gpu()  + ib * (box_head_dims_[1] * box_head_dims_[2]);

        decode_dfine_plan(
            output_labels_device,
            output_scores_device,
            output_boxes_device,
            box_head_dims_[1],
            confidence_threshold_,
            box_count,
            start_x,
            start_y,
            boxarray_device,
            max_image_boxes_,
            num_box_element_,
            stream_);
    }


    fast_nms_kernel_invoker(result_.gpu(), box_count, max_image_boxes_, nms_threshold_, num_box_element_, stream_);

    checkRuntime(cudaMemcpyAsync(result_.cpu(), result_.gpu(), result_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
    checkRuntime(cudaMemcpyAsync(image_box_count_.cpu(),
                                 image_box_count_.gpu(),
                                 image_box_count_.gpu_bytes(),
                                 cudaMemcpyDeviceToHost,
                                 stream_));
    checkRuntime(cudaStreamSynchronize(stream_));
    
    std::vector<object::DetectionResultArray> arrout(1);
    for (int ib = 0; ib < 1; ++ib)
    {
        float *parray                        = result_.cpu();
        int count                            = min(max_image_boxes_, *(image_box_count_.cpu()));
        object::DetectionResultArray &output = arrout[ib];
        for (int i = 0; i < count; ++i)
        {
            float *pbox  = parray + i * num_box_element_;
            int label    = pbox[5];
            int keepflag = pbox[6];
            if (keepflag == 1)
            {
                std::string name = class_names_[label];
                object::Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label, name);
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
    DFineSahiModelImpl *impl = new DFineSahiModelImpl();
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

std::shared_ptr<InferBase> load_dfine_sahi(const std::string &engine_file,
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
        return std::shared_ptr<DFineSahiModelImpl>((
            DFineSahiModelImpl *)loadraw(engine_file,
                names,
                confidence_threshold,
                nms_threshold,
                gpu_id,
                max_batch_size,
                auto_slice,
                slice_width,
                slice_height,
                slice_horizontal_ratio,
                slice_vertical_ratio));
    }
    catch (const std::exception &ex)
    {
        return nullptr;
    }
}

};