#include "trt/dfine/dfine.hpp"

namespace  dfine
{

bool DFineModelImpl::load(const std::string &engine_file,
    const std::vector<std::string> &names,
    float confidence_threshold,
    float nms_threshold,
    int gpu_id,
    int max_batch_size)
{
    trt_       = TensorRT::load(engine_file);
    device_id_ = gpu_id;
    if (trt_ == nullptr)
        return false;
    
    trt_->print();

    this->confidence_threshold_ = confidence_threshold;
    this->nms_threshold_        = nms_threshold;
    this->class_names_          = names;
    this->max_batch_size_       = max_batch_size;

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


InferResult DFineModelImpl::forwards(const std::vector<cv::Mat> &inputs, void *stream)
{
    int num_image = inputs.size();
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
    std::vector<affine::ResizeMatrix> affine_matrixs(infer_batch_size);

    cudaStream_t stream_ = (cudaStream_t)stream;
    for (int i = 0; i < num_image; ++i)
    {
        preprocess(i,
                   tensor::Image(inputs[i].data, inputs[i].cols, inputs[i].rows),
                   preprocess_images_buffers_[i],
                   affine_matrixs[i],
                   stream);
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
            output_labels_.gpu(),
            output_boxes_.gpu(),
            output_scores_.gpu()};
        if (!trt_->forward(bindings, stream_))
        {
            printf("Failed to tensorRT forward.\n");
            return {};
        }
    #endif

    checkRuntime(cudaMemcpyAsync(output_labels_.cpu(), output_labels_.gpu(), output_labels_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
    checkRuntime(cudaMemcpyAsync(output_boxes_.cpu(), output_boxes_.gpu(), output_boxes_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
    checkRuntime(cudaMemcpyAsync(output_scores_.cpu(), output_scores_.gpu(), output_scores_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
    
    checkRuntime(cudaStreamSynchronize(stream_));
    
    std::vector<object::DetectionResultArray> arrout(num_image);
    for (int ib = 0; ib < infer_batch_size; ++ib)
    {
        int64_t* labels_host   = output_labels_.cpu() + ib * label_head_dims_[1];
        float* boxes_host  = output_boxes_.cpu() + ib * box_head_dims_[1] * box_head_dims_[2];
        float* scores_host = output_scores_.cpu() + ib * score_head_dims_[1];

        object::DetectionResultArray &output = arrout[ib];
        for (int i=0; i < label_head_dims_[1]; i++)
        {
            int label = labels_host[i];
            float score = scores_host[i];

            float * pbox = boxes_host + i * box_head_dims_[2];
            if (score > confidence_threshold_)
            {
                std::string name = class_names_[label];
                object::Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], score, label, name);
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
    int max_batch_size)
{
    DFineModelImpl *impl = new DFineModelImpl();
    if (!impl->load(engine_file, names, confidence_threshold, nms_threshold, gpu_id, max_batch_size))
    {
        delete impl;
        impl = nullptr;
    }
    return impl;
}

std::shared_ptr<InferBase> load_dfine(const std::string &engine_file,
    const std::vector<std::string> &names,
    int gpu_id,
    float confidence_threshold,
    float nms_threshold,
    int max_batch_size)
{
    try
    {
        checkRuntime(cudaSetDevice(gpu_id));
        return std::shared_ptr<DFineModelImpl>((
            DFineModelImpl *)loadraw(engine_file, names, confidence_threshold, nms_threshold, gpu_id, max_batch_size));
    }
    catch (const std::exception &ex)
    {
        return nullptr;
    }
}

};