#include "common/affine.hpp"
#include "common/check.hpp"
#include "common/image.hpp"
#include "kernels/kernel_warp.hpp"
#include "trt/yolo/yolo11obb.hpp"

namespace yolo
{

bool Yolo11ObbModelImpl::load(const std::string &engine_file,
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

    this->num_box_element_      = 10;
    this->confidence_threshold_ = confidence_threshold;
    this->nms_threshold_        = nms_threshold;
    this->class_names_          = names;
    this->max_batch_size_       = max_batch_size;

    auto input_dim  = trt_->static_dims(0);
    bbox_head_dims_ = trt_->static_dims(1);

    network_input_width_  = input_dim[3];
    network_input_height_ = input_dim[2];
    isdynamic_model_      = trt_->has_dynamic_dim();

    normalize_   = norm_image::Norm::alpha_beta(1 / 255.0f, 0.0f, norm_image::ChannelType::SwapRB);
    num_classes_ = bbox_head_dims_[2] - 5;
    return true;
}


InferResult Yolo11ObbModelImpl::forwards(const std::vector<cv::Mat> &inputs, void *stream)
{
    // 推理图片数量
    int num_image = inputs.size();
    assert(num_image <= max_batch_size_);
    // 输入的维度 batch x 3 x width x height
    auto input_dims      = trt_->static_dims(0);
    int infer_batch_size = input_dims[0];

    if (infer_batch_size != num_image)
    {
        if (isdynamic_model_)
        {
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
                       "less than or equal to the maximum batch[%d].\n",
                       num_image,
                       infer_batch_size);
                return {};
            }
        }
    }
    // 分配存储空间
    adjust_memory(infer_batch_size);
    std::vector<affine::LetterBoxMatrix> affine_matrixs(infer_batch_size);

    cudaStream_t stream_ = (cudaStream_t)stream;
    for (int i = 0; i < num_image; ++i)
    {
        preprocess(i,
                   tensor::Image(inputs[i].data, inputs[i].cols, inputs[i].rows),
                   preprocess_buffers_[i],
                   affine_matrixs[i],
                   stream);
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

    for (int ib = 0; ib < infer_batch_size; ++ib)
    {
        int *box_count = image_box_counts_[ib]->gpu();
        checkRuntime(cudaMemsetAsync(box_count, 0, sizeof(int), stream_));
        float *boxarray_device         = output_boxarray_.gpu() + ib * (max_image_boxes_ * num_box_element_);
        float *affine_matrix_device    = affine_matrixs_[ib]->gpu();
        float *image_based_bbox_output = bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
        checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
        decode_kernel_invoker_v11_obb(image_based_bbox_output,
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
                                      0,
                                      0,
                                      ib,
                                      stream_);
        fast_nms_kernel_invoker_v11_obb(boxarray_device,
                                        box_count,
                                        max_image_boxes_,
                                        nms_threshold_,
                                        num_box_element_,
                                        stream_);
        checkRuntime(cudaMemcpyAsync(image_box_counts_[ib]->cpu(),
                                     image_box_counts_[ib]->gpu(),
                                     image_box_counts_[ib]->gpu_bytes(),
                                     cudaMemcpyDeviceToHost,
                                     stream_));
    }
    checkRuntime(cudaMemcpyAsync(output_boxarray_.cpu(),
                                 output_boxarray_.gpu(),
                                 output_boxarray_.gpu_bytes(),
                                 cudaMemcpyDeviceToHost,
                                 stream_));
    checkRuntime(cudaStreamSynchronize(stream_));
    std::vector<object::DetectionObbResultArray> arrout(num_image);

    for (int ib = 0; ib < num_image; ++ib)
    {
        float *parray                           = output_boxarray_.cpu() + ib * (max_image_boxes_ * num_box_element_);
        int count                               = min(max_image_boxes_, *(image_box_counts_[ib]->cpu()));
        object::DetectionObbResultArray &output = arrout[ib];
        for (int i = 0; i < count; ++i)
        {
            float *pbox      = parray + i * num_box_element_;
            int label        = pbox[6];
            int keepflag     = pbox[7];
            std::string name = class_names_[label];
            if (keepflag == 1)
            {
                object::OBBox result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], pbox[5], label, name);
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
    Yolo11ObbModelImpl *impl = new Yolo11ObbModelImpl();
    if (!impl->load(engine_file, names, confidence_threshold, nms_threshold, gpu_id, max_batch_size))
    {
        delete impl;
        impl = nullptr;
    }
    return impl;
}

std::shared_ptr<InferBase> load_yolo_11_obb(const std::string &engine_file,
                                            const std::vector<std::string> &names,
                                            int gpu_id,
                                            float confidence_threshold,
                                            float nms_threshold,
                                            int max_batch_size)
{
    try
    {
        checkRuntime(cudaSetDevice(gpu_id));
        return std::shared_ptr<Yolo11ObbModelImpl>(
            (Yolo11ObbModelImpl *)
                loadraw(engine_file, names, confidence_threshold, nms_threshold, gpu_id, max_batch_size),
            [](InferBase *impl)
            {
                if (impl != nullptr)
                {
                    delete impl;
                }
            });
    }
    catch (const std::exception &ex)
    {
        return nullptr;
    }
}

} // end namespace yolo