#include "kernels/kernel_warp.hpp"
#include "trt/sahiyolo/yolo11_obb_sahi.hpp"
namespace sahiyolo
{

bool Yolo11ObbSahiModelImpl::load(const std::string &engine_file,
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
    bbox_head_dims_ = trt_->static_dims(1);

    network_input_width_  = input_dim[3];
    network_input_height_ = input_dim[2];
    isdynamic_model_      = trt_->has_dynamic_dim();

    normalize_   = norm_image::Norm::alpha_beta(1 / 255.0f, 0.0f, norm_image::ChannelType::SwapRB);
    num_classes_ = bbox_head_dims_[2] - 5;
    return true;
}

void Yolo11ObbSahiModelImpl::adjust_memory(int batch_size)
{
    size_t input_numel = network_input_width_ * network_input_height_ * 3;
    input_buffer_.gpu(batch_size * input_numel);
    bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
    output_boxarray_.gpu(max_image_boxes_ * num_box_element_);
    output_boxarray_.cpu(max_image_boxes_ * num_box_element_);

    affine_matrix_.gpu(6);
    affine_matrix_.cpu(6);

    image_box_count_.gpu(1);
    image_box_count_.cpu(1);
}

void Yolo11ObbSahiModelImpl::preprocess(int ibatch, void *stream)
{
    size_t input_numel  = network_input_width_ * network_input_height_ * 3;
    float *input_device = input_buffer_.gpu() + ibatch * input_numel;
    size_t size_image   = slice_->slice_width_ * slice_->slice_height_ * 3;

    float *affine_matrix_device = affine_matrix_.gpu();
    uint8_t *image_device       = slice_->output_images_.gpu() + ibatch * size_image;

    // speed up
    cudaStream_t stream_ = (cudaStream_t)stream;

    warp_affine_bilinear_and_normalize_plane(image_device,
                                             slice_->slice_width_ * 3,
                                             slice_->slice_width_,
                                             slice_->slice_height_,
                                             input_device,
                                             network_input_width_,
                                             network_input_height_,
                                             affine_matrix_device,
                                             114,
                                             normalize_,
                                             stream_);
} // namespace sahiyolo

void Yolo11ObbSahiModelImpl::compute_affine_matrix(affine::LetterBoxMatrix &affine, void *stream)
{
    affine.compute(std::make_tuple(slice_->slice_width_, slice_->slice_height_),
                   std::make_tuple(network_input_width_, network_input_height_));

    float *affine_matrix_device = affine_matrix_.gpu();
    float *affine_matrix_host   = affine_matrix_.cpu();

    cudaStream_t stream_ = (cudaStream_t)stream;
    memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
    checkRuntime(
        cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream_));
}

InferResult Yolo11ObbSahiModelImpl::forwards(const std::vector<cv::Mat> &inputs, void *stream)
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
    this->max_image_boxes_ = max_image_boxes_ * num_image;
    auto input_dims        = trt_->static_dims(0);
    int infer_batch_size   = input_dims[0];
    if (infer_batch_size != num_image)
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
                                  start_x,
                                  start_y,
                                  ib,
                                  stream_);
    }

    float *boxarray_device = output_boxarray_.gpu();
    fast_nms_kernel_invoker_v11_obb(boxarray_device, box_count, max_image_boxes_, nms_threshold_, num_box_element_, stream_);

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

    std::vector<object::DetectionObbResultArray> arrout(1);
    for (int ib = 0; ib < 1; ++ib)
    {
        float *parray                        = output_boxarray_.cpu();
        int count                            = min(max_image_boxes_, *(image_box_count_.cpu()));
        printf("image %d, count %d\n", ib, count);
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
                          int max_batch_size,
                          bool auto_slice,
                          int slice_width,
                          int slice_height,
                          double slice_horizontal_ratio,
                          double slice_vertical_ratio)
{
    Yolo11ObbSahiModelImpl *impl = new Yolo11ObbSahiModelImpl();
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

std::shared_ptr<InferBase> load_yolo_11_obb_sahi(const std::string &engine_file,
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
        return std::shared_ptr<Yolo11ObbSahiModelImpl>((Yolo11ObbSahiModelImpl *)loadraw(engine_file,
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