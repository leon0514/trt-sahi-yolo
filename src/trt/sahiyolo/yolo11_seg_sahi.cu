#include "kernels/kernel_warp.hpp"
#include "trt/sahiyolo/yolo11_seg_sahi.hpp"
namespace sahiyolo
{

static __host__ __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

bool Yolo11SegSahiModelImpl::load(const std::string &engine_file,
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

    auto input_dim        = trt_->static_dims(0);
    bbox_head_dims_       = trt_->static_dims(1);
    segment_head_dims_    = trt_->static_dims(2);
    network_input_width_  = input_dim[3];
    network_input_height_ = input_dim[2];
    isdynamic_model_      = trt_->has_dynamic_dim();
    normalize_            = norm_image::Norm::alpha_beta(1 / 255.0f, 0.0f, norm_image::ChannelType::SwapRB);
    num_classes_          = bbox_head_dims_[2] - 4 - segment_head_dims_[1];
    return true;
}

void Yolo11SegSahiModelImpl::adjust_memory(int batch_size)
{
    size_t input_numel = network_input_width_ * network_input_height_ * 3;
    input_buffer_.gpu(batch_size * input_numel);
    bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
    segment_predict_.gpu(batch_size * segment_head_dims_[1] * segment_head_dims_[2] * segment_head_dims_[3]);
    output_boxarray_.gpu(max_image_boxes_ * num_box_element_);
    output_boxarray_.cpu(max_image_boxes_ * num_box_element_);

    mask_affine_matrix_.gpu(6);
    mask_affine_matrix_.cpu(6);

    affine_matrix_.gpu(6);
    affine_matrix_.cpu(6);

    inverse_affine_matrix_.gpu(6);
    inverse_affine_matrix_.cpu(6);

    image_box_count_.gpu(1);
    image_box_count_.cpu(1);
}

void Yolo11SegSahiModelImpl::preprocess(int ibatch, void *stream)
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

void Yolo11SegSahiModelImpl::compute_affine_matrix(affine::LetterBoxMatrix &affine, void *stream)
{
    affine.compute(std::make_tuple(slice_->slice_width_, slice_->slice_height_),
                   std::make_tuple(network_input_width_, network_input_height_));

    float *affine_matrix_device = affine_matrix_.gpu();
    float *affine_matrix_host   = affine_matrix_.cpu();

    float *inverse_affine_matrix_device = inverse_affine_matrix_.gpu();
    float *inverse_affine_matrix_host   = inverse_affine_matrix_.cpu();

    cudaStream_t stream_ = (cudaStream_t)stream;
    memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
    checkRuntime(
        cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream_));

    memcpy(inverse_affine_matrix_host, affine.i2d, sizeof(affine.i2d));
    checkRuntime(cudaMemcpyAsync(inverse_affine_matrix_device,
                                 inverse_affine_matrix_host,
                                 sizeof(affine.i2d),
                                 cudaMemcpyHostToDevice,
                                 stream_));
}

std::shared_ptr<object::SegmentMap> Yolo11SegSahiModelImpl::decode_segment(int ib, float *pbox, void *stream)
{
    int row_index                           = pbox[7];
    int batch_index                         = pbox[8];
    std::shared_ptr<object::SegmentMap> seg = nullptr;
    float *bbox_output_device               = bbox_predict_.gpu();

    int start_x = slice_->slice_start_point_.cpu()[batch_index * 2];
    int start_y = slice_->slice_start_point_.cpu()[batch_index * 2 + 1];

    float *mask_weights =
        bbox_output_device + (batch_index * bbox_head_dims_[1] + row_index) * bbox_head_dims_[2] + num_classes_ + 4;

    float *mask_head_predict = segment_predict_.gpu();

    // 变回640 x 640下的坐标
    float left, top, right, bottom;
    float *i2d = inverse_affine_matrix_.cpu();
    affine_project(i2d, pbox[0] - start_x, pbox[1] - start_y, &left, &top);
    affine_project(i2d, pbox[2] - start_x, pbox[3] - start_y, &right, &bottom);

    // 原始框大小
    int oirginal_box_width  = pbox[2] - pbox[0];
    int oirginal_box_height = pbox[3] - pbox[1];

    float box_width  = right - left;
    float box_height = bottom - top;

    // 变成160 x 160下的坐标
    float scale_to_predict_x = segment_head_dims_[3] / (float)network_input_width_;
    float scale_to_predict_y = segment_head_dims_[2] / (float)network_input_height_;

    left                 = left * scale_to_predict_x + 0.5f;
    top                  = top * scale_to_predict_y + 0.5f;
    int mask_out_width   = box_width * scale_to_predict_x + 0.5f;
    int mask_out_height  = box_height * scale_to_predict_y + 0.5f;
    cudaStream_t stream_ = (cudaStream_t)stream;
    if (mask_out_width > 0 && mask_out_height > 0)
    {
        seg                    = std::make_shared<object::SegmentMap>(oirginal_box_width, oirginal_box_height);
        int bytes_of_mask_out  = mask_out_width * mask_out_height;
        float *mask_out_device = box_segment_cache_.gpu(bytes_of_mask_out);

        int mask_dim                          = segment_head_dims_[1];
        unsigned char *original_mask_out_host = seg->data;
        decode_single_mask_invoker(left,
                                   top,
                                   mask_weights,
                                   mask_head_predict + batch_index * segment_head_dims_[1] * segment_head_dims_[2] *
                                                           segment_head_dims_[3],
                                   segment_head_dims_[3],
                                   segment_head_dims_[2],
                                   mask_out_device,
                                   mask_dim,
                                   mask_out_width,
                                   mask_out_height,
                                   stream_);
        original_box_segment_cache_.gpu(oirginal_box_width * oirginal_box_height);
        unsigned char *original_mask_out_device = original_box_segment_cache_.gpu();
        // 将160 x 160下的mask变换回原图下的mask 的变换矩阵
        affine::LetterBoxMatrix mask_affine_matrix;
        mask_affine_matrix.compute(std::make_tuple(mask_out_width, mask_out_height),
                                   std::make_tuple(oirginal_box_width, oirginal_box_height));
        float *mask_affine_matrix_device = mask_affine_matrix_.gpu();
        float *mask_affine_matrix_host   = mask_affine_matrix_.cpu();

        memcpy(mask_affine_matrix_host, mask_affine_matrix.d2i, sizeof(mask_affine_matrix.d2i));
        checkRuntime(cudaMemcpyAsync(mask_affine_matrix_device,
                                     mask_affine_matrix_host,
                                     sizeof(mask_affine_matrix.d2i),
                                     cudaMemcpyHostToDevice,
                                     stream_));

        // 单通道的变换矩阵
        // 在这里做过插值后将mask的值由0-1 变为 0-255，并且将 < 0.5的丢弃，不然范围会很大。
        // 先变为0-255再做插值会有锯齿
        warp_affine_bilinear_single_channel_mask_plane(mask_out_device,
                                                       mask_out_width,
                                                       mask_out_width,
                                                       mask_out_height,
                                                       original_mask_out_device,
                                                       oirginal_box_width,
                                                       oirginal_box_height,
                                                       mask_affine_matrix_device,
                                                       0,
                                                       stream_);

        checkRuntime(cudaMemcpyAsync(original_mask_out_host,
                                     original_mask_out_device,
                                     original_box_segment_cache_.gpu_bytes(),
                                     cudaMemcpyDeviceToHost,
                                     stream_));
    }
    checkRuntime(cudaStreamSynchronize(stream_));
    return seg;
}

InferResult Yolo11SegSahiModelImpl::forwards(const std::vector<cv::Mat> &inputs, void *stream)
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
            printf("num_image : %d\n", num_image);
            printf("max_batch_size : %d\n", max_batch_size_);
            // 这里的num_image是slice的数量，可能会大于max_batch_size
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

    float *bbox_output_device    = bbox_predict_.gpu();
    float *segment_output_device = segment_predict_.gpu();
#if NV_TENSORRT_MAJOR >= 10
    // yolov5 模型推理
    // TensorRT10需要指定输入输出名字，这里的输入输出分别是images, output0
    std::unordered_map<std::string, const void *> bindings = {{"images", input_buffer_.gpu()},
                                                              {"output0", bbox_output_device},
                                                              {"output1", segment_output_device}};
    if (!trt_->forward(bindings, stream_))
    {
        printf("Failed to tensorRT forward.\n");
        return {};
    }
#else
    std::vector<void *> bindings{input_buffer_.gpu(), bbox_output_device, segment_output_device};
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
        int start_x = slice_->slice_start_point_.cpu()[ib * 2];
        int start_y = slice_->slice_start_point_.cpu()[ib * 2 + 1];

        float *boxarray_device         = output_boxarray_.gpu();
        float *affine_matrix_device    = affine_matrix_.gpu();
        float *image_based_bbox_output = bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
        decode_kernel_invoker_v11(image_based_bbox_output,
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
    checkRuntime(cudaStreamSynchronize(stream_));
    float *boxarray_device = output_boxarray_.gpu();
    fast_nms_kernel_invoker(boxarray_device, box_count, max_image_boxes_, nms_threshold_, num_box_element_, stream_);

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

    std::vector<object::SegmentationResultArray> arrout(1);
    for (int ib = 0; ib < 1; ++ib)
    {
        float *parray                           = output_boxarray_.cpu();
        int count                               = std::min(max_image_boxes_, *(image_box_count_.cpu()));
        object::SegmentationResultArray &output = arrout[ib];
        for (int i = 0; i < count; ++i)
        {
            float *pbox  = parray + i * num_box_element_;
            int label    = pbox[5];
            int keepflag = pbox[6];
            // printf("keepflag : %d\n", keepflag);
            if (keepflag == 1)
            {
                std::string name = class_names_[label];
                auto seg         = decode_segment(ib, pbox, stream);
                object::Box seg_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label, name);
                object::SegmentationInstance result_object_box(seg_box, seg);
                output.emplace_back(std::move(result_object_box));
            }
        }
    }
    checkRuntime(cudaStreamSynchronize(stream_));
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
    Yolo11SegSahiModelImpl *impl = new Yolo11SegSahiModelImpl();
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

std::shared_ptr<InferBase> load_yolo_11_seg_sahi(const std::string &engine_file,
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
        return std::shared_ptr<Yolo11SegSahiModelImpl>((Yolo11SegSahiModelImpl *)loadraw(engine_file,
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