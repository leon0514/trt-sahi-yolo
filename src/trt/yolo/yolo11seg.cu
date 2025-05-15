#include "common/affine.hpp"
#include "common/check.hpp"
#include "common/image.hpp"
#include "kernels/kernel_warp.hpp"
#include "trt/yolo/yolo11seg.hpp"

static __host__ __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

namespace yolo
{

bool Yolo11SegModelImpl::load(const std::string &engine_file,
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

void Yolo11SegModelImpl::adjust_memory(int batch_size)
{
    size_t input_numel = network_input_width_ * network_input_height_ * 3;
    input_buffer_.gpu(batch_size * input_numel);
    bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
    segment_predict_.gpu(batch_size * segment_head_dims_[1] * segment_head_dims_[2] * segment_head_dims_[3]);
    output_boxarray_.gpu(batch_size * (max_image_boxes_ * num_box_element_));
    output_boxarray_.cpu(batch_size * (max_image_boxes_ * num_box_element_));

    mask_affine_matrix_.gpu(6);
    mask_affine_matrix_.cpu(6);

    if ((int)preprocess_buffers_.size() < batch_size)
    {
        for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
        {
            // 分配图片所需要的空间指针
            preprocess_buffers_.push_back(std::make_shared<tensor::Memory<unsigned char>>());
            image_box_counts_.push_back(std::make_shared<tensor::Memory<int>>());
            affine_matrixs_.push_back(std::make_shared<tensor::Memory<float>>());
            inverse_affine_matrixs_.push_back(std::make_shared<tensor::Memory<float>>());
            // 分配记录框所需要的空间
            image_box_counts_[i]->gpu(1);
            image_box_counts_[i]->cpu(1);
            // 分配仿射矩阵苏需要的空间
            affine_matrixs_[i]->gpu(6);
            affine_matrixs_[i]->cpu(6);
            // 分配逆仿射矩阵苏需要的空间
            inverse_affine_matrixs_[i]->gpu(6);
            inverse_affine_matrixs_[i]->cpu(6);
        }
    }
}

void Yolo11SegModelImpl::preprocess(int ibatch,
                                    const tensor::Image &image,
                                    std::shared_ptr<tensor::Memory<unsigned char>> preprocess_buffer,
                                    affine::LetterBoxMatrix &affine,
                                    void *stream)
{
    affine.compute(std::make_tuple(image.width, image.height),
                   std::make_tuple(network_input_width_, network_input_height_));
    size_t input_numel  = network_input_width_ * network_input_height_ * 3;
    float *input_device = input_buffer_.gpu() + ibatch * input_numel;
    size_t size_image   = image.width * image.height * 3;

    uint8_t *image_device = preprocess_buffer->gpu(size_image);
    uint8_t *image_host   = preprocess_buffer->cpu(size_image);

    float *affine_matrix_device = affine_matrixs_[ibatch]->gpu();
    float *affine_matrix_host   = affine_matrixs_[ibatch]->cpu();

    float *inverse_affine_matrix_device = inverse_affine_matrixs_[ibatch]->gpu();
    float *inverse_affine_matrix_host   = inverse_affine_matrixs_[ibatch]->cpu();

    cudaStream_t stream_ = (cudaStream_t)stream;
    memcpy(image_host, image.bgrptr, size_image);
    memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
    memcpy(inverse_affine_matrix_host, affine.i2d, sizeof(affine.i2d));
    checkRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
    checkRuntime(
        cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream_));

    checkRuntime(cudaMemcpyAsync(inverse_affine_matrix_device,
                                 inverse_affine_matrix_host,
                                 sizeof(affine.i2d),
                                 cudaMemcpyHostToDevice,
                                 stream_));

    warp_affine_bilinear_and_normalize_plane(image_device,
                                             image.width * 3,
                                             image.width,
                                             image.height,
                                             input_device,
                                             network_input_width_,
                                             network_input_height_,
                                             affine_matrix_device,
                                             114,
                                             normalize_,
                                             stream_);
}

std::shared_ptr<object::SegmentMap> Yolo11SegModelImpl::decode_segment(int ib, float *pbox, void *stream)
{
    int row_index                           = pbox[7];
    int batch_index                         = ib;
    std::shared_ptr<object::SegmentMap> seg = nullptr;
    float *bbox_output_device               = bbox_predict_.gpu();

    float *mask_weights =
        bbox_output_device + (batch_index * bbox_head_dims_[1] + row_index) * bbox_head_dims_[2] + num_classes_ + 4;

    float *mask_head_predict = segment_predict_.gpu();

    // 变回640 x 640下的坐标
    float left, top, right, bottom;
    float *i2d = inverse_affine_matrixs_[ib]->cpu();
    affine_project(i2d, pbox[0], pbox[1], &left, &top);
    affine_project(i2d, pbox[2], pbox[3], &right, &bottom);

    // 原始框大小
    int oirginal_box_width  = pbox[2] - pbox[0];
    int oirginal_box_height = pbox[3] - pbox[1];

    float box_width  = right - left;
    float box_height = bottom - top;

    // 变成160 x 160下的坐标
    float scale_to_predict_x = segment_head_dims_[3] / (float)network_input_width_;
    float scale_to_predict_y = segment_head_dims_[2] / (float)network_input_height_;

    left                = left * scale_to_predict_x + 0.5f;
    top                 = top * scale_to_predict_y + 0.5f;
    int mask_out_width  = box_width * scale_to_predict_x + 0.5f;
    int mask_out_height = box_height * scale_to_predict_y + 0.5f;

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
    return seg;
}

InferResult Yolo11SegModelImpl::forwards(const std::vector<cv::Mat> &inputs, void *stream)
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

    for (int ib = 0; ib < infer_batch_size; ++ib)
    {
        int *box_count = image_box_counts_[ib]->gpu();
        checkRuntime(cudaMemsetAsync(box_count, 0, sizeof(int), stream_));
        float *boxarray_device         = output_boxarray_.gpu() + ib * (max_image_boxes_ * num_box_element_);
        float *affine_matrix_device    = affine_matrixs_[ib]->gpu();
        float *image_based_bbox_output = bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
        checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
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
                                  0,
                                  0,
                                  ib,
                                  stream_);
        fast_nms_kernel_invoker(boxarray_device,
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
    std::vector<object::SegmentationResultArray> arrout(num_image);

    for (int ib = 0; ib < num_image; ++ib)
    {
        float *parray = output_boxarray_.cpu() + ib * (max_image_boxes_ * num_box_element_);
        // printf("image %d, count %d\n", ib, *(image_box_counts_[ib]->cpu()));
        int count                               = min(max_image_boxes_, *(image_box_counts_[ib]->cpu()));
        object::SegmentationResultArray &output = arrout[ib];
        for (int i = 0; i < count; ++i)
        {
            float *pbox  = parray + i * num_box_element_;
            int label    = pbox[5];
            int keepflag = pbox[6];
            if (keepflag == 1)
            {
                std::string name = class_names_[label];

                auto seg = decode_segment(ib, pbox, stream);
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
                          int max_batch_size)
{
    Yolo11SegModelImpl *impl = new Yolo11SegModelImpl();
    if (!impl->load(engine_file, names, confidence_threshold, nms_threshold, gpu_id, max_batch_size))
    {
        delete impl;
        impl = nullptr;
    }
    return impl;
}

std::shared_ptr<InferBase> load_yolo_11_seg(const std::string &engine_file,
                                            const std::vector<std::string> &names,
                                            int gpu_id,
                                            float confidence_threshold,
                                            float nms_threshold,
                                            int max_batch_size)
{
    try
    {
        checkRuntime(cudaSetDevice(gpu_id));
        return std::shared_ptr<Yolo11SegModelImpl>(
            (Yolo11SegModelImpl *)
                loadraw(engine_file, names, confidence_threshold, nms_threshold, gpu_id, max_batch_size));
    }
    catch (const std::exception &ex)
    {
        return nullptr;
    }
}

} // end namespace yolo