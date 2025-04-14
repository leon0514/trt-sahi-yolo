#include "model/yolo.hpp"
#include <vector>
#include <memory>
#include "slice/slice.hpp"
#include "model/affine.hpp"
#include "common/check.hpp"

#ifdef TRT10
#include "common/tensorrt.hpp"
namespace TensorRT = TensorRT10;
#else
#include "common/tensorrt8.hpp"
namespace TensorRT = TensorRT8;
#endif

#define GPU_BLOCK_THREADS 512

namespace yolo
{


static const int NUM_BOX_ELEMENT = 9;  // left, top, right, bottom, confidence, class, keepflag, row_index(output), batch_index
// 9个元素，分别是：左上角坐标，右下角坐标，置信度，类别，是否保留，行索引（mask weights），batch_index
// 其中行索引用于找到mask weights，batch_index用于找到当前batch的图片位置
// row_index 用于找到mask weights
static const int MAX_IMAGE_BOXES = 1024 * 4;

static const int KEY_POINT_NUM   = 17; // 关键点数量

static dim3 grid_dims(int numJobs){
  int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
  return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

static dim3 block_dims(int numJobs){
  return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

static __host__ __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy) 
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel_v5(float *predict, int num_bboxes, int num_classes,
                                              int output_cdim, float confidence_threshold,
                                              float *invert_affine_matrix, float *parray, int *box_count,
                                              int max_image_boxes, int start_x, int start_y, int batch_index) 
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float *pitem = predict + output_cdim * position;
    float objectness = pitem[4];
    if (objectness < confidence_threshold) return;

    float *class_confidence = pitem + 5;
    
    float confidence = *class_confidence++;
    int label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) 
    {
        if (*class_confidence > confidence) 
        {
            confidence = *class_confidence;
            label = i;
        }
    }
    confidence *= objectness;
    if (confidence < confidence_threshold) return;
    
    int index = atomicAdd(box_count, 1);
    if (index >= max_image_boxes) return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float *pout_item = parray + index * NUM_BOX_ELEMENT;
    *pout_item++ = left + start_x;
    *pout_item++ = top + start_y;
    *pout_item++ = right + start_x;
    *pout_item++ = bottom + start_y;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
    *pout_item++ = position;
    *pout_item++ = batch_index; // batch_index
    // 这里的batch_index是为了在后续的mask weights中使用，方便找到当前batch的图片位置
}

static __global__ void decode_kernel_v8(float *predict, int num_bboxes, int num_classes,
                                              int output_cdim, float confidence_threshold,
                                              float *invert_affine_matrix, float *parray, int *box_count,
                                              int max_image_boxes, int start_x, int start_y, int batch_index) 
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float *pitem = predict + output_cdim * position;
    float *class_confidence = pitem + 4;
    float confidence = *class_confidence++;
    int label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) 
    {
        if (*class_confidence > confidence) 
        {
            confidence = *class_confidence;
            label = i;
        }
    }
    if (confidence < confidence_threshold) return;

    int index = atomicAdd(box_count, 1);
    if (index >= max_image_boxes) return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float *pout_item = parray + index * NUM_BOX_ELEMENT;
    *pout_item++ = left + start_x;
    *pout_item++ = top + start_y;
    *pout_item++ = right + start_x;
    *pout_item++ = bottom + start_y;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
    *pout_item++ = position;
    *pout_item++ = batch_index; // batch_index
    // 这里的batch_index是为了在后续的mask weights中使用，方便找到当前batch的图片位置
}

static __global__ void decode_kernel_11pose(float *predict, int num_bboxes, int num_classes,
    int output_cdim, float confidence_threshold,
    float *invert_affine_matrix, float *parray,
    int *box_count, int max_image_boxes, int start_x, int start_y, int batch_index) 
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float *pitem            = predict + output_cdim * position;
    float *class_confidence = pitem + 4;
    float *key_points       = pitem + 4 + num_classes;
    float confidence        = *class_confidence++;

    int label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) 
    {
        if (*class_confidence > confidence) 
        {
            confidence = *class_confidence;
            label = i;
        }
    }
    if (confidence < confidence_threshold) return;

    int index = atomicAdd(box_count, 1);
    if (index >= max_image_boxes) return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float *pout_item = parray + index * (NUM_BOX_ELEMENT + KEY_POINT_NUM * 3);
    *pout_item++ = left + start_x;
    *pout_item++ = top + start_y;
    *pout_item++ = right + start_x;
    *pout_item++ = bottom + start_y;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
    *pout_item++ = position;
    *pout_item++ = batch_index; // batch_index
    for (int i = 0; i < KEY_POINT_NUM; i++)
    {
        float x = *key_points++;
        float y = *key_points++;
        affine_project(invert_affine_matrix, x, y, &x, &y);
        float score  = *key_points++;
        *pout_item++ = x + start_x;
        *pout_item++ = y + start_y;
        *pout_item++ = score;
    }
}


static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft,
                                float btop, float bright, float bbottom)
{
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f) return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}


static __global__ void fast_nms_kernel(float *bboxes, int* box_count, int max_image_boxes, float threshold) 
{
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*box_count, MAX_IMAGE_BOXES);
    if (position >= count) return;

    // left, top, right, bottom, confidence, class, keepflag
    float *pcurrent = bboxes + position * NUM_BOX_ELEMENT;
    for (int i = 0; i < count; ++i) 
    {
        float *pitem = bboxes + i * NUM_BOX_ELEMENT;
        if (i == position || pcurrent[5] != pitem[5]) continue;

        if (pitem[4] >= pcurrent[4]) 
        {
            if (pitem[4] == pcurrent[4] && i < position) continue;

            float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1],
                                pitem[2], pitem[3]);

            if (iou > threshold) 
            {
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
}


static __global__ void fast_nms_pose_kernel(float *bboxes, int* box_count, int max_image_boxes, float threshold) 
{
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*box_count, MAX_IMAGE_BOXES);
    if (position >= count) return;

    // left, top, right, bottom, confidence, class, keepflag
    float *pcurrent = bboxes + position * (NUM_BOX_ELEMENT + KEY_POINT_NUM * 3);
    for (int i = 0; i < count; ++i) 
    {
        float *pitem = bboxes + i * (NUM_BOX_ELEMENT + KEY_POINT_NUM * 3);
        if (i == position || pcurrent[5] != pitem[5]) continue;

        if (pitem[4] >= pcurrent[4]) 
        {
            if (pitem[4] == pcurrent[4] && i < position) continue;

            float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1],
                                pitem[2], pitem[3]);

            if (iou > threshold) 
            {
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
}


static __global__ void decode_single_mask_kernel(int left, int top, float *mask_weights,
    float *mask_predict, int mask_width,
    int mask_height, float *mask_out,
    int mask_dim, int out_width, int out_height) 
{
    // mask_predict to mask_out
    // mask_weights @ mask_predict
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= out_width || dy >= out_height) return;

    int sx = left + dx;
    int sy = top + dy;
    if (sx < 0 || sx >= mask_width || sy < 0 || sy >= mask_height) 
    {
        mask_out[dy * out_width + dx] = 0;
        return;
    }

    float cumprod = 0;
    for (int ic = 0; ic < mask_dim; ++ic) 
    {
        float cval = mask_predict[(ic * mask_height + sy) * mask_width + sx];
        float wval = mask_weights[ic];
        cumprod += cval * wval;
    }

    float alpha = 1.0f / (1.0f + exp(-cumprod));
    // 在这里先返回float值，再将mask采样回原图后才x255
    mask_out[dy * out_width + dx] = alpha;
}

static void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
                            int mask_width, int mask_height, float *mask_out,
                            int mask_dim, int out_width, int out_height, cudaStream_t stream) 
{
    // mask_weights is mask_dim(32 element) gpu pointer
    dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
    dim3 block(32, 32);

    checkKernel(decode_single_mask_kernel<<<grid, block, 0, stream>>>(
    left, top, mask_weights, mask_predict, mask_width, mask_height, mask_out, mask_dim, out_width,
    out_height));
}

static void decode_kernel_invoker_v8(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                  float confidence_threshold, float nms_threshold,
                                  float *invert_affine_matrix, float *parray, int* box_count, int max_image_boxes,
                                  int start_x, int start_y, int batch_index, cudaStream_t stream) 
{
    auto grid = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);

    checkKernel(decode_kernel_v8<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
            parray, box_count, max_image_boxes, start_x, start_y, batch_index));
}


static void decode_kernel_invoker_v5(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                  float confidence_threshold, float nms_threshold,
                                  float *invert_affine_matrix, float *parray, int* box_count, int max_image_boxes,
                                  int start_x, int start_y, int batch_index, cudaStream_t stream) 
{
    auto grid = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);

    checkKernel(decode_kernel_v5<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
            parray, box_count, max_image_boxes, start_x, start_y, batch_index));
}

static void decode_kernel_invoker_v11pose(float *predict, int num_bboxes, int num_classes, int output_cdim,
    float confidence_threshold, float nms_threshold,
    float *invert_affine_matrix, float *parray, int* box_count, int max_image_boxes,
    int start_x, int start_y, int batch_index, cudaStream_t stream) 
{
    auto grid = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);

    checkKernel(decode_kernel_11pose<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
            parray, box_count, max_image_boxes, start_x, start_y, batch_index));
}

static void fast_nms_kernel_invoker(float *parray, int* box_count, int max_image_boxes, float nms_threshold, cudaStream_t stream)
{
    auto grid = grid_dims(max_image_boxes);
    auto block = block_dims(max_image_boxes);
    checkKernel(fast_nms_kernel<<<grid, block, 0, stream>>>(parray, box_count, max_image_boxes, nms_threshold));
}

static void fast_nms_pose_kernel_invoker(float *parray, int* box_count, int max_image_boxes, float nms_threshold, cudaStream_t stream)
{
    auto grid = grid_dims(max_image_boxes);
    auto block = block_dims(max_image_boxes);
    checkKernel(fast_nms_pose_kernel<<<grid, block, 0, stream>>>(parray, box_count, max_image_boxes, nms_threshold));
}

class YoloModelImpl : public Infer 
{
public:
    YoloType yolo_type_;

    // for sahi crop image
    std::shared_ptr<slice::SliceImage> slice_;
    std::shared_ptr<TensorRT::Engine> trt_;
    std::string engine_file_;

    tensor::Memory<int> box_count_;

    tensor::Memory<float> affine_matrix_;
    tensor::Memory<float> invert_affine_matrix_;
    tensor::Memory<float> mask_affine_matrix_;
    tensor::Memory<float> input_buffer_, bbox_predict_, segment_predict_, output_boxarray_;

    int network_input_width_, network_input_height_;
    affine::Norm normalize_;
    std::vector<int> bbox_head_dims_;
    std::vector<int> segment_head_dims_;
    bool isdynamic_model_ = false;
    bool has_segment_ = false;

    std::vector<std::shared_ptr<tensor::Memory<float>>> box_segment_cache_;

    float confidence_threshold_;
    float nms_threshold_;

    int num_classes_ = 0;

    virtual ~YoloModelImpl() = default;

    void adjust_memory(int batch_size) 
    {
        // the inference batch_size
        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        input_buffer_.gpu(batch_size * input_numel);
        bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
        output_boxarray_.gpu(MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
        output_boxarray_.cpu(MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);

        if (has_segment_)
        {
            segment_predict_.gpu(batch_size * segment_head_dims_[1] * segment_head_dims_[2] *
                                 segment_head_dims_[3]);
        }
        

        affine_matrix_.gpu(6);
        affine_matrix_.cpu(6);
        invert_affine_matrix_.gpu(6);
        invert_affine_matrix_.cpu(6);

        mask_affine_matrix_.gpu(6);
        mask_affine_matrix_.cpu(6);

        box_count_.gpu(1);
        box_count_.cpu(1);
    }

    void preprocess(int ibatch, affine::LetterBoxMatrix &affine, void *stream = nullptr)
    {
        affine.compute(std::make_tuple(slice_->slice_width_, slice_->slice_height_),
                    std::make_tuple(network_input_width_, network_input_height_));

        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        float *input_device = input_buffer_.gpu() + ibatch * input_numel;
        size_t size_image = slice_->slice_width_ * slice_->slice_height_ * 3;

        float *affine_matrix_device = affine_matrix_.gpu();
        uint8_t *image_device = slice_->output_images_.gpu() + ibatch * size_image;

        float *affine_matrix_host = affine_matrix_.cpu();

	float *invert_affine_matrix_device = invert_affine_matrix_.gpu();
        float *invert_affine_matrix_host = invert_affine_matrix_.cpu();

        // speed up
        cudaStream_t stream_ = (cudaStream_t)stream;
        
	memcpy(invert_affine_matrix_host, affine.i2d, sizeof(affine.i2d));
        checkRuntime(cudaMemcpyAsync(invert_affine_matrix_device, invert_affine_matrix_host, sizeof(affine.i2d), cudaMemcpyHostToDevice, stream_));
	
	memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        checkRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                                    cudaMemcpyHostToDevice, stream_));

        affine::warp_affine_bilinear_and_normalize_plane(image_device, slice_->slice_width_ * 3, slice_->slice_width_,
                                                slice_->slice_height_, input_device, network_input_width_,
                                                network_input_height_, affine_matrix_device, 114,
                                                normalize_, stream_);
    }

    bool load(const std::string &engine_file, YoloType yolo_type, float confidence_threshold, float nms_threshold) 
    {
        trt_ = TensorRT::load(engine_file);
        if (trt_ == nullptr) return false;

        trt_->print();

        this->confidence_threshold_ = confidence_threshold;
        this->nms_threshold_ = nms_threshold;
        this->yolo_type_ = yolo_type;

        auto input_dim  = trt_->static_dims(0);
        bbox_head_dims_ = trt_->static_dims(1);

        has_segment_ = yolo_type == YoloType::YOLOV8SEG || yolo_type == YoloType::YOLOV5SEG || yolo_type == YoloType::YOLOV11SEG;
	printf("has_segment_ = %d\n", has_segment_);
        if (has_segment_) {
            bbox_head_dims_    = trt_->static_dims(1);
            segment_head_dims_ = trt_->static_dims(2);
        }
	for (const auto& dim : segment_head_dims_)
	{
	    printf("dim = %d\t", dim);
	}
	printf("\n");

        network_input_width_ = input_dim[3];
        network_input_height_ = input_dim[2];
        isdynamic_model_ = trt_->has_dynamic_dim();

        normalize_ = affine::Norm::alpha_beta(1 / 255.0f, 0.0f, affine::ChannelType::SwapRB);
        if (this->yolo_type_ == YoloType::YOLOV8 || this->yolo_type_ == YoloType::YOLOV11)
        {
            num_classes_ = bbox_head_dims_[2] - 4;
        }
        else if (this->yolo_type_ == YoloType::YOLOV5)
        {
            num_classes_ = bbox_head_dims_[2] - 5;
        }
        else if (this->yolo_type_ == YoloType::YOLOV11POSE)
        {
            num_classes_ = bbox_head_dims_[2] - 4 - KEY_POINT_NUM * 3;
            // NUM_BOX_ELEMENT = 8 + KEY_POINT_NUM * 3;
        }
	else if (this->yolo_type_ == YoloType::YOLOV11SEG)
	{
	    num_classes_ = bbox_head_dims_[2] - 4 - segment_head_dims_[1];
	}
        return true;
    }


    virtual BoxArray forward(const tensor::Image &image, int slice_width, int slice_height, float overlap_width_ratio, float overlap_height_ratio, void *stream = nullptr) override 
    {
        slice_->slice(image, slice_width, slice_height, overlap_width_ratio, overlap_height_ratio, stream);
        return forwards(stream);
    }

    virtual BoxArray forward(const tensor::Image &image, void *stream = nullptr) override 
    {
        slice_->autoSlice(image, stream);
        return forwards(stream);
    }

    virtual BoxArray forwards(void *stream = nullptr) override 
    {
        int num_image = slice_->slice_num_h_ * slice_->slice_num_v_;
        if (num_image == 0) return {};
        
        auto input_dims = trt_->static_dims(0);
        int infer_batch_size = input_dims[0];
        if (infer_batch_size != num_image) 
        {
            if (isdynamic_model_) 
            {
                infer_batch_size = num_image;
                input_dims[0] = num_image;
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
                    printf(
                        "When using static shape model, number of images[%d] must be "
                        "less than or equal to the maximum batch[%d].",
                        num_image, infer_batch_size);
                    return {};
                }
            }
        }
        adjust_memory(infer_batch_size);

        affine::LetterBoxMatrix affine_matrix;
        cudaStream_t stream_ = (cudaStream_t)stream;
        for (int i = 0; i < num_image; ++i)
            preprocess(i, affine_matrix, stream);

        float *bbox_output_device = bbox_predict_.gpu();
        #ifdef TRT10
        std::unordered_map<std::string, const void *> bindings;
        if (has_segment_)
        {
            float *segment_output_device = segment_predict_.gpu();
            bindings = {
                { "images", input_buffer_.gpu() }, 
                { "output0", bbox_output_device },
                { "output1", segment_output_device }
            };
        }
        else
        {
            bindings = {
                { "images", input_buffer_.gpu() }, 
                { "output0", bbox_output_device }
            };
           
        } 
        if (!trt_->forward(bindings, stream_))
        {
            printf("Failed to tensorRT forward.");
            return {};
        }
        #else
        std::vector<void *> bindings{input_buffer_.gpu(), bbox_output_device};
        if (has_segment_)
        {
            float *segment_output_device = segment_predict_.gpu();
            bindings = { input_buffer_.gpu(), bbox_output_device, segment_predict_.gpu()};
        }
        else
        {
            bindings = { input_buffer_.gpu(), bbox_output_device };
           
        } 
        if (!trt_->forward(bindings, stream_))
        {
            printf("Failed to tensorRT forward.");
            return {};
        }
        #endif

        int* box_count = box_count_.gpu();
        checkRuntime(cudaMemsetAsync(box_count, 0, sizeof(int), stream_));
        for (int ib = 0; ib < num_image; ++ib) 
        {
            int start_x = slice_->slice_start_point_.cpu()[ib*2];
            int start_y = slice_->slice_start_point_.cpu()[ib*2+1];
            // float *boxarray_device =
            //     output_boxarray_.gpu() + ib * (MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            float *boxarray_device = output_boxarray_.gpu();
            float *affine_matrix_device = affine_matrix_.gpu();
            float *image_based_bbox_output =
                bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
            if (yolo_type_ == YoloType::YOLOV5 || yolo_type_ == YoloType::YOLOV5SEG)
            {
                decode_kernel_invoker_v5(image_based_bbox_output, bbox_head_dims_[1], num_classes_,
                                    bbox_head_dims_[2], confidence_threshold_, nms_threshold_,
                                    affine_matrix_device, boxarray_device, box_count, MAX_IMAGE_BOXES, start_x, start_y, ib, stream_);
            }
            else if (yolo_type_ == YoloType::YOLOV8 || yolo_type_ == YoloType::YOLOV11 || yolo_type_ == YoloType::YOLOV8SEG || yolo_type_ == YoloType::YOLOV11SEG)
            {
                decode_kernel_invoker_v8(image_based_bbox_output, bbox_head_dims_[1], num_classes_,
                                    bbox_head_dims_[2], confidence_threshold_, nms_threshold_,
                                    affine_matrix_device, boxarray_device, box_count, MAX_IMAGE_BOXES, start_x, start_y, ib, stream_);
            }
            else if (yolo_type_ == YoloType::YOLOV11POSE)
            {
                decode_kernel_invoker_v11pose(image_based_bbox_output, bbox_head_dims_[1], num_classes_,
                    bbox_head_dims_[2], confidence_threshold_, nms_threshold_,
                    affine_matrix_device, boxarray_device, box_count, MAX_IMAGE_BOXES, start_x, start_y, ib, stream_);
            }
            
        }
        float *boxarray_device =  output_boxarray_.gpu();
        if (yolo_type_ == YoloType::YOLOV11POSE)
        {
            fast_nms_pose_kernel_invoker(boxarray_device, box_count, MAX_IMAGE_BOXES, nms_threshold_, stream_);
        }
        else
        {
            fast_nms_kernel_invoker(boxarray_device, box_count, MAX_IMAGE_BOXES, nms_threshold_, stream_);
        }
        
        checkRuntime(cudaMemcpyAsync(output_boxarray_.cpu(), output_boxarray_.gpu(),
                                    output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaMemcpyAsync(box_count_.cpu(), box_count_.gpu(),
                                    box_count_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaStreamSynchronize(stream_));

        BoxArray result;
        float *parray = output_boxarray_.cpu();
        int count = min(MAX_IMAGE_BOXES, *(box_count_.cpu()));

        int imemory = 0;
        for (int i = 0; i < count; ++i) 
        {
            int box_element = (yolo_type_ == YoloType::YOLOV11POSE) ? (NUM_BOX_ELEMENT + KEY_POINT_NUM * 3) : NUM_BOX_ELEMENT;
            float *pbox = parray + i * box_element;
            int label = pbox[5];
            int keepflag = pbox[6];
            if (keepflag == 1) 
            {
                Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                if (yolo_type_ == YoloType::YOLOV11POSE)
                {
                    result_object_box.pose.reserve(KEY_POINT_NUM);
                    for (int i = 0; i < KEY_POINT_NUM; i++)
                    {
                        result_object_box.pose.emplace_back(pbox[9+i*3], pbox[9+i*3+1], pbox[9+i*3+2]);
                    }
                }
                if (has_segment_)
                {
                    int row_index = pbox[7];
                    int batch_index = pbox[8];

                    int start_x = slice_->slice_start_point_.cpu()[batch_index*2];
                    int start_y = slice_->slice_start_point_.cpu()[batch_index*2+1];

                    int mask_dim = segment_head_dims_[1];
                    float *mask_weights = bbox_output_device +
                                        (batch_index * bbox_head_dims_[1] + row_index) * bbox_head_dims_[2] +
                                        num_classes_ + 4;

                    float *mask_head_predict = segment_predict_.gpu();
                    float left, top, right, bottom;
                    // 变回640 x 640下的坐标
                    float *i2d = invert_affine_matrix_.cpu();
                    affine_project(i2d, pbox[0] - start_x, pbox[1] - start_y, &left, &top);
                    affine_project(i2d, pbox[2] - start_x, pbox[3] - start_y, &right, &bottom);

                    int oirginal_box_width  = pbox[2] - pbox[0];
                    int oirginal_box_height = pbox[3] - pbox[1];

                    float box_width = right - left;
                    float box_height = bottom - top;
                    
                    // 变成160 x 160下的坐标
                    float scale_to_predict_x = segment_head_dims_[3] / (float)network_input_width_;
                    float scale_to_predict_y = segment_head_dims_[2] / (float)network_input_height_;
                    left = left * scale_to_predict_x + 0.5f;
                    top = top * scale_to_predict_y + 0.5f;
                    int mask_out_width = box_width * scale_to_predict_x + 0.5f;
                    int mask_out_height = box_height * scale_to_predict_y + 0.5f;

                    if (mask_out_width > 0 && mask_out_height > 0) 
                    {
                        if (imemory >= (int)box_segment_cache_.size()) 
                        {
                            box_segment_cache_.push_back(std::make_shared<tensor::Memory<float>>());
                        }
                        int bytes_of_mask_out = mask_out_width * mask_out_height;
                        auto box_segment_output_memory = box_segment_cache_[imemory];

                        result_object_box.seg =
                        std::make_shared<InstanceSegmentMap>(oirginal_box_width, oirginal_box_height);

                        float *mask_out_device = box_segment_output_memory->gpu(bytes_of_mask_out);
                        unsigned char *original_mask_out_host = result_object_box.seg->data;

                        decode_single_mask(left, top, mask_weights,
                                            mask_head_predict + batch_index * segment_head_dims_[1] *
                                                                    segment_head_dims_[2] *
                                                                    segment_head_dims_[3],
                                            segment_head_dims_[3], segment_head_dims_[2], mask_out_device,
                                            mask_dim, mask_out_width, mask_out_height, stream_);
                    
                        tensor::Memory<unsigned char> original_mask_out;
                        original_mask_out.gpu(oirginal_box_width * oirginal_box_height);
                        unsigned char *original_mask_out_device = original_mask_out.gpu();
                        
                        // 将160 x 160下的mask变换回原图下的mask 的变换矩阵
                        affine::LetterBoxMatrix mask_affine_matrix;
                        mask_affine_matrix.compute(std::make_tuple(mask_out_width, mask_out_height),
                                                std::make_tuple(oirginal_box_width, oirginal_box_height));

                        float *mask_affine_matrix_device = mask_affine_matrix_.gpu();
                        float *mask_affine_matrix_host = mask_affine_matrix_.cpu();

                        memcpy(mask_affine_matrix_host, mask_affine_matrix.d2i, sizeof(mask_affine_matrix.d2i));
                        checkRuntime(cudaMemcpyAsync(mask_affine_matrix_device, mask_affine_matrix_host,
                                                    sizeof(mask_affine_matrix.d2i), cudaMemcpyHostToDevice,
                                                    stream_));

                        // 单通道的变换矩阵
                        // 在这里做过插值后将mask的值由0-1 变为 0-255，并且将 < 0.5的丢弃，不然范围会很大。
                        // 先变为0-255再做插值会有锯齿
                        affine::warp_affine_bilinear_single_channel_plane(
                            mask_out_device, mask_out_width, mask_out_width, mask_out_height,
                            original_mask_out_device, oirginal_box_width, oirginal_box_height,
                            mask_affine_matrix_device, 0, stream_);
                        checkRuntime(cudaMemcpyAsync(original_mask_out_host, original_mask_out_device,
                                                    original_mask_out.gpu_bytes(),
                                                    cudaMemcpyDeviceToHost, stream_));
                    }
                }
                
                result.emplace_back(result_object_box);
            }
        }
        return result;
    }

};


Infer *loadraw(const std::string &engine_file, YoloType yolo_type, float confidence_threshold,
               float nms_threshold) 
{
    YoloModelImpl *impl = new YoloModelImpl();
    if (!impl->load(engine_file, yolo_type, confidence_threshold, nms_threshold)) 
    {
        delete impl;
        impl = nullptr;
    }
    impl->slice_ = std::make_shared<slice::SliceImage>();
    return impl;
}

std::shared_ptr<Infer> load(const std::string &engine_file, YoloType yolo_type, int gpu_id, float confidence_threshold, float nms_threshold) 
{
    checkRuntime(cudaSetDevice(gpu_id));
    return std::shared_ptr<YoloModelImpl>((YoloModelImpl *)loadraw(engine_file, yolo_type, confidence_threshold, nms_threshold));
}

}
