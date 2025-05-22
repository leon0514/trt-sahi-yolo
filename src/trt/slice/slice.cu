#include "common/check.hpp"
#include "kernels/kernel_warp.hpp"
#include "trt/slice/slice.hpp"
#include <cmath>

namespace slice
{

int calculateNumCuts(int dimension, int subDimension, double overlapRatio) // Use double for ratio
{
    if (subDimension <= 0)
        return (dimension > 0) ? 1 : 0;
    if (dimension <= subDimension)
        return 1;
    if (overlapRatio >= 1.0)
        return 1;
    double subDim_d = static_cast<double>(subDimension);
    double step     = subDim_d * (1.0 - overlapRatio);

    double remaining_dim = static_cast<double>(dimension) - subDim_d;

    double cuts = remaining_dim / step;

    const double epsilon    = 1e-6;
    int num_additional_cuts = static_cast<int>(cuts);
    if (std::abs(cuts - std::round(cuts)) > epsilon)
    {
        num_additional_cuts = static_cast<int>(std::ceil(cuts));
    }
    return 1 + num_additional_cuts;
}

static int calc_resolution_factor(int resolution)
{
    int expo = 0;
    while (pow(2, expo) < resolution)
        expo++;
    return expo - 1;
}

static std::string calc_aspect_ratio_orientation(int width, int height)
{
    if (width < height)
        return "vertical";
    else if (width > height)
        return "horizontal";
    else
        return "square";
}

static std::tuple<int, int, float, float>
calc_ratio_and_slice(const std::string &orientation, int slide = 1, float ratio = 0.1f)
{
    int slice_row, slice_col;
    float overlap_height_ratio, overlap_width_ratio;
    if (orientation == "vertical")
    {
        slice_row            = slide;
        slice_col            = slide * 2;
        overlap_height_ratio = ratio;
        overlap_width_ratio  = ratio;
    }
    else if (orientation == "horizontal")
    {
        slice_row            = slide * 2;
        slice_col            = slide;
        overlap_height_ratio = ratio;
        overlap_width_ratio  = ratio;
    }
    else if (orientation == "square")
    {
        slice_row            = slide;
        slice_col            = slide;
        overlap_height_ratio = ratio;
        overlap_width_ratio  = ratio;
    }
    return std::make_tuple(slice_row, slice_col, overlap_height_ratio, overlap_width_ratio);
}

static std::tuple<int, int, float, float>
calc_slice_and_overlap_params(const std::string &resolution, int width, int height, std::string orientation)
{
    int split_row, split_col;
    float overlap_height_ratio, overlap_width_ratio;
    if (resolution == "medium")
        std::tie(split_row, split_col, overlap_height_ratio, overlap_width_ratio) =
            calc_ratio_and_slice(orientation, 1, 0.8);

    else if (resolution == "high")
        std::tie(split_row, split_col, overlap_height_ratio, overlap_width_ratio) =
            calc_ratio_and_slice(orientation, 2, 0.4);

    else if (resolution == "ultra-high")
        std::tie(split_row, split_col, overlap_height_ratio, overlap_width_ratio) =
            calc_ratio_and_slice(orientation, 4, 0.4);
    else
    {
        split_col            = 1;
        split_row            = 1;
        overlap_width_ratio  = 1;
        overlap_height_ratio = 1;
    }
    int slice_height = height / split_col;
    int slice_width  = width / split_row;
    return std::make_tuple(slice_width, slice_height, overlap_height_ratio, overlap_width_ratio);
}

static std::tuple<int, int, float, float> get_resolution_selector(const std::string &resolution, int width, int height)
{
    std::string orientation = calc_aspect_ratio_orientation(width, height);
    return calc_slice_and_overlap_params(resolution, width, height, orientation);
}

static std::tuple<int, int, float, float> get_auto_slice_params(int width, int height)
{
    int resolution = height * width;
    int factor     = calc_resolution_factor(resolution);
    if (factor <= 18)
        return get_resolution_selector("low", width, height);
    else if (18 <= factor && factor < 21)
        return get_resolution_selector("medium", width, height);
    else if (21 <= factor && factor < 24)
        return get_resolution_selector("high", width, height);
    else
        return get_resolution_selector("ultra-high", width, height);
}

void SliceImage::autoSlice(const tensor::Image &image, void *stream)
{
    int slice_width;
    int slice_height;
    float overlap_width_ratio;
    float overlap_height_ratio;
    std::tie(slice_width, slice_height, overlap_width_ratio, overlap_height_ratio) =
        get_auto_slice_params(image.width, image.height);
    slice(image, slice_width, slice_height, overlap_width_ratio, overlap_height_ratio, stream);
}

void SliceImage::slice(const tensor::Image &image,
                       const int slice_width,
                       const int slice_height,
                       const double overlap_width_ratio,
                       const double overlap_height_ratio,
                       void *stream)
{
    slice_width_  = slice_width;
    slice_height_ = slice_height;

    int width  = image.width;
    int height = image.height;

    slice_num_h_ = calculateNumCuts(width, slice_width_, overlap_width_ratio);
    slice_num_v_ = calculateNumCuts(height, slice_height_, overlap_height_ratio);
    printf("------------------------------------------------------\n"
           "CUDA SAHI CROP IMAGE ✂️\n"
           "------------------------------------------------------\n"
           "%-30s: %-10d\n"
           "%-30s: %-10d\n"
           "%-30s: %-10d\n"
           "%-30s: %-10d\n"
           "%-30s: %-10f\n"
           "%-30s: %-10f\n"
           "%-30s: %-10d\n"
           "%-30s: %-10d\n"
           "------------------------------------------------------\n",
           "Image width",
           width,
           "Image height",
           height,
           "Slice width",
           slice_width_,
           "Slice height",
           slice_height_,
           "Overlap width ratio",
           overlap_width_ratio,
           "Overlap height ratio",
           overlap_height_ratio,
           "Number of horizontal cuts",
           slice_num_h_,
           "Number of vertical cuts",
           slice_num_v_);
    int slice_num            = slice_num_h_ * slice_num_v_;
    int overlap_width_pixel  = std::ceil(slice_width_ * overlap_width_ratio - 1);
    int overlap_height_pixel = std::ceil(slice_height_ * overlap_height_ratio - 1);

    size_t size_image      = 3 * width * height;
    size_t output_img_size = 3 * slice_width_ * slice_height_;

    input_image_.gpu(size_image);
    output_images_.gpu(slice_num * output_img_size);

    cudaStream_t stream_ = (cudaStream_t)stream;
    checkRuntime(cudaMemsetAsync(output_images_.gpu(), 114, output_images_.gpu_bytes(), stream_));

    checkRuntime(cudaMemcpyAsync(input_image_.gpu(), image.bgrptr, size_image, cudaMemcpyHostToDevice, stream_));

    uint8_t *input_device  = input_image_.gpu();
    uint8_t *output_device = output_images_.gpu();

    slice_start_point_.cpu(slice_num * 2);
    slice_start_point_.gpu(slice_num * 2);

    int *slice_start_point_ptr = slice_start_point_.cpu();

    for (int i = 0; i < slice_num_h_; i++)
    {
        int x = std::min(width - slice_width, std::max(0, i * (slice_width - overlap_width_pixel)));
        for (int j = 0; j < slice_num_v_; j++)
        {
            int y     = std::min(height - slice_height, std::max(0, j * (slice_height - overlap_height_pixel)));
            int index = (i * slice_num_v_ + j) * 2;
            slice_start_point_ptr[index]     = x;
            slice_start_point_ptr[index + 1] = y;
        }
    }

    checkRuntime(cudaMemcpyAsync(slice_start_point_.gpu(),
                                 slice_start_point_.cpu(),
                                 slice_start_point_.gpu_bytes(),
                                 cudaMemcpyHostToDevice,
                                 stream_));
    // checkRuntime(cudaStreamSynchronize(stream_));
    slice_plane(input_device,
                output_device,
                slice_start_point_.gpu(),
                width,
                height,
                slice_width,
                slice_height,
                slice_num_h_,
                slice_num_v_,
                stream_);

    // checkRuntime(cudaStreamSynchronize(stream_));

    // for (int i = 0; i < slice_num_h_; i++)
    // {
    //     for (int j = 0; j < slice_num_v_; j++)
    //     {
    //         int index = i * slice_num_v_ + j;

    //         cv::Mat image = cv::Mat::zeros(slice_height, slice_width, CV_8UC3);
    //         uint8_t* output_img_data = image.ptr<uint8_t>();
    //         cudaMemcpyAsync(output_img_data, output_device+index*output_img_size, output_img_size*sizeof(uint8_t),
    //         cudaMemcpyDeviceToHost, stream_);
    //         checkRuntime(cudaStreamSynchronize(stream_));
    //         cv::imwrite(std::to_string(index) + ".png", image);
    //     }
    // }
}

} // namespace slice
