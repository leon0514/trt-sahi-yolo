#include "kernels/slice_kernel.cuh"

namespace cuda
{

__global__ void slice_kernel(const unsigned char *__restrict__ image,
                             unsigned char *__restrict__ outs,
                             const int width,
                             const int height,
                             const int slice_width,
                             const int slice_height,
                             const int slice_num_h,
                             const int slice_num_v,
                             const int *__restrict__ slice_start_point)
{
    if (width <= 0 || height <= 0 || slice_width <= 0 || slice_height <= 0 || slice_num_h <= 0 || slice_num_v <= 0)
    {
        return;
    }
    const int slice_idx = blockIdx.z;

    const int start_x = slice_start_point[slice_idx * 2];
    const int start_y = slice_start_point[slice_idx * 2 + 1];

    // 当前像素在切片内的相对位置
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= slice_width || y >= slice_height)
        return;

    const int dx = start_x + x;
    const int dy = start_y + y;

    if (dx >= width || dy >= height)
        return;

    // 读取像素
    const size_t src_base_index = (size_t)dy * (size_t)width * 3 + (size_t)dx * 3;

    if (src_base_index >= (size_t)width * (size_t)height * 3 - 2)
    {
        return;
    }

    // 写入切片
    const size_t dst_base_index = (size_t)slice_idx * (size_t)slice_width * (size_t)slice_height * 3 +
                                  (size_t)y * (size_t)slice_width * 3 + (size_t)x * 3;

    if (dst_base_index >=
        (size_t)slice_num_h * (size_t)slice_num_v * (size_t)slice_width * (size_t)slice_height * 3 - 2)
    {
        return;
    }

    outs[dst_base_index + 0] = image[src_base_index + 0]; // B
    outs[dst_base_index + 1] = image[src_base_index + 1]; // G
    outs[dst_base_index + 2] = image[src_base_index + 2]; // R
}

} // namespace cuda