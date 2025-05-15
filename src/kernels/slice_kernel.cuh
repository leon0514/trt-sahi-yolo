
#ifndef SLICE_KERNEL_CUH__
#define SLICE_KERNEL_CUH__

#include "common/image.hpp"
#include "common/memory.hpp"
#include "opencv2/opencv.hpp"
#include <vector>

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
                             const int *__restrict__ slice_start_point);
}

#endif
