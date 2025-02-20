#include "common/timer.hpp"

void v11NoSlicedInfer();
void v11SlicedInfer();
void v5NoSlicedInfer();
void v5SlicedInfer();

void yolo11poseSlicedInfer();
void yolo11poseNoSlicedInfer();

void SpeedTest();



int main()
{
    yolo11poseSlicedInfer();
    yolo11poseNoSlicedInfer();
    // v11SlicedInfer();
    // v5SlicedInfer();
    // SpeedTest();
    return 0;
}