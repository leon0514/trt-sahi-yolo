#include "common/timer.hpp"

void run_yolov5();
void run_yolov5_sahi();
void run_yolo11obb();
void run_yolo11obb_sahi();
void run_yolo11seg();
void run_yolo11seg_sahi();
void run_yolo11pose();
void run_yolo11pose_sahi();
void run_yolo11();
void run_yolo11_sahi();

void run_dfine();
void run_dfine_sahi();


int main()
{
    // run_yolov5();
    // run_yolov5_sahi();
    // run_yolo11obb();
    // run_yolo11obb_sahi();
    // run_yolo11seg();
    // run_yolo11seg_sahi();
    // run_yolo11pose();
    // run_yolo11pose_sahi();
    // run_yolo11();
    // run_yolo11_sahi();
    run_dfine();
    run_dfine_sahi();
    return 0;
}

/*

/opt/nvidia/TensorRT-10.9.0.34/bin/trtexec  --onnx=models/onnx/yolo11l.transd.onnx \
    --minShapes=images:1x3x640x640 \
    --maxShapes=images:16x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --saveEngine=models/engine/yolo11l.transd.engine --fp16
*/