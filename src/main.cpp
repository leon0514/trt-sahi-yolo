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


int main()
{
    run_yolov5();
    run_yolov5_sahi();
    run_yolo11obb();
    run_yolo11obb_sahi();
    run_yolo11seg();
    run_yolo11seg_sahi();
    run_yolo11pose();
    run_yolo11pose_sahi();
    run_yolo11();
    run_yolo11_sahi();
    return 0;
}