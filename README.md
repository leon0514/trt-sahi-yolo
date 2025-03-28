# TRT-SAHI-YOLO

## 项目简介

**TRT-SAHI-YOLO** 是一个基于 **SAHI** 图像切割和 **TensorRT** 推理引擎的目标检测系统。该项目结合了高效的图像预处理与加速推理技术，旨在提供快速、精准的目标检测能力。通过切割大图像成多个小块进行推理，并应用非极大值抑制（NMS）来优化检测结果，最终实现对物体的精确识别。

## 功能特性

1. **SAHI 图像切割**  
   利用 CUDA 实现 **SAHI** 的功能将输入图像切割成多个小块，支持重叠切割，以提高目标检测的准确性，特别是在边缘和密集物体区域。

2. **TensorRT 推理**  
   使用 **TensorRT** 进行深度学习模型推理加速。
   目前支持 **TensorRT8** 和 **TensorRT10** API


## 注意事项
1. 模型需要是动态batch的
2. 如果模型切割后的数量大于batch的最大数量会导致无法推理
3. **TensorRT 10**在执行推理的时候需要指定输入和输出的名称，名称可以在netron中查看
   ```C++
   #ifdef TRT10
   if (!trt_->forward(std::unordered_map<std::string, const void *>{
            { "images", input_buffer_.gpu() }, 
            { "output0", bbox_predict_.gpu() }
      }, stream_))
   {
      printf("Failed to tensorRT forward.");
      return {};
   }
   #else
   std::vector<void *> bindings{input_buffer_.gpu(), bbox_output_device};
   if (!trt_->forward(bindings, stream)) 
   {
      printf("Failed to tensorRT forward.");
      return {};
   }
   #endif
   ```
4. yolov8和yolov11模型导出的onnx输出shape是 1x84x8400 ，需要使用v8trans.py将输出转换为1x8400x84 

## 关于 **sahi** 后处理说明
与原始的多bacth后处理有一些改变。
1. 内存显存申请 
```diff
- output_boxarray_.gpu(batch_size * (MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
- output_boxarray_.cpu(batch_size * (MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));

+ output_boxarray_.gpu(MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
+ output_boxarray_.cpu(MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
```

- 一整张图即使分为了多个batch，最多也只分配MAX_IMAGE_BOXES个框
2. decode
```diff
- float *boxarray_device =
-      output_boxarray_.gpu() + ib * (MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
+ float *boxarray_device = output_boxarray_.gpu();
float *affine_matrix_device = affine_matrix_.gpu();
float *image_based_bbox_output =
      bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
if (yolo_type_ == YoloType::YOLOV5)
{
      decode_kernel_invoker_v5(image_based_bbox_output, bbox_head_dims_[1], num_classes_,
                        bbox_head_dims_[2], confidence_threshold_, nms_threshold_,
                        affine_matrix_device, boxarray_device, box_count, MAX_IMAGE_BOXES, start_x, start_y, stream_);
}
else if (yolo_type_ == YoloType::YOLOV8 || yolo_type_ == YoloType::YOLOV11)
{
      decode_kernel_invoker_v8(image_based_bbox_output, bbox_head_dims_[1], num_classes_,
                        bbox_head_dims_[2], confidence_threshold_, nms_threshold_,
                        affine_matrix_device, boxarray_device, box_count, MAX_IMAGE_BOXES, start_x, start_y, stream_);
}
```
- 单独使用一个变量`box_count`记录目前有效的框的数量
- decode时增加每个子图对应原图的起始点坐标`(start_x, start_y)`, 映射回原图坐标
```C++
int index = atomicAdd(box_count, 1);
if (index >= max_image_boxes) return;
```
- 上一张子图计算有效框的结束点是下一张子图的开始，通过`box_count`控制

3. nms
```c++
float *boxarray_device =  output_boxarray_.gpu();
fast_nms_kernel_invoker(boxarray_device, box_count, MAX_IMAGE_BOXES, nms_threshold_, stream_);
```
- 最后对所有子图合在一起的结果做nms，不是每个子图单独做nms。


## C++ 使用
```C++
cv::Mat image = cv::imread("inference/persons.jpg");
auto yolo = yolo::load("helmetv5.engine", yolo::YoloType::YOLOV5);
if (yolo == nullptr) return;
auto objs = yolo->forward(tensor::cvimg(image));
printf("objs size : %d\n", objs.size());
```

## 结果对比
<div align="center">
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/sliced.jpg?raw=true" width="45%"/>
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/no_sliced.jpg?raw=true" width="45%"/>
</div>

## 新增Yolo11-Pose
<div align="center">
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/yolo11poseSlicedInfer.jpg?raw=true" width="45%"/>
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/yolo11poseNoSlicedInfer.jpg?raw=true" width="45%"/>
</div>

## 新增Yolo11-seg
<div align="center">
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/yolo11seg-slice.jpg?raw=true" width="45%"/>
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/yolo11seg-noslice.jpg?raw=true" width="45%"/>
</div>

代码整理中


## 速度对比
| 显卡   | 模型   | 切割数量 | 运行次数 | 时间       |
|--------|--------|----------|----------|------------|
| RTX 3090 | YOLOv8n | 1       | 100     | 116.69206 ms |
| RTX 3090 | YOLOv8n | 6       | 100     | 353.99503 ms |
| RTX 3090 | YOLOv8n | 12      | 100     | 620.60980 ms |
| RTX 3090 | YOLOv5s | 1       | 100     | 133.62320 ms |
| RTX 3090 | YOLOv5s | 6       | 100     | 401.84650 ms |
| RTX 3090 | YOLOv5s | 12      | 100     | 682.81891 ms |

对sahi的cuda实现做了优化，速度应该会更快一点，但是没有之前相同的环境测试了。

## TensorRT8 API支持
在Makefile中通过 **TRT_VERSION** 来控制编译哪个版本的 **TensorRT** 封装文件

## 优化文字显示
目标检测模型识别到多个目标时，在图上显示文字可能会有重叠，导致类别置信度显示被遮挡。
优化了目标文字显示，尽可能改善遮挡情况    
详细说明见 [目标检测可视化文字重叠](https://www.jianshu.com/p/a6e289df4b90)
<div align="center">
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/sliced_text.jpg?raw=true" width="100%"/>
</div>

## 添加Python支持
使用pybind11封装程序

### 生成存根文件
```shell

pip install pybind11-stubgen

cd workspace # workspace 是 trtsahiyolo.so所在目录
export PYTHONPATH=`pwd`
pybind11-stubgen trtsahiyolo.so -o ./

```

### Python 使用
```python
import trtsahiyolo
from trtsahiyolo import YoloType
import cv2

model = trtsahiyolo.TrtSahiYolo("yolo11s.engine", YoloType.YOLOV11, 0, 0.3, 0.45)

frame = cv2.imread("test.jpg")

result = model.autoSliceForward(frame)

print(result)
```

## TODO
- [x] **NMS 实现**：完成所有子图的 NMS 处理逻辑，去除冗余框。已完成
- [x] **TensorRT8支持**：完成使用 **TensorRT8** 和 **TensorRT10** API
- [x] **Python支持**：使用 **Pybind11** 封装，使用 **Pyton** 调用
- [ ] **更多模型支持**：添加对其他 YOLO 模型版本的支持。目前支持 **YOLO11/YOLO11-Pose/YOLO11-SEG/YOLOv8/YOLOv5**

