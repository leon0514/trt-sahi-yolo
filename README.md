# TRT-SAHI-YOLO

## 项目简介

**TRT-SAHI-YOLO** 是一个基于 **SAHI** 图像切割和 **TensorRT** 推理引擎的目标检测系统。该项目结合了高效的图像预处理与加速推理技术，旨在提供快速、精准的目标检测能力。通过切割大图像成多个小块进行推理，并应用非极大值抑制（NMS）来优化检测结果，最终实现对物体的精确识别。

## 功能特性

1. **SAHI 图像切割**  
   利用 CUDA 实现 **SAHI** 的功能将输入图像切割成多个小块，支持重叠切割，以提高目标检测的准确性，特别是在边缘和密集物体区域。

2. **TensorRT 推理**  
   使用 **TensorRT** 进行深度学习模型推理加速。
   目前支持 **TensorRT8** 和 **TensorRT10** API

## Engine 导出
1. 导出动态onnx
2. 对于yolov8、yolov11模型需要执行v8trans.py脚本对输出做一个转置
3. 使用trtexec 导出engine模型，指定为动态batch，需要将最大batch设置的大一些，避免切割图的数量大于最大batch

## 注意事项
1. 模型需要是动态batch的
2. 如果模型切割后的数量大于batch的最大数量会导致无法推理
3. **TensorRT 10**在执行推理的时候需要指定输入和输出的名称，名称可以在netron中查看
4. yolov8和yolov11模型导出的onnx输出shape是 1x84x8400 ，需要使用v8trans.py将输出转换为1x8400x84 

## 结果对比
### YOLOv5 检测
<div align="center">
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/yolov5.jpg?raw=true" width="45%"/>
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/yolov5sahi.jpg?raw=true" width="45%"/>
</div>

### YOLO11 检测
<div align="center">
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/yolo11.jpg?raw=true" width="45%"/>
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/yolo11sahi.jpg?raw=true" width="45%"/>
</div>

### YOLO11 姿态
<div align="center">
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/yolo11pose.jpg?raw=true" width="45%"/>
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/yolo11posesahi.jpg?raw=true" width="45%"/>
</div>

### YOLO11 分割
<div align="center">
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/yolo11seg.jpg?raw=true" width="45%"/>
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/yolo11segsahi.jpg?raw=true" width="45%"/>
</div>

### YOLO11 旋转目标检测
<div align="center">
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/yolo11obb.jpg?raw=true" width="45%"/>
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/yolo11obbsahi.jpg?raw=true" width="45%"/>
</div>

### D-FINE 检测
<div align="center">
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/dfine.jpg?raw=true" width="45%"/>
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/dfinesahi.jpg?raw=true" width="45%"/>
</div>

#### D-FINE 导出engin
```shell
trtexec  --onnx=models/onnx/dfine_l_obj2coco.onnx \
--minShapes=images:1x3x640x640,orig_target_sizes:1x2 \
--maxShapes=images:16x3x640x640,orig_target_sizes:16x2 \
--optShapes=images:1x3x640x640,orig_target_sizes:1x2 \
--saveEngine=models/engine/dfine_l_obj2coco.engine --fp16
```

## TensorRT8 API支持
在Makefile中通过 **TRT_VERSION** 来控制编译哪个版本的 **TensorRT** 封装文件

## 优化文字显示
目标检测模型识别到多个目标时，在图上显示文字可能会有重叠，导致类别置信度显示被遮挡。
优化了目标文字显示，尽可能改善遮挡情况    
详细说明见 [目标检测可视化文字重叠](https://www.jianshu.com/p/a6e289df4b90)

<div align="center">
   <img src="https://github.com/leon0514/trt-sahi-yolo/blob/main/assert/yolo11sahi.jpg?raw=true" width="100%"/>
</div>

## python 支持
```python
def yolov5():
    # Load the model
    names = ["person", "helmet"]
    model = trtsahi.TrtSahi(
        model_path="models/helmet.engine",
        model_type=trtsahi.ModelType.YOLOV5SAHI,
        names=names,
        gpu_id=0,
        confidence_threshold=0.5,
        nms_threshold=0.4,
        max_batch_size=32,
        auto_slice=True,
        slice_width=640,
        slice_height=640,
        slice_horizontal_ratio=0.5,
        slice_vertical_ratio=0.5
    )

    images = [cv2.imread("inference/persons.jpg")]
    # Run inference
    results = model.forwards(images)
    # Print results
    for result in results[0]:
        print(result.box)
```

## 环境依赖
- cuda
- opencv
- tensorrt
- python（如果需要python使用）

## 支持的模型
| 模型名称              |  是否支持sahi | 
|-----------------------|--------------|
| YOLO11                |  是           |
| YOLO11-Pose           |  是           |
| YOLO11-SEG            |  是           |
| YOLO11-Obb            |  是           |
| YOLOv8                |  是           |
| YOLOv5                |  是           |
| D-FINE                |  是           |

## TODO
- [x] **NMS 实现**：完成所有子图的 NMS 处理逻辑，去除冗余框。已完成
- [x] **TensorRT8支持**：完成使用 **TensorRT8** 和 **TensorRT10** API
- [x] **Python支持**：使用 **Pybind11** 封装，使用 **Pyton** 调用
- [ ] **更多模型支持**：添加对其他模型的支持。

