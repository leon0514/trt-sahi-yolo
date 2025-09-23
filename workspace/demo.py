import trtsahi
import time
import cv2


def yolov5():
    # Load the model
    names = ["person", "helmet"]
    model = trtsahi.TrtSahi(
        model_path="models/engine/helmet.engine",
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
    print(images[0].shape)
    # Run inference
    results = model.forwards(images)
    # Print results
    for result in results[0]:
        print(result.box)


def yolo11():
    # Load the model
    names = ["person", "helmet"]
    model = trtsahi.TrtSahi(
        model_path="models/engine/yolo11l.transd.engine",
        model_type=trtsahi.ModelType.YOLO11,
        names=names,
        gpu_id=0,
        confidence_threshold=0.5,
        nms_threshold=0.4,
        max_batch_size=32,
        auto_slice=False,
        slice_width=0,
        slice_height=0,
        slice_horizontal_ratio=0,
        slice_vertical_ratio=0
    )
    images = [cv2.imread("inference/wallhaven-9ddovd.jpg")]
    results = model.forwards(images)
    # Print results
    for result in results[0]:
        print(result.box)

def yolo11obb():
    names = ["plane", "ship", "storage tank", "baseball diamond", "tennis court",
            "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
            "small vehicle", "helicopter", "roundabout", "soccer ball field",
            "swimming pool", "container crane"]

    model = trtsahi.TrtSahi(
        model_path="models/yolo11m-obb.transd.engine",
        model_type=trtsahi.ModelType.YOLO11OBBSAHI,
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

    images = [cv2.imread("inference/car.jpg")]
    # Run inference
    results = model.forwards(images)
    # Print results
    for result in results[0]:
        print(result.box)

def yolo11pose():
    names = ["person"]

    model = trtsahi.TrtSahi(
        model_path="models/yolo11l-pose.transd.engine",
        model_type=trtsahi.ModelType.YOLO11POSESAHI,
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

def yolo11seg():
    names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"]

    model = trtsahi.TrtSahi(
        model_path="models/yolo11l-seg.transd.engine",
        model_type=trtsahi.ModelType.YOLO11SEGSAHI,
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
    print(images[0].shape)
    # Run inference
    results = model.forwards(images)
    # Print results
    for result in results[0]:
        print(result.box)

if __name__ == "__main__":
    yolo11()