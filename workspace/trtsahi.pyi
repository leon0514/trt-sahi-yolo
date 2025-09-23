"""
Python bindings for TensorRT SAHI inference library
"""
from __future__ import annotations
import numpy
import typing
__all__ = ['Box', 'DETECTION', 'DetectionBox', 'DetectionBoxArray', 'ModelType', 'OBB', 'Obb', 'ObjectType', 'POSE', 'Pose', 'PosePoint', 'SEGMENTATION', 'Segmentation', 'TrtSahi', 'UNKNOW', 'YOLO11', 'YOLO11OBB', 'YOLO11OBBSAHI', 'YOLO11POSE', 'YOLO11POSESAHI', 'YOLO11SAHI', 'YOLO11SEG', 'YOLO11SEGSAHI', 'YOLOV5', 'YOLOV5SAHI']
class Box:
    bottom: float
    left: float
    right: float
    top: float
    def __init__(self, left: float = 0, top: float = 0, right: float = 0, bottom: float = 0) -> None:
        ...
    def __repr__(self) -> str:
        ...
class DetectionBox:
    box: Box
    class_id: int
    class_name: str
    obb: Obb | None
    pose: Pose | None
    score: float
    segmentation: Segmentation | None
    type: ObjectType
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
class DetectionBoxArray:
    def __bool__(self: list[DetectionBox]) -> bool:
        """
        Check whether the list is nonempty
        """
    @typing.overload
    def __delitem__(self: list[DetectionBox], arg0: int) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self: list[DetectionBox], arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    @typing.overload
    def __getitem__(self: list[DetectionBox], s: slice) -> list[DetectionBox]:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self: list[DetectionBox], arg0: int) -> DetectionBox:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: list[DetectionBox]) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None:
        ...
    def __iter__(self: list[DetectionBox]) -> typing.Iterator[DetectionBox]:
        ...
    def __len__(self: list[DetectionBox]) -> int:
        ...
    def __repr__(self: list[DetectionBox]) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __setitem__(self: list[DetectionBox], arg0: int, arg1: DetectionBox) -> None:
        ...
    @typing.overload
    def __setitem__(self: list[DetectionBox], arg0: slice, arg1: list[DetectionBox]) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self: list[DetectionBox], x: DetectionBox) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self: list[DetectionBox]) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def extend(self: list[DetectionBox], L: list[DetectionBox]) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self: list[DetectionBox], L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self: list[DetectionBox], i: int, x: DetectionBox) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self: list[DetectionBox]) -> DetectionBox:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self: list[DetectionBox], i: int) -> DetectionBox:
        """
        Remove and return the item at index ``i``
        """
class ModelType:
    """
    Members:
    
      YOLOV5
    
      YOLO11
    
      YOLO11POSE
    
      YOLO11SEG
    
      YOLO11OBB
    
      YOLOV5SAHI
    
      YOLO11SAHI
    
      YOLO11POSESAHI
    
      YOLO11SEGSAHI
    
      YOLO11OBBSAHI
    """
    YOLO11: typing.ClassVar[ModelType]  # value = <ModelType.YOLO11: 2>
    YOLO11OBB: typing.ClassVar[ModelType]  # value = <ModelType.YOLO11OBB: 8>
    YOLO11OBBSAHI: typing.ClassVar[ModelType]  # value = <ModelType.YOLO11OBBSAHI: 9>
    YOLO11POSE: typing.ClassVar[ModelType]  # value = <ModelType.YOLO11POSE: 4>
    YOLO11POSESAHI: typing.ClassVar[ModelType]  # value = <ModelType.YOLO11POSESAHI: 5>
    YOLO11SAHI: typing.ClassVar[ModelType]  # value = <ModelType.YOLO11SAHI: 3>
    YOLO11SEG: typing.ClassVar[ModelType]  # value = <ModelType.YOLO11SEG: 6>
    YOLO11SEGSAHI: typing.ClassVar[ModelType]  # value = <ModelType.YOLO11SEGSAHI: 7>
    YOLOV5: typing.ClassVar[ModelType]  # value = <ModelType.YOLOV5: 0>
    YOLOV5SAHI: typing.ClassVar[ModelType]  # value = <ModelType.YOLOV5SAHI: 1>
    __members__: typing.ClassVar[dict[str, ModelType]]  # value = {'YOLOV5': <ModelType.YOLOV5: 0>, 'YOLO11': <ModelType.YOLO11: 2>, 'YOLO11POSE': <ModelType.YOLO11POSE: 4>, 'YOLO11SEG': <ModelType.YOLO11SEG: 6>, 'YOLO11OBB': <ModelType.YOLO11OBB: 8>, 'YOLOV5SAHI': <ModelType.YOLOV5SAHI: 1>, 'YOLO11SAHI': <ModelType.YOLO11SAHI: 3>, 'YOLO11POSESAHI': <ModelType.YOLO11POSESAHI: 5>, 'YOLO11SEGSAHI': <ModelType.YOLO11SEGSAHI: 7>, 'YOLO11OBBSAHI': <ModelType.YOLO11OBBSAHI: 9>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Obb:
    angle: float
    cx: float
    cy: float
    height: float
    width: float
    def __init__(self, cx: float = 0, cy: float = 0, width: float = 0, height: float = 0, angle: float = 0) -> None:
        ...
class ObjectType:
    """
    Members:
    
      UNKNOW
    
      DETECTION
    
      POSE
    
      OBB
    
      SEGMENTATION
    """
    DETECTION: typing.ClassVar[ObjectType]  # value = <ObjectType.DETECTION: 7>
    OBB: typing.ClassVar[ObjectType]  # value = <ObjectType.OBB: 2>
    POSE: typing.ClassVar[ObjectType]  # value = <ObjectType.POSE: 1>
    SEGMENTATION: typing.ClassVar[ObjectType]  # value = <ObjectType.SEGMENTATION: 3>
    UNKNOW: typing.ClassVar[ObjectType]  # value = <ObjectType.UNKNOW: -1>
    __members__: typing.ClassVar[dict[str, ObjectType]]  # value = {'UNKNOW': <ObjectType.UNKNOW: -1>, 'DETECTION': <ObjectType.DETECTION: 7>, 'POSE': <ObjectType.POSE: 1>, 'OBB': <ObjectType.OBB: 2>, 'SEGMENTATION': <ObjectType.SEGMENTATION: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Pose:
    points: list[PosePoint]
    def __init__(self) -> None:
        ...
class PosePoint:
    vis: float
    x: float
    y: float
    def __init__(self, x: float = 0, y: float = 0, vis: float = 0) -> None:
        ...
    def __repr__(self) -> str:
        ...
class Segmentation:
    mask: numpy.ndarray
    def __init__(self) -> None:
        ...
class TrtSahi:
    def __init__(self, model_path: str, model_type: ModelType, names: list[str], gpu_id: int = 0, confidence_threshold: float = 0.5, nms_threshold: float = 0.44999998807907104, max_batch_size: int = 32, auto_slice: bool = True, slice_width: int = 640, slice_height: int = 640, slice_horizontal_ratio: float = 0.3, slice_vertical_ratio: float = 0.3) -> None:
        ...
    def forwards(self, images: list[numpy.ndarray]) -> list[list[DetectionBox]]:
        """
        对一批图像执行推理，返回检测结果列表的列表
        """
    @property
    def valid(self) -> bool:
        """
        检查推理引擎是否有效
        """
DETECTION: ObjectType  # value = <ObjectType.DETECTION: 7>
OBB: ObjectType  # value = <ObjectType.OBB: 2>
POSE: ObjectType  # value = <ObjectType.POSE: 1>
SEGMENTATION: ObjectType  # value = <ObjectType.SEGMENTATION: 3>
UNKNOW: ObjectType  # value = <ObjectType.UNKNOW: -1>
YOLO11: ModelType  # value = <ModelType.YOLO11: 2>
YOLO11OBB: ModelType  # value = <ModelType.YOLO11OBB: 8>
YOLO11OBBSAHI: ModelType  # value = <ModelType.YOLO11OBBSAHI: 9>
YOLO11POSE: ModelType  # value = <ModelType.YOLO11POSE: 4>
YOLO11POSESAHI: ModelType  # value = <ModelType.YOLO11POSESAHI: 5>
YOLO11SAHI: ModelType  # value = <ModelType.YOLO11SAHI: 3>
YOLO11SEG: ModelType  # value = <ModelType.YOLO11SEG: 6>
YOLO11SEGSAHI: ModelType  # value = <ModelType.YOLO11SEGSAHI: 7>
YOLOV5: ModelType  # value = <ModelType.YOLOV5: 0>
YOLOV5SAHI: ModelType  # value = <ModelType.YOLOV5SAHI: 1>
