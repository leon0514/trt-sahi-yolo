from __future__ import annotations
import numpy
import typing
__all__ = ['Box', 'DetResult', 'KeyPoint', 'ModelType', 'OBBox', 'TrtSahi', 'YOLO11', 'YOLO11OBB', 'YOLO11OBBSAHI', 'YOLO11POSE', 'YOLO11POSESAHI', 'YOLO11SAHI', 'YOLO11SEG', 'YOLO11SEGSAHI', 'YOLOV5', 'YOLOV5SAHI']
class Box:
    bottom: float
    class_name: str
    left: float
    right: float
    score: float
    top: float
    def __repr__(self) -> str:
        ...
class DetResult:
    box: Box
    keypoints: list[KeyPoint]
    obb: OBBox
    seg: numpy.ndarray
class KeyPoint:
    vis: float
    x: float
    y: float
    def __repr__(self) -> str:
        ...
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
class OBBox:
    angle: float
    class_name: str
    cx: float
    cy: float
    height: float
    score: float
    width: float
    def __repr__(self) -> str:
        ...
class TrtSahi:
    def __init__(self, model_path: str, model_type: ModelType, names: list[str], gpu_id: int, confidence_threshold: float, nms_threshold: float, max_batch_size: int, auto_slice: bool, slice_width: int, slice_height: int, slice_horizontal_ratio: float, slice_vertical_ratio: float) -> None:
        ...
    def forwards(self, images: list[numpy.ndarray]) -> list[list[DetResult]]:
        ...
    @property
    def valid(self) -> bool:
        ...
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
