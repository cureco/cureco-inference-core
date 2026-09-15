"""Public input signatures and task result types for the native SDK bindings."""
from typing import Any, Literal, TypeAlias, TypedDict
import numpy as np
from numpy.typing import NDArray

IMG_TENSOR_NAME = "img_tensor"
PARAM_TENSOR_NAME = "param_tensor"
OUTPUT_TENSOR_NAME = "output"


class ModelInputInfo(TypedDict):
    name: str
    shape: list[Any]
    type: str


class ModelOutputInfo(TypedDict):
    name: str
    shape: list[Any]
    type: str


class ClassificationResult(TypedDict):
    label: str
    confidence: float


class DetectionResult(TypedDict):
    class_id: int
    label: str
    confidence: float
    box: list[float]


class SegmentationResult(TypedDict):
    task: Literal["segmentation"]
    width: int
    height: int
    classes: dict[str, str]
    mask_policy: dict[str, Any] | None
    mask: NDArray[np.int32]


InferenceResult: TypeAlias = list[ClassificationResult] | list[DetectionResult] | SegmentationResult
