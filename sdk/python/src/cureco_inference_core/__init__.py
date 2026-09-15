"""Cureco Inference — ONNX推論パッケージ。

共通ネイティブSDKを使用するPythonバインディングです。
対応する契約のONNXモデルで分類・検出・セグメンテーションを実行します。

基本的な使い方:
    ```python
    from cureco_inference_core import CurecoInference

    engine = CurecoInference()
    engine.load_model("model.onnx")
    results = engine.inference(image, params=[1.0])
    ```
"""

from .core import CurecoInference
from .types import (
    ModelInputInfo,
    ModelOutputInfo,
    InferenceResult,
    ClassificationResult,
    DetectionResult,
    SegmentationResult,
    IMG_TENSOR_NAME,
    PARAM_TENSOR_NAME,
    OUTPUT_TENSOR_NAME,
)

__all__ = [
    'CurecoInference',
    'ModelInputInfo',
    'ModelOutputInfo',
    'InferenceResult',
    'ClassificationResult',
    'DetectionResult',
    'SegmentationResult',
    'IMG_TENSOR_NAME',
    'PARAM_TENSOR_NAME',
    'OUTPUT_TENSOR_NAME',
]

from .native import NativeInference
__all__.append("NativeInference")
