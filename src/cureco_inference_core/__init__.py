"""Cureco Inference — ONNX推論パッケージ。

Curecoの学習プラットフォームでエクスポートしたONNXモデルを使用して、
画像分類などの推論を実行するためのパッケージです。

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
    PreprocessedInput,
    ClassificationResult,
    IMG_TENSOR_NAME,
    PARAM_TENSOR_NAME,
    OUTPUT_TENSOR_NAME,
)

__all__ = [
    'CurecoInference',
    'ModelInputInfo',
    'ModelOutputInfo',
    'InferenceResult',
    'PreprocessedInput',
    'ClassificationResult',
    'IMG_TENSOR_NAME',
    'PARAM_TENSOR_NAME',
    'OUTPUT_TENSOR_NAME',
]
