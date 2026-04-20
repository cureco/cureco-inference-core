"""型定義とテンソル名定数。

推論パイプライン全体で使用される型定義と、
ONNXモデルの入出力テンソル名の定数を提供します。
"""

from typing import List, TypedDict, Any
import numpy as np

IMG_TENSOR_NAME = 'img_tensor'
"""画像入力テンソルの名前。"""

PARAM_TENSOR_NAME = 'param_tensor'
"""パラメータ入力テンソルの名前。"""

OUTPUT_TENSOR_NAME = 'output'
"""出力テンソルの名前。"""


class ModelInputInfo(TypedDict):
    """モデル入力情報の型定義。

    Attributes:
        name: テンソル名（例: `'img_tensor'`, `'param_tensor'`）
        shape: テンソルの形状（例: `[1, 3, 224, 224]`）
        type: データ型（例: `'tensor(float)'`）
    """
    name: str
    shape: List[Any]
    type: str


class ModelOutputInfo(TypedDict):
    """モデル出力情報の型定義。

    Attributes:
        name: テンソル名（例: `'output'`）
        shape: テンソルの形状（例: `[1, 10]`）
        type: データ型（例: `'tensor(float)'`）
    """
    name: str
    shape: List[Any]
    type: str


class InferenceResult(TypedDict):
    """推論結果の型定義。

    Attributes:
        class_name: 予測されたクラス名
        confidence: 信頼度（0.0〜1.0）
    """
    class_name: str
    confidence: float


class PreprocessedInput(TypedDict):
    """前処理済み入力データの型定義。

    Attributes:
        img_tensor: 前処理済み画像テンソル。形状は `(1, C, H, W)`
        param_tensor: パラメータテンソル。形状は `(1, N)`
    """
    img_tensor: np.ndarray
    param_tensor: np.ndarray


class ClassificationResult(TypedDict):
    """分類結果の型定義。

    Attributes:
        label: 予測されたクラスのラベル名
        confidence: 信頼度（0.0〜1.0）
    """
    label: str
    confidence: float
