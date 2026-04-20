"""推論の抽象基底クラス。

新しいモデルタイプ（検出等）を追加する場合は、
`InferenceBase` を継承して `preprocess` / `postprocess` を実装します。
"""

from PIL import Image
import numpy as np
from ..types import ModelInputInfo, ModelOutputInfo
from typing import Any, TypedDict, List


class PreprocessingDict(TypedDict, total=False):
    """前処理設定の型定義。

    ONNXモデルのメタデータに埋め込まれた正規化パラメータを表します。

    Attributes:
        normalize_mean: チャンネルごとの平均値（例: `[0.485, 0.456, 0.406]`）
        normalize_std: チャンネルごとの標準偏差（例: `[0.229, 0.224, 0.225]`）
    """
    normalize_mean: List[float]
    normalize_std: List[float]


class InferenceBase:
    """推論処理の抽象基底クラス。

    モデルタイプごとの前処理・後処理を定義するためのベースクラスです。
    サブクラスでは `preprocess` と `postprocess` を実装してください。

    Attributes:
        supported_model_types: このクラスが対応するモデルタイプ。
            サブクラスでオーバーライドします。

    Examples:
        カスタム推論タイプの実装:

        ```python
        class InferenceDetection(InferenceBase):
            supported_model_types = 'detection'

            def preprocess(self, image, params):
                ...

            def postprocess(self, output_data):
                ...
        ```
    """

    supported_model_types = None

    def __init__(self, idx_to_class: dict[str, str], preprocessing: PreprocessingDict, model_inputs: list[ModelInputInfo], model_outputs: list[ModelOutputInfo]):
        """基底クラスを初期化する。

        Args:
            idx_to_class: クラスインデックスからクラス名へのマッピング。
            preprocessing: 正規化パラメータなどの前処理設定。
            model_inputs: モデルの入力テンソル情報。
            model_outputs: モデルの出力テンソル情報。
        """
        self._idx_to_class = idx_to_class
        self._preprocessing = preprocessing
        self._model_inputs = model_inputs
        self._model_outputs = model_outputs

    def preprocess(self, image: Image.Image, params: list[float]) -> dict[str, np.ndarray]:
        """入力画像を前処理する。

        Args:
            image: 推論対象のRGB画像。
            params: モデルに渡す追加パラメータ。

        Returns:
            テンソル名をキーとした前処理済みnumpy配列の辞書。

        Raises:
            NotImplementedError: サブクラスで実装されていない場合。
        """
        raise NotImplementedError("preprocess method is not implemented")

    def postprocess(self, output_data: list[np.ndarray]) -> list[dict[str, Any]]:
        """推論出力を後処理する。

        Args:
            output_data: ONNXランタイムの出力テンソルのリスト。

        Returns:
            後処理済みの推論結果のリスト。

        Raises:
            NotImplementedError: サブクラスで実装されていない場合。
        """
        raise NotImplementedError("postprocess method is not implemented")

    @property
    def idx_to_class(self) -> dict[str, str]:
        """クラスインデックスからクラス名へのマッピング。"""
        return self._idx_to_class

    @property
    def preprocessing(self) -> PreprocessingDict:
        """前処理設定。"""
        return self._preprocessing

    @property
    def model_inputs(self) -> list[ModelInputInfo]:
        """モデルの入力テンソル情報。"""
        return self._model_inputs

    @property
    def model_outputs(self) -> list[ModelOutputInfo]:
        """モデルの出力テンソル情報。"""
        return self._model_outputs
