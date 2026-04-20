"""画像分類の推論実装。

ソフトマックスによる確率変換と、クラスマッピングによるラベル付けを行います。
"""

from PIL import Image
import numpy as np
from .base import InferenceBase, PreprocessingDict
from ..types import (
    ModelInputInfo, ModelOutputInfo, PreprocessedInput, ClassificationResult,
    IMG_TENSOR_NAME, PARAM_TENSOR_NAME,
)
from typing import List


class InferenceClassification(InferenceBase):
    """画像分類の推論処理。

    前処理ではリサイズ・正規化・チャンネル変換を行い、
    後処理ではソフトマックスによる確率変換とクラスマッピングを行います。

    前処理パイプライン:
        1. モデルの入力サイズにリサイズ（BICUBIC補間）
        2. `[0, 255]` → `[0, 1]` に正規化
        3. メタデータの `normalize_mean` / `normalize_std` で標準化（設定がある場合）
        4. HWC → CHW に転置
        5. バッチ次元を追加 → `(1, C, H, W)`
    """

    supported_model_types = 'classification'

    def __init__(self, idx_to_class: dict[str, str], preprocessing: PreprocessingDict, model_inputs: list[ModelInputInfo], model_outputs: list[ModelOutputInfo]):
        super().__init__(idx_to_class, preprocessing, model_inputs, model_outputs)

    def preprocess(self, image: Image.Image, params: list[float]) -> PreprocessedInput:
        """入力画像を分類モデル用に前処理する。

        Args:
            image: 推論対象のRGB画像。任意のサイズ。
            params: モデルに渡す追加パラメータ。
                `param_tensor` 入力を持つモデルの場合に使用されます。

        Returns:
            テンソル名をキーとした前処理済みnumpy配列の辞書。

        Raises:
            ValueError: `params` の要素数がモデルの期待する次元数と一致しない場合。
        """
        input_data = {}
        for input_info in self.model_inputs:
            input_name = input_info['name']
            input_shape = input_info['shape']
            input_type = input_info['type']
            if input_name == IMG_TENSOR_NAME:
                img = image.resize((input_shape[2], input_shape[3]), resample=Image.Resampling.BICUBIC)
                img = np.array(img, dtype=np.float32)
                img = img / 255.0

                if 'normalize_mean' in self._preprocessing and 'normalize_std' in self._preprocessing:
                    mean = np.array(self._preprocessing['normalize_mean'], dtype=np.float32)
                    std = np.array(self._preprocessing['normalize_std'], dtype=np.float32)
                    img = (img - mean) / std

                img = img.transpose((2, 0, 1))
                img = img[np.newaxis, :, :, :]
                input_data[input_name] = img
            elif input_name == PARAM_TENSOR_NAME:
                params_array = np.array(params, dtype=np.float32)
                params_array = params_array[np.newaxis, :]
                if params_array.shape[1] != input_shape[1]:
                    raise ValueError(f"param_tensorのshapeが不正です。{input_shape[1]} != {params_array.shape[1]}")
                input_data[input_name] = params_array
        return input_data

    def postprocess(self, output_data: list[np.ndarray]) -> List[ClassificationResult]:
        """推論出力をクラスラベルと信頼度に変換する。

        ロジットにソフトマックスを適用して確率分布に変換し、
        最大確率のクラスをラベルとして返します。

        Args:
            output_data: ONNXランタイムの出力。`output_data[0]` がロジットテンソル。

        Returns:
            分類結果のリスト。各要素は `label`（クラス名）と `confidence`（信頼度）を含みます。
        """
        logits: np.ndarray = output_data[0]

        # 数値安定性のため最大値を引いてからソフトマックスを適用
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        confidence_list = probabilities.max(axis=1).tolist()

        pred_class_id_list = probabilities.argmax(axis=1).tolist()
        pred_class_name_list = [self._idx_to_class.get(str(pred_class_id_list[i]), "unknown") for i in range(len(pred_class_id_list))]

        result_list: List[ClassificationResult] = []
        for i in range(len(pred_class_name_list)):
            result_list.append({
                'label': pred_class_name_list[i],
                'confidence': confidence_list[i],
            })
        return result_list
