"""CurecoInference — ONNX推論エンジンのコアモジュール。

ONNXモデルのロード、Execution Providerの自動選択、
前処理・推論・後処理のパイプライン実行を提供します。
"""

import logging
import onnxruntime
import json
from PIL import Image
import numpy as np
from .inferences.classification import InferenceClassification
from .types import (
    ModelInputInfo, ModelOutputInfo, InferenceResult,
    IMG_TENSOR_NAME, PARAM_TENSOR_NAME, OUTPUT_TENSOR_NAME,
)
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


def _decode_metadata_value(key: str, raw: Any) -> Any:
    """ONNXメタデータの値を1つデコードする。

    **規約: ONNXメタデータの値は「すべて」JSONエンコードされている。**

    ONNXの `metadata_props` は値に文字列しか持てないため、メタデータの値は
    辞書・数値・文字列を区別せず `json.dumps` で書き込む。
    したがって読む側は「すべての値を `json.loads`
    して読む」。素の文字列として扱うと、

      - `dataset_type` は `'"classification"'`（引用符込み）になり、
        `== 'classification'` の比較が**必ず外れる**
      - 逆に書き出し側が素の文字列で書くと、`dataset_name = "20260420"` のような
        数字だけの名前が読み取り時に**整数 20260420 になってしまう**

    という食い違いが起きる。一律JSONにすることで型が保たれる。

    規約に従っていない値（古いモデルや手作りのモデル）は、警告を出したうえで
    生の値をそのまま返す。ここで例外を投げると推論そのものが止まってしまい、
    害のほうが大きいため。

    Args:
        key: メタデータのキー名（警告メッセージ用）。
        raw: `custom_metadata_map` から取り出した生の値。

    Returns:
        デコード済みの値。デコードできなかった場合は生の値。
    """
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        logger.warning(
            f"ONNXメタデータ '{key}' がJSONとして読めません。"
            f"メタデータの値はすべてJSONエンコードされている必要があります。"
            f"生の値をそのまま使います: {raw!r}"
        )
        return raw


class CurecoInference:
    """ONNX推論エンジン。

    Curecoの学習プラットフォームでエクスポートしたONNXモデルを使用して、
    画像分類などの推論を実行します。

    モデルのメタデータから前処理パラメータやクラスマッピングを自動的に読み取り、
    利用可能なハードウェア（NPU/GPU/CPU）を自動検出して最適なExecution Providerを選択します。

    Examples:
        基本的な推論:

        ```python
        from PIL import Image
        from cureco_inference_core import CurecoInference

        engine = CurecoInference()
        engine.load_model("model.onnx")

        image = Image.open("test.jpg").convert("RGB")
        results = engine.inference(image, params=[1.0])

        for result in results:
            print(f"{result['label']}: {result['confidence']:.2%}")
        ```

        メタデータのみの軽量読み取り:

        ```python
        classes = CurecoInference.read_classes("model.onnx")
        print(classes)  # ['cat', 'dog', ...]
        ```
    """

    def __init__(self):
        self._onnx_session = None
        self._model_metadata = {}
        self._idx_to_class = {}
        self._model_inputs: List[ModelInputInfo] = []
        self._model_outputs: List[ModelOutputInfo] = []
        self._model_type = ''

    def load_model(self, model_path: str, providers: Optional[List[str]] = None):
        """ONNXモデルをロードする。

        モデルファイルを読み込み、メタデータ（前処理設定、クラスマッピング等）を
        自動的にパースします。

        Args:
            model_path: ONNXモデルファイルのパス。
            providers: 使用するExecution Providerのリスト。
                指定しない場合は利用可能なプロバイダーから自動選択されます。
                優先順位: QNN → VitisAI → OpenVINO → CUDA → DirectML → RKNPU → CPU

        Raises:
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument:
                モデルファイルが不正な場合。
            FileNotFoundError: モデルファイルが存在しない場合。
        """
        if providers is None:
            providers = self._get_available_providers()

        self._onnx_session = onnxruntime.InferenceSession(model_path, providers=providers)
        custom_metadata = self._onnx_session.get_modelmeta().custom_metadata_map

        # メタデータの値はすべてJSONエンコードされている（`_decode_metadata_value` 参照）。
        # 辞書として使う項目は、規約違反で辞書にならなかった場合も空辞書として推論を続ける。
        decoded = {}
        for key in ('augmentations', 'preprocessing', 'idx_to_class', 'label_slugs'):
            value = _decode_metadata_value(key, custom_metadata.get(key, '{}'))
            if not isinstance(value, dict):
                logger.warning(f"ONNXメタデータ '{key}' が辞書ではありません。空として扱います。")
                value = {}
            decoded[key] = value

        self._augmentations = decoded['augmentations']
        self._preprocessing = decoded['preprocessing']
        self._idx_to_class = decoded['idx_to_class']
        self._label_slugs = decoded['label_slugs']
        # `dataset_type` も同じ規約でJSONエンコードされている（実物の値は '"classification"'）。
        # 生のまま比較するとモデルタイプの判定が必ず外れるので、必ずデコードしてから持つ。
        raw_model_type = custom_metadata.get('dataset_type')
        self._model_type = (
            '' if raw_model_type is None
            else str(_decode_metadata_value('dataset_type', raw_model_type))
        )

        self._model_inputs = self._get_inputs()
        self._model_outputs = self._get_outputs()

        logger.info(f"使用中のExecution Provider: {self._onnx_session.get_providers()}")

    def inference(self, image: Image.Image, params: list[float]) -> List[InferenceResult]:
        """画像に対して推論を実行する。

        モデルタイプに応じた前処理・推論・後処理のパイプラインを実行します。

        Args:
            image: 推論対象のRGB画像。
            params: モデルに渡す追加パラメータのリスト。
                パラメータ入力を持たないモデルの場合は空リスト `[]` を指定します。

        Returns:
            推論結果のリスト。各要素は `label`（クラス名）と `confidence`（信頼度）を含みます。

        Raises:
            RuntimeError: モデルがロードされていない場合。
            NotImplementedError: 未対応のモデルタイプ（物体検出等）の場合。
        """
        if self.model_type == 'detection':
            # 物体検出は未実装。ここを素通りさせると `inference` が未定義のまま使われ、
            # 分かりにくい UnboundLocalError になるので、はっきり断る。
            raise NotImplementedError(
                "物体検出モデルには対応していません（この版は画像分類のみ対応）。"
                "dataset_type が 'classification' のモデルを指定してください。"
            )
        # メタデータを持たない古いモデルは model_type が空になる。従来どおり分類として扱う。
        inference = InferenceClassification(self._idx_to_class, self._preprocessing, self._model_inputs, self._model_outputs)

        input_data = inference.preprocess(image, params)
        output_data = self._onnx_session.run(None, input_data)
        result_data = inference.postprocess(output_data)

        return result_data

    def get_available_providers(self) -> List[str]:
        """利用可能なExecution Providerのリストを取得する。

        Returns:
            利用可能なプロバイダー名のリスト。
        """
        return onnxruntime.get_available_providers()

    def get_current_providers(self) -> List[str]:
        """現在使用中のExecution Providerのリストを取得する。

        Returns:
            現在使用中のプロバイダー名のリスト。モデル未ロード時は空リスト。
        """
        if self._onnx_session is None:
            return []
        return self._onnx_session.get_providers()

    @property
    def idx_to_class(self) -> dict[int, str]:
        """クラスインデックスからクラス名へのマッピング。"""
        return self._idx_to_class

    @property
    def model_type(self) -> str:
        """モデルタイプ（例: `'classification'`, `'detection'`）。"""
        return self._model_type

    @property
    def classes(self) -> list[str]:
        """モデルが認識可能なクラス名のリスト。"""
        return list(self._idx_to_class.values())

    @property
    def model_inputs(self) -> List[ModelInputInfo]:
        """モデルの入力テンソル情報のリスト。"""
        return self._model_inputs

    @property
    def model_outputs(self) -> List[ModelOutputInfo]:
        """モデルの出力テンソル情報のリスト。"""
        return self._model_outputs

    @staticmethod
    def read_metadata(model_path: str) -> Dict[str, str]:
        """ONNXモデルのメタデータのみを軽量に読み取る。

        グラフ最適化を無効化しCPUのみでセッションを作成するため、
        `load_model` よりも高速です。

        !!! warning "値はすべてJSONエンコードされています"
            返すのは ONNX に格納された**生のまま**の値です。ONNXメタデータの値は
            すべて `json.dumps` された文字列なので（`dataset_type` は
            `'"classification"'`、`num_classes` は `'6'`）、利用側で `json.loads`
            してから使ってください。理由は `_decode_metadata_value` の説明を参照。

        Args:
            model_path: ONNXモデルファイルのパス。

        Returns:
            メタデータのキー・値辞書（値はJSON文字列のまま）。
        """
        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = onnxruntime.InferenceSession(
            model_path, sess_options=opts, providers=['CPUExecutionProvider']
        )
        return session.get_modelmeta().custom_metadata_map

    @staticmethod
    def read_classes(model_path: str) -> List[str]:
        """ONNXモデルからクラス名リストのみを軽量に取得する。

        内部で `read_metadata` を使用してメタデータを読み取り、
        `idx_to_class` フィールドからクラス名を抽出します。

        Args:
            model_path: ONNXモデルファイルのパス。

        Returns:
            クラス名のリスト。

        Examples:
            ```python
            classes = CurecoInference.read_classes("model.onnx")
            print(classes)  # ['cat', 'dog', 'bird']
            ```
        """
        metadata = CurecoInference.read_metadata(model_path)
        idx_to_class = json.loads(metadata.get('idx_to_class', '{}'))
        return list(idx_to_class.values())

    def _get_available_providers(self) -> List[str]:
        """利用可能なExecution Providerを優先順位付きで取得する。"""
        preferred_providers = []
        available_providers = onnxruntime.get_available_providers()

        npu_gpu_providers = [
            'QNNExecutionProvider',      # Qualcomm NPU
            'VitisAIExecutionProvider',  # AMD Ryzen AI
            'OpenVINOExecutionProvider', # Intel NPU
            'CUDAExecutionProvider',     # NVIDIA GPU
            'DmlExecutionProvider',      # DirectML (Windows GPU)
            'RKNPUExecutionProvider',    # Rockchip NPU
        ]

        for provider in npu_gpu_providers:
            if provider in available_providers:
                preferred_providers.append(provider)

        if 'CPUExecutionProvider' in available_providers:
            preferred_providers.append('CPUExecutionProvider')

        return preferred_providers

    def _get_inputs(self) -> List[ModelInputInfo]:
        """モデルの入力テンソル情報を取得する。"""
        input_list = []
        for onnx_input in self._onnx_session.get_inputs():
            if onnx_input.name in [IMG_TENSOR_NAME, PARAM_TENSOR_NAME]:
                input_list.append({
                    'name': onnx_input.name,
                    'shape': onnx_input.shape,
                    'type': onnx_input.type,
                })
        return input_list

    def _get_outputs(self) -> List[ModelOutputInfo]:
        """モデルの出力テンソル情報を取得する。"""
        output_list = []
        for onnx_output in self._onnx_session.get_outputs():
            if onnx_output.name in [OUTPUT_TENSOR_NAME]:
                output_list.append({
                    'name': onnx_output.name,
                    'shape': onnx_output.shape,
                    'type': onnx_output.type,
                })
        return output_list
