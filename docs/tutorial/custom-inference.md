# カスタム推論タイプの追加

Cureco Inferenceは `InferenceBase` を継承することで、新しいモデルタイプ（物体検出等）に対応できます。

## アーキテクチャ

```
CurecoInference (core.py)
│
├── model_type に応じてルーティング
│
├── InferenceClassification  ← classification
├── InferenceDetection       ← detection（例）
└── ...
```

`CurecoInference.inference()` はモデルのメタデータに含まれる `dataset_type` に基づいて、
適切な `InferenceBase` サブクラスにルーティングします。

## 実装手順

### 1. サブクラスの作成

`inferences/` ディレクトリに新しいファイルを作成します。

```python
# inferences/detection.py
from PIL import Image
import numpy as np
from .base import InferenceBase, PreprocessingDict
from ..types import ModelInputInfo, ModelOutputInfo, IMG_TENSOR_NAME
from typing import List


class InferenceDetection(InferenceBase):
    """物体検出の推論処理。"""

    supported_model_types = 'detection'

    def __init__(self, idx_to_class, preprocessing, model_inputs, model_outputs):
        super().__init__(idx_to_class, preprocessing, model_inputs, model_outputs)

    def preprocess(self, image: Image.Image, params: list[float]) -> dict[str, np.ndarray]:
        input_data = {}
        for input_info in self.model_inputs:
            if input_info['name'] == IMG_TENSOR_NAME:
                input_shape = input_info['shape']
                img = image.resize((input_shape[2], input_shape[3]))
                img = np.array(img, dtype=np.float32) / 255.0
                img = img.transpose((2, 0, 1))[np.newaxis, :, :, :]
                input_data[input_info['name']] = img
        return input_data

    def postprocess(self, output_data: list[np.ndarray]) -> list[dict]:
        # 検出結果のパース（モデルの出力形式に応じて実装）
        ...
```

### 2. サブパッケージへの登録

`inferences/__init__.py` にインポートを追加します。

```python
from .base import InferenceBase, PreprocessingDict
from .classification import InferenceClassification
from .detection import InferenceDetection  # 追加
```

### 3. コアへのルーティング追加

`core.py` の `inference()` メソッドに分岐を追加します。

```python
def inference(self, image, params):
    if self.model_type == 'classification':
        inference = InferenceClassification(...)
    elif self.model_type == 'detection':
        inference = InferenceDetection(...)   # 追加
    else:
        inference = InferenceClassification(...)

    input_data = inference.preprocess(image, params)
    output_data = self._onnx_session.run(None, input_data)
    return inference.postprocess(output_data)
```

## 前処理・後処理のガイドライン

### 前処理 (`preprocess`)

- 入力: `Image.Image`（任意サイズのRGB画像）+ `params`（追加パラメータ）
- 出力: `dict[str, np.ndarray]`（テンソル名→numpy配列の辞書）
- モデルの `model_inputs` から入力テンソルの形状を読み取ってリサイズする
- テンソル名の定数（`IMG_TENSOR_NAME` 等）を使用する

### 後処理 (`postprocess`)

- 入力: `list[np.ndarray]`（ONNX Runtimeの出力）
- 出力: 結果のリスト（形式はモデルタイプに応じて定義）
- 必要に応じて `types.py` に新しい結果型を追加する
