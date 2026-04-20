# クイックスタート

## 前提条件

- Python 3.10以上
- `onnxruntime` がインストール済み
- Curecoの学習プラットフォームでエクスポートしたONNXモデル

## インストール

```bash
pip install cureco-inference-core
```

## 基本的な使い方

### 1. モデルのロード

```python
from cureco_inference_core import CurecoInference

engine = CurecoInference()
engine.load_model("path/to/model.onnx")
```

`load_model` は以下を自動的に行います:

- 利用可能なExecution Provider（NPU/GPU/CPU）の検出と選択
- モデルメタデータ（前処理設定、クラスマッピング等）の読み取り

### 2. 推論の実行

```python
from PIL import Image

image = Image.open("test.jpg").convert("RGB")
results = engine.inference(image, params=[])
```

`image` は任意のサイズのRGB画像で構いません。モデルの入力サイズへのリサイズは自動的に行われます。

### 3. 結果の取得

```python
for result in results:
    print(f"ラベル: {result['label']}")
    print(f"信頼度: {result['confidence']:.2%}")
```

返り値は `ClassificationResult` のリストで、各要素には `label`（クラス名）と `confidence`（信頼度 0.0〜1.0）が含まれます。

## パラメータ付きモデル

学習時に追加パラメータ（重量など）を使用したモデルの場合、`params` 引数にパラメータを渡します。

```python
# 重量パラメータを持つモデルの場合
weight = 1.5
results = engine.inference(image, params=[weight])
```

パラメータの数はモデルの `param_tensor` 入力の次元数と一致する必要があります。

```python
# パラメータ数の確認
print(engine.model_inputs)
# [{'name': 'img_tensor', 'shape': [1, 3, 224, 224], 'type': 'tensor(float)'},
#  {'name': 'param_tensor', 'shape': [1, 2], 'type': 'tensor(float)'}]
# → params には2要素のリストを渡す
```

## メタデータの軽量読み取り

モデルのクラス名だけを取得したい場合、フルロードは不要です。

```python
# クラス名のみ取得（高速）
classes = CurecoInference.read_classes("model.onnx")
print(classes)  # ['cat', 'dog', 'bird']

# メタデータ全体を取得
metadata = CurecoInference.read_metadata("model.onnx")
print(metadata.keys())
```

これらの静的メソッドはグラフ最適化を無効化しCPUのみで実行するため、`load_model` よりも大幅に高速です。

## モデル情報の確認

```python
engine = CurecoInference()
engine.load_model("model.onnx")

# モデルタイプ
print(engine.model_type)  # 'classification'

# 認識可能なクラス
print(engine.classes)  # ['cat', 'dog', 'bird']

# 使用中のExecution Provider
print(engine.get_current_providers())  # ['DmlExecutionProvider', 'CPUExecutionProvider']

# 入出力テンソル情報
print(engine.model_inputs)
print(engine.model_outputs)
```
