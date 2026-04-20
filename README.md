# cureco-inference-core

Cureco学習プラットフォームでエクスポートしたONNXモデルを使用して、画像分類などの推論を実行するためのPythonパッケージです。

## 特徴

- **ハードウェア自動検出**: NPU/GPU/CPUを自動検出し、最適なExecution Providerを選択
- **前処理・後処理の自動化**: モデルメタデータから正規化パラメータやクラスマッピングを自動読み取り
- **拡張可能な設計**: 抽象基底クラスを継承して、独自の推論タイプを追加可能

## インストール

```bash
pip install cureco-inference-core
```

GPU (CUDA) を使用する場合:

```bash
pip install cureco-inference-core[gpu]
```

DirectML (Windows GPU) を使用する場合:

```bash
pip install cureco-inference-core[directml]
```

## クイックスタート

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

## メタデータの軽量読み取り

推論セッションを起動せずに、モデルのクラス名だけを取得できます。

```python
classes = CurecoInference.read_classes("model.onnx")
print(classes)  # ['cat', 'dog', 'bird']
```

## 対応 Execution Provider

優先順位順:

| Provider | ハードウェア |
|---|---|
| QNNExecutionProvider | Qualcomm NPU |
| VitisAIExecutionProvider | AMD Ryzen AI |
| OpenVINOExecutionProvider | Intel NPU |
| CUDAExecutionProvider | NVIDIA GPU |
| DmlExecutionProvider | DirectML (Windows GPU) |
| RKNPUExecutionProvider | Rockchip NPU |
| CPUExecutionProvider | CPU (フォールバック) |

## ライセンス

MIT
