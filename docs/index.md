# Cureco Inference

Curecoの学習プラットフォームでエクスポートしたONNXモデルを使用して、推論を実行するためのPythonパッケージです。

## 特徴

- **簡単なAPI** — モデルのロードから推論まで数行で実行可能
- **自動前処理** — モデルのメタデータから前処理パラメータを自動読み取り
- **ハードウェア自動検出** — NPU / GPU / CPU を自動選択し、最適なExecution Providerで推論
- **軽量メタデータ読み取り** — モデルをフルロードせずにクラス名等を取得可能

## 対応モデルタイプ

| タイプ | 状態 |
|--------|------|
| 画像分類 (`classification`) | 対応済み |
| 物体検出 (`detection`) | 開発予定 |

## 対応Execution Provider

| プロバイダー | ハードウェア | 優先順位 |
|-------------|-------------|---------|
| QNNExecutionProvider | Qualcomm NPU | 1 |
| VitisAIExecutionProvider | AMD Ryzen AI | 2 |
| OpenVINOExecutionProvider | Intel NPU | 3 |
| CUDAExecutionProvider | NVIDIA GPU | 4 |
| DmlExecutionProvider | DirectML (Windows GPU) | 5 |
| RKNPUExecutionProvider | Rockchip NPU | 6 |
| CPUExecutionProvider | CPU（フォールバック） | 7 |

## インストール

```bash
pip install cureco-inference-core
```

## クイックスタート

```python
from PIL import Image
from cureco_inference_core import CurecoInference

# モデルをロード
engine = CurecoInference()
engine.load_model("model.onnx")

# 推論を実行
image = Image.open("test.jpg").convert("RGB")
results = engine.inference(image, params=[])

for result in results:
    print(f"{result['label']}: {result['confidence']:.2%}")
```

詳しくは[チュートリアル](tutorial/quickstart.md)を参照してください。
