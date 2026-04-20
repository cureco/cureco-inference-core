# Execution Provider

ONNX Runtimeは複数のハードウェアバックエンドに対応しています。
Cureco Inferenceは利用可能なプロバイダーを自動検出し、最適なものを選択します。

## 自動選択の仕組み

`load_model` をプロバイダー指定なしで呼び出すと、以下の優先順位で選択されます:

| 優先順位 | プロバイダー | ハードウェア |
|---------|-------------|-------------|
| 1 | QNNExecutionProvider | Qualcomm NPU |
| 2 | VitisAIExecutionProvider | AMD Ryzen AI NPU |
| 3 | OpenVINOExecutionProvider | Intel NPU |
| 4 | CUDAExecutionProvider | NVIDIA GPU |
| 5 | DmlExecutionProvider | DirectML (Windows GPU全般) |
| 6 | RKNPUExecutionProvider | Rockchip NPU |
| 7 | CPUExecutionProvider | CPU（常にフォールバックとして追加） |

## プロバイダーの確認

```python
engine = CurecoInference()

# インストール済みのプロバイダー一覧
print(engine.get_available_providers())
# ['DmlExecutionProvider', 'CPUExecutionProvider']

# モデルロード後、実際に使用中のプロバイダー
engine.load_model("model.onnx")
print(engine.get_current_providers())
# ['DmlExecutionProvider', 'CPUExecutionProvider']
```

## プロバイダーの手動指定

特定のプロバイダーを使用したい場合は、`providers` 引数で指定します。

```python
# CPUのみで実行
engine.load_model("model.onnx", providers=["CPUExecutionProvider"])

# CUDA GPUを明示的に指定
engine.load_model("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

!!! warning "注意"
    指定したプロバイダーがインストールされていない場合、ONNX Runtimeはエラーを返します。
    `get_available_providers()` で事前に確認することを推奨します。

## プロバイダー別のセットアップ

### NVIDIA GPU (CUDA)

```bash
pip install onnxruntime-gpu
```

### Intel NPU (OpenVINO)

```bash
pip install onnxruntime-openvino
```

### Windows GPU (DirectML)

```bash
pip install onnxruntime-directml
```

### CPU（デフォルト）

```bash
pip install onnxruntime
```

!!! note
    プロバイダー固有の `onnxruntime` パッケージは互いに競合する場合があります。
    1つの環境には1種類のみインストールしてください。
