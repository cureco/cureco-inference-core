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


## 異常検知モジュール（PatchCore-lite・良品学習のみ）

`pip install cureco-inference-core[anomaly]`（torch / torchvision）で、
**アノテーション不要・良品画像のみ・エッジ内で数分**の異常検知が使えます
（cureco-edge の検出コアの実体。実測は同リポジトリ docs/REPORT.md）。

```python
from cureco_inference_core.anomaly import PatchCoreLite

judge = PatchCoreLite(weights_path="wrn50_backbone.pt")  # オフライン環境は重みを同梱
judge.fit(normal_crops)          # (N,96,96) グレー or (N,96,96,3) BGR カラー
judge.calibrate_dense(images)    # 全面スキャン（画像単位OK/NG）を使う場合
judge.save("bank.pt")            # バンク＝モデルファイル

scores, maps = judge.score_crops(crops)              # パッチ判定
tiles, origins = judge.tile_image(image)             # 全面スキャン用タイル分割（dense_stride で重なり可）
judge.absorb(fp_crops, tag=7)                        # 誤検出を良品として即時追記（再学習不要）
judge.register_defect(fn_crops, tag=7)               # 見逃しをNG見本として登録（少数ショット）
judge.forget(7)                                      # 訂正のアンドゥ（由来タグで行単位に取消）
```

教師あり分類（ONNX）とはモデル形式が異なるため独立モジュールです。
クラウド学習を経由せず**完全オフラインで学習が完結**する点が製品ライン上の役割
（Cureco AI Edge の「欠陥を見つける」プロファイル）。
