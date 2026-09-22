# Python

Python 3.10以降・Windows x64。PyPIからインストールできます。
ネイティブランタイムはパッケージに含まれます。

```sh
python -m pip install --upgrade cureco-inference-core
python examples/python/minimal/infer.py model.onnx image.png
```

```python
from PIL import Image
from cureco_inference_core import CurecoInference

with CurecoInference() as engine:
    engine.load_model("model.onnx")
    with Image.open("image.png") as image:
        result = engine.inference(image.convert("RGB"))
    print(result)
```

分類はラベル・信頼度のリスト、検出は検出結果リスト、
セグメンテーションはmask（numpy.int32）などを含む辞書です。
maskは入力画像解像度です。保存時にクラスIDを8bitへ無条件に縮めないでください。

補助入力にはinference(image, params=[...])を使用します。名前が定義されたモデルは辞書でも指定できます。
NumPy画像はuint8、BGR入力はcolor="BGR"を指定します。
コンテキストマネージャーまたはclose()で解放します。同一エンジンの処理は直列化されます。
CURECO_NATIVE_LIBRARYまたはlibrary_pathはカスタムビルド用の明示的な上書きであり、通常不要です。
ライブラリ不足時に別の推論実装へフォールバックすることはありません。
