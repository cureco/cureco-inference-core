# 利用開始

## アプリ

検証済みのWindows x64アプリZIPを展開してCurecoInference.exeを実行します。
DLLを含むフォルダ全体を保持してください。PythonやVisual Studioは不要です。
モデルと画像を選び「推論を実行」を押します。[詳細](../inference-app/README.md)

## SDK

- Python: 対象OSのwheelをインストール。[組み込み例](../sdk/python.md)
- C / C++: 対象OSのNative SDK ZIPを展開。[CMakeサンプル](../sdk/native.md)
- C#: 対象RIDのランタイムを含むNuGetを参照。[組み込み例](../sdk/csharp.md)

モデル・画像は利用者が用意します。テストは合成モデルだけで再現できます。
[モデル契約](../sdk/model-contract.md)も確認してください。
