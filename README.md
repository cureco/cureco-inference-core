# Cureco Inference SDK

ONNXモデルを組み込むための推論SDKと、実行形式でも使えるC#サンプルアプリです。

| 目的 | 入口 |
| --- | --- |
| アプリで推論する | [簡易アプリ](docs/inference-app/README.md) |
| Pythonへ組み込む | [Python SDK](docs/sdk/python.md) |
| C / C++へ組み込む | [Native SDK](docs/sdk/native.md) |
| C#へ組み込む | [C# SDK](docs/sdk/csharp.md) |
| ソースからビルドする | [ビルドと配布](docs/building/README.md) |

配布物は[GitHub Releases](https://github.com/cureco/cureco-inference-core/releases)から取得してください。
アプリZIPは署名なしです。Windowsの発行元確認・警告については[利用手順](docs/inference-app/README.md)を確認してください。
wheel・NuGetもRelease添付ファイルから取得します。PyPI/NuGet.orgで同じバージョンが公開されているとは限りません。
初期検証対象はWindows x64 / CPUです。他のOS・アクセラレータは対象環境で検証後に対応を明示します。

## 構成

```text
sdk/
  native/                  C++共通実装、C ABI、C++ラッパー
  python/                  Pythonバインディング・wheelビルド
  csharp/                  C#バインディング・NuGetビルド
examples/
  python/minimal/          GUIなしの最小サンプル
  c/minimal/
  cpp/minimal/
  csharp/minimal/
  csharp/inference-app/    C#アプリ、Visual Studio solution、テスト、installer
tests/integration/         合成モデルによる横断テスト
docs/                     利用・組み込み・ビルド手順
.github/                  公開ソース検査とビルドCI
```

推論処理はsdk/nativeに一本化しています。PythonとC#は同じネイティブSDKを呼びます。
簡易アプリもSDK利用例であり、配布NuGetを参照します。
分類、物体検出、セマンティックセグメンテーションに対応します。
任意のONNXが使えるわけではありません。[モデル契約](docs/sdk/model-contract.md)を確認してください。

ソースは[MIT](LICENSE)。依存ライブラリのライセンスも配布物に同梱します。
信頼できるモデルのみを使用し、[セキュリティ方針](SECURITY.md)を確認してください。
