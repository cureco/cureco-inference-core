# ビルドと配布

ビルド定義は各成果物の直下に置きます。sdk/native/package.py、
sdk/python/pyproject.toml、sdk/csharpのcsproj、アプリのinstallerがそれぞれ担当します。
重複するトップレベルpackagingディレクトリは設けません。

## Native SDK

CMake 3.24以降、C++17コンパイラ、対象OS/CPUのONNX Runtime SDK、
nlohmann/jsonヘッダーが必要です。初期CPU検証はORT 1.22.1 / Windows x64です。
公式依存物のLICENSEとThirdPartyNoticesを保持します。

```sh
cmake -S sdk/native -B build/native -DONNXRUNTIME_ROOT=/path/to/onnxruntime-sdk -DCURECO_JSON_INCLUDE_DIR=/path/to/json/include
cmake --build build/native --config Release
python sdk/native/package.py --build-dir build/native --ort-root /path/to/onnxruntime-sdk --json-license /path/to/json/LICENSE.MIT --target win-x64 --output sdk/native/runtime
```

既存出力は上書きしません。生成物はソース管理しません。
Linux用プリセットもありますが、対応明示前に対象環境で横断テストが必要です。

## Python

```sh
python -m pip wheel --no-deps ./sdk/python -w dist/wheels
python -m pip install dist/wheels/<platform-wheel>.whl
```

sdk/native/runtimeを使用し、CURECO_NATIVE_SDK_DIRで変更できます。
ABI・CPUアーキテクチャ・ライセンス同梱を検査します。pure Python wheelは生成しません。
ソース配布からビルドする場合も準備済みNative SDKが必要です。

## C#とアプリ

.NET SDK 8以降、Windowsビルド環境を使います。

```sh
dotnet pack sdk/csharp/Cureco.Inference/Cureco.Inference.csproj -c Release -o dist/nuget -p:RuntimeIdentifier=win-x64 -p:CurecoNativeRuntimeDir=/path/to/native-sdk/bin
dotnet restore examples/csharp/inference-app/InferenceApp.sln -r win-x64 --source ./dist/nuget --source https://api.nuget.org/v3/index.json
dotnet publish examples/csharp/inference-app/src/InferenceApp.csproj -c Release -r win-x64 --self-contained true -o dist/app
```

Visual StudioではInferenceApp.slnを開き、同じNuGetフィードを指定します。
アプリ・最小例は配布NuGetのみを使います。同一バージョンの再生成検証では、
専用RestorePackagesPathを使いキャッシュ混同を避けてください。

自己完結型publish先の全ファイルとlicensesをZIPにします。
Inno Setupでinstaller/InferenceApp.issへPublishDir（publish先絶対パス）を渡すとインストーラを作れます。
自動起動登録は含めません。

## 検証と公開

```sh
python -m pip install pytest onnx numpy Pillow
python tests/integration/test_sdk.py build/fixtures
python -m pytest tests/integration sdk/python/tests -q
dotnet run --project examples/csharp/inference-app/tests/InferenceApp.Tests.csproj -c Release -r win-x64 -- build/fixtures
python .github/scripts/public_source.py
```

C/C++例を同じフォルダへ集めてCURECO_EXAMPLES_DIRを指定します。
CURECO_DOTNET_SMOKEはビルドしたMinimal.dllです。
指定しない横断テストはskipなのでリリース検証では必ず指定します。

公開前にソース・全Git履歴・依存物・ライセンス・生成アーカイブを監査し、
SDKだけで4言語例とアプリが動くことを確認します。
対象OSのクリーン環境で3タスク、GUI保存、起動設定、APIを検証してください。
EXEとインストーラは組織の署名基盤で署名・検証し、チェックサムと対応環境を添えて公開します。
証明書・秘密鍵・署名/公開トークンはリポジトリに置きません。
CIは署名前の検証成果物のみで、レジストリ/Releaseへの自動公開は行いません。
PyPI/NuGetへ公開するバージョンはリリース承認時に揃えます。

ソースZIPは検査済みファイルだけから作ります。

```sh
python .github/scripts/public_source.py --archive dist/cureco-inference-source.zip
```

作業フォルダ全体のZIPはキャッシュ・データ・設定が混入するため使用しません。
