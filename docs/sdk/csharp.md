# C#

.NET 8以降・Windows x64。Cureco.Inference.Coreは共通ネイティブSDKを呼ぶバインディングです。

```xml
<PackageReference Include="Cureco.Inference.Core" Version="0.2.0" />
```

NuGet.orgからインストールできます。プロジェクトのフォルダーで実行します。

```sh
dotnet add package Cureco.Inference.Core
```

Visual Studioでは「NuGetパッケージの管理」で `Cureco.Inference.Core` を検索します。
ネイティブランタイムはパッケージに含まれます。最小サンプルの実行例:

```sh
dotnet restore examples/csharp/minimal/Minimal.csproj -r win-x64
dotnet run --project examples/csharp/minimal/Minimal.csproj -c Release -r win-x64 -- model.onnx image.rgb 640 480
```

```csharp
using Cureco.Inference;
using var engine = new Engine("model.onnx");
InferenceResult result = engine.Infer(rgbBytes, width, height, width * 3);
Console.WriteLine(result.Data.GetRawText());
int[]? mask = result.Mask;
```

PixelFormatでRGB/BGR/Grayを指定できます。ストライドはバイト単位です。
Dataは所有されたJSON、Maskは入力画像解像度のint32配列です。エンジン破棄後も結果は有効です。
補助入力はparametersで渡します。同一エンジンの推論・破棄は直列化されます。
[簡易アプリ](../inference-app/README.md)が画像デコード・可視化・APIの応用例です。
SDK自体はGUIやHTTPに依存しません。
