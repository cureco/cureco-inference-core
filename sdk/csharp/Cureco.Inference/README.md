# Cureco.Inference.Core

C# bindings to the shared native inference SDK. This package is not a second inference engine.
Use a package containing the native runtime for your target RID (initial validation: win-x64).

```csharp
using Cureco.Inference;
using var engine = new Engine("model.onnx");
var result = engine.Infer(rgbBytes, width, height, width * 3);
Console.WriteLine(result.Data);
```

Inputs are packed RGB/BGR/gray byte arrays; results own their JSON and mask data.
Dispose the engine when finished. An engine serializes its inference calls.
Supported model contracts and build instructions are in the repository's public docs.
