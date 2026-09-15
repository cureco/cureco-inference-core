# Native inference SDK

The implementation is C++17 with a stable C ABI and a header-only C++ RAII wrapper.
Both entry points use the same ONNX Runtime session, validation, preprocessing and postprocessing.

The binary SDK contains include/, lib/, bin/ (Windows), licenses/ and manifest.json.
Consume it with find_package(Cureco CONFIG REQUIRED) and link Cureco::cureco_inference.
The examples under examples/c/minimal and examples/cpp/minimal build against the installed SDK,
not the source tree. Keep the accompanying ONNX Runtime shared libraries with your program.

See docs/building/README.md and docs/sdk/model-contract.md in the source repository.
