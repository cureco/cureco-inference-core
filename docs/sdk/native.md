# C / C++

公開ABIはsdk/native/include/cureco/inference.h、
C++用RAIIラッパーはinference.hppです。

展開した配布SDKを参照してビルドします。

```sh
cmake -S examples/c/minimal -B build/c -DCMAKE_PREFIX_PATH=/path/to/native-sdk
cmake --build build/c --config Release
cmake -S examples/cpp/minimal -B build/cpp -DCMAKE_PREFIX_PATH=/path/to/native-sdk
cmake --build build/cpp --config Release
```

最小サンプルはデコーダ依存を避けるため、ヘッダーなしRGB8を受け取ります。

```text
cureco_c_example model.onnx image.rgb width height
cureco_cpp_example model.onnx image.rgb width height
```

image.rgbはwidth × height × 3バイト、行順RGBです。PNG/JPEGをそのまま渡せません。
実際の組み込みでは利用中の画像ライブラリでデコードしたバッファをci_runへ渡します。
WindowsではSDKのbin内のDLLを実行ファイルと同じフォルダに置きます。
Linuxではlibをライブラリ探索パスへ追加するかRPATHを設定します。

ci_create → ci_run → ci_result_destroy → ci_destroyの順で使用します。
入力バッファはci_run完了まで有効に保ちます。
結果JSONとint32 maskはci_result_destroyまで有効で、保持する場合はコピーします。
戻り値でエラーを確認し、同じスレッドでci_last_errorを読みます。文字列はUTF-8です。
C++のEngineとResultは破棄時にリソースを解放します。詳細は公開ヘッダーを参照してください。
