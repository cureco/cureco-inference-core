# Cureco inference Python SDK

Thin Python bindings to the shared native inference SDK.
Platform wheels include the native runtime and its license notices; Python ONNX Runtime is not required.

Install on Windows x64 with Python 3.10 or later:

```sh
python -m pip install --upgrade cureco-inference-core
```

```python
from PIL import Image
from cureco_inference_core import CurecoInference

with CurecoInference() as engine:
    engine.load_model("model.onnx")
    result = engine.inference(Image.open("image.png").convert("RGB"))
    print(result)
```

Classification returns label/confidence results, detection returns detections,
and semantic segmentation returns a dictionary including an int32 mask.
Use only trusted model files. See the public repository docs for the model contract.
