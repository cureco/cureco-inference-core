"""Run with an installed platform wheel: python infer.py model.onnx image.png."""
import argparse
import json
from PIL import Image
from cureco_inference_core import CurecoInference

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model")
parser.add_argument("image")
args = parser.parse_args()
with CurecoInference() as engine, Image.open(args.image) as image:
    engine.load_model(args.model)
    result = engine.inference(image.convert("RGB"))
    print(json.dumps(result, default=lambda value: value.tolist(), ensure_ascii=False))
