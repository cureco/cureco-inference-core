"""Native-only facade and bundled runtime checks; no Python ORT session fallback."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image
from cureco_inference_core import CurecoInference
from test_sdk import write_model


def test_bundled_runtime_without_python_ort(tmp_path, monkeypatch):
    monkeypatch.delenv('CURECO_NATIVE_LIBRARY', raising=False)
    path = write_model(tmp_path/'分類.onnx')
    script = """
import sys
from pathlib import Path
class BlockOrt:
    def find_spec(self, fullname, *args):
        if fullname == 'onnxruntime' or fullname.startswith('onnxruntime.'):
            raise AssertionError('Python ONNX Runtime must not be imported')
sys.meta_path.insert(0, BlockOrt())
from cureco_inference_core import CurecoInference
from PIL import Image
with CurecoInference() as engine:
    assert Path(engine.library_path).parent.name == '_native'
    assert '分類' not in engine.library_path
    engine.load_model(sys.argv[1])
    assert engine.inference(Image.new('RGB',(17,31),(255,0,0)))[0]['label'] == 'red'
    assert set(CurecoInference.read_classes(sys.argv[1])) == {'red','green','blue'}
    assert 'onnxruntime' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', script, str(path)], check=True, env=os.environ.copy())


def test_metadata_is_raw_and_does_not_require_adapter(tmp_path):
    path = write_model(tmp_path/'metadata.onnx', mutate=lambda c,m: c.update(schema_version=99))
    metadata = CurecoInference.read_metadata(path)
    assert metadata['dataset_type'] == '"classification"'
    assert json.loads(metadata['cureco_contract'])['schema_version'] == 99
    assert set(CurecoInference.read_classes(path)) == {'red','green','blue'}
    with CurecoInference() as engine:
        with pytest.raises(RuntimeError):
            engine.load_model(path)


def test_missing_native_fails_without_fallback(tmp_path):
    with pytest.raises(RuntimeError, match='fallback is not available'):
        CurecoInference(library_path=tmp_path/'missing.dll')


def test_public_state_survives_failed_reload(tmp_path):
    with CurecoInference() as engine:
        engine.load_model(write_model(tmp_path/'valid.onnx'))
        classes, inputs = engine.classes, engine.model_inputs
        with pytest.raises(RuntimeError):
            engine.load_model(tmp_path/'missing.onnx')
        assert engine.classes == classes and engine.model_inputs == inputs
        engine.close()
        assert engine.classes == [] and engine.model_inputs == []
