"""Real ORT execution fixtures, shared by Python, C/C++, and .NET smoke tests."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from PIL import Image
onnx = pytest.importorskip('onnx')
from onnx import helper as H, TensorProto as T, numpy_helper
from cureco_inference_core import CurecoInference, NativeInference
from cureco_inference_core.contract import validate_contract


def write_model(path, task='classification', batched=False, empty=False, parameters=False, mutate=None):
    inputs = [H.make_tensor_value_info('img_tensor', T.FLOAT, ['batch', 3, 8, 12])]
    if task == 'classification':
        outputs = [H.make_tensor_value_info('output', T.FLOAT, ['batch', 3])]
        nodes = [H.make_node('ReduceMean', ['img_tensor'], ['mean' if parameters else 'output'], axes=[2, 3], keepdims=0)]
        if parameters:
            inputs.append(H.make_tensor_value_info('param_tensor', T.FLOAT, ['batch', 3]))
            nodes.append(H.make_node('Add', ['mean', 'param_tensor'], ['output']))
    elif task == 'segmentation':
        outputs = [H.make_tensor_value_info('output', T.FLOAT, ['batch', 3, 8, 12])]
        nodes = [H.make_node('Identity', ['img_tensor'], ['output'])]
    else:
        inputs[0] = H.make_tensor_value_info('img_tensor', T.FLOAT, [1, 3, 8, 12])
        values = [np.array([[1, 2, 11, 7], [0, 0, 3, 4]], np.float32), np.array([1, 2], np.int64), np.array([0.9, 0.1], np.float32)]
        if empty:
            values = [v[:0] for v in values]
        if batched:
            values = [v[None] for v in values]
        names = ['boxes', 'labels', 'scores']
        outputs = [H.make_tensor_value_info(n, T.INT64 if i == 1 else T.FLOAT, list(v.shape)) for i, (n, v) in enumerate(zip(names, values))]
        nodes = [H.make_node('Constant', [], [n], value=numpy_helper.from_array(v)) for n, v in zip(names, values)]
    model = H.make_model(H.make_graph(nodes, 'sdk-fixture', inputs, outputs), opset_imports=[H.make_opsetid('', 17)], ir_version=8)
    def sig(items):
        return [dict(name=v.name, dtype=T.DataType.Name(v.type.tensor_type.elem_type).lower(),
                     shape=[d.dim_value if d.HasField('dim_value') else d.dim_param or None for d in v.type.tensor_type.shape.dim]) for v in items]
    metadata = {'dataset_type': task, 'preprocessing': {}, 'idx_to_class': {'0': 'red', '1': 'green', '2': 'blue'},
                'parameter_names': ['weight', 'height', 'length'] if parameters else [], 'num_params': 3 if parameters else 0}
    c = dict(schema_version=1, task=task, image_layout='NCHW', inputs=sig(inputs), outputs=sig(outputs),
             preprocessing=dict(location='outside_graph', color='RGB', scale=1/255, normalization=None,
                                resize=dict(mode='stretch', size=[8, 12], interpolation='bicubic', antialias=True)),
             output_semantics={'output': 'logits', 'class_axis': 1}, mask_policy=None, nms=None)
    if task == 'detection':
        c['output_semantics'] = dict(boxes='xyxy', coordinate_space='input_image_pixels', labels='zero_based_class_id', scores='confidence')
        c['nms'] = 'not_applied' if batched else 'inside_graph'
    if mutate:
        mutate(c, metadata)
    metadata['cureco_contract'] = c
    for k, v in metadata.items():
        m = model.metadata_props.add(); m.key = k; m.value = json.dumps(v)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


@pytest.fixture
def engine():
    with CurecoInference() as instance:
        yield instance


def test_classification_parameters(engine, tmp_path):
    assert engine.get_current_providers() == []
    assert 'CPUExecutionProvider' in engine.get_available_providers()
    engine.load_model(str(write_model(tmp_path/'params.onnx', parameters=True)), ['CPUExecutionProvider'])
    assert engine.model_inputs[0]['type'] == 'tensor(float)'
    result = engine.inference(Image.new('RGB', (24, 16), (255, 0, 0)), [0, 3, 0])
    assert result[0]['label'] == 'green'
    assert result[0]['confidence'] == pytest.approx(np.exp(3)/(np.exp(1)+np.exp(3)+1), abs=1e-6)
    with pytest.raises((RuntimeError, ValueError)):
        engine.inference(Image.new('RGB', (12, 8)), [1])


@pytest.mark.parametrize('batched,empty', [(False, False), (True, False), (False, True), (True, True)])
def test_detection(engine, tmp_path, batched, empty):
    engine.load_model(str(write_model(tmp_path/'det.onnx', 'detection', batched, empty)), ['CPUExecutionProvider'])
    result = engine.inference(Image.new('RGB', (24, 16)), [])
    if empty:
        assert result == []
    else:
        assert len(result) == 1
        assert result[0]['class_id'] == 1
        assert result[0]['box'] == [2, 4, 22, 14]
        assert result[0]['confidence'] == pytest.approx(0.9)


def test_segmentation(engine, tmp_path):
    engine.load_model(str(write_model(tmp_path/'seg.onnx', 'segmentation')), ['CPUExecutionProvider'])
    result = engine.inference(Image.new('RGB', (25, 17), (0, 255, 0)), [])
    assert result['mask'].shape == (17, 25)
    assert result['mask'].dtype == np.int32
    assert (result['mask'] == 1).all()


@pytest.mark.parametrize('case', ['layout', 'names', 'normalization', 'nms', 'mask'])
def test_invalid_models_rejected(engine, tmp_path, case):
    task = 'detection' if case == 'nms' else 'segmentation' if case == 'mask' else 'classification'
    def mutate(c, m):
        if case == 'layout': c['image_layout'] = 'NHWC'
        elif case == 'names': m['parameter_names'] = ['weight']; m['num_params'] = 2
        elif case == 'normalization': c['preprocessing']['normalization'] = {'mean': [0]*3, 'std': [1]*3}
        elif case == 'nms': c['nms'] = 'unknown'
        else: c['mask_policy'] = {'background_mode': 'auto', 'ignore_index': 255}
    with pytest.raises((RuntimeError, ValueError)):
        engine.load_model(str(write_model(tmp_path/'invalid.onnx', task, mutate=mutate)), ['CPUExecutionProvider'])


@pytest.mark.parametrize('size', [(12, 8), (3, 5), (65, 49), (31, 7)])
def test_native_reference_parity(tmp_path, size):
    path = write_model(tmp_path/'rgb.onnx')
    native = NativeInference(); native.load_model(path)
    pixels = np.random.default_rng(42).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    # Independent numerical oracle in tests only, not a shipped inference backend.
    resized = np.asarray(Image.fromarray(pixels).resize((12, 8), Image.Resampling.BICUBIC), dtype=np.float32)/255
    logits = resized.mean(axis=(0, 1))
    scores = np.exp(logits - logits.max()); scores /= scores.sum()
    a = [dict(label=['red','green','blue'][scores.argmax()], confidence=float(scores.max()))]
    b = native.inference(pixels)
    assert a[0]['label'] == b[0]['label']
    assert a[0]['confidence'] == pytest.approx(b[0]['confidence'], abs=1e-6)
    # BGR and gray input conversion also share the native implementation.
    assert native.inference(pixels[..., ::-1], color='BGR') == b
    gray = np.full((5, 3), 128, np.uint8)
    assert native.inference(gray, color='GRAY')[0]['confidence'] == pytest.approx(1/3)
    native.close()
    with pytest.raises(RuntimeError): native.inference(pixels)


def test_native_mask_parity(tmp_path):
    path = write_model(tmp_path/'mask.onnx', 'segmentation')
    with NativeInference() as native:
        native.load_model(path)
        image = Image.fromarray(np.random.default_rng(42).integers(0,256,(31,49,3),dtype=np.uint8))
        expected = np.asarray(image.resize((12,8), Image.Resampling.BICUBIC)).argmax(axis=2)
        ys = ((2*np.arange(31)+1)*8)//(2*31)
        xs = ((2*np.arange(49)+1)*12)//(2*49)
        np.testing.assert_array_equal(expected[ys[:,None],xs[None,:]], native.inference(image)['mask'])


@pytest.mark.parametrize('task', ['classification', 'detection', 'segmentation'])
def test_c_cpp_dotnet(tmp_path, task):
    folder = os.environ.get('CURECO_EXAMPLES_DIR')
    if not folder: pytest.skip('Compiled examples not configured')
    model = write_model(tmp_path/f'{task}.onnx', task)
    suffix = '.exe' if os.name == 'nt' else ''
    image = Image.new('RGB', (2, 2), (255, 0, 0))
    raw = tmp_path/'image.rgb'
    raw.write_bytes(image.tobytes())
    arguments = [str(model), str(raw), '2', '2']
    commands = [[str(Path(folder)/(name+suffix)), *arguments] for name in ('cureco_c_example','cureco_cpp_example')]
    dotnet = os.environ.get('CURECO_DOTNET_SMOKE')
    if dotnet: commands.append(['dotnet', dotnet, *arguments])
    results = [json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout) for cmd in commands]
    assert all(r == results[0] for r in results)
    assert results[0]['task'] == task

def test_native_errors_and_failed_reload(tmp_path):
    import ctypes as ct
    path = write_model(tmp_path/'valid.onnx')
    with NativeInference() as native:
        native.load_model(path)
        with pytest.raises(RuntimeError, match='Unavailable provider'):
            native.load_model(path, providers=['NonexistentExecutionProvider'])
        assert native.inference(Image.new('RGB', (2,2), (255,0,0)))[0]['label'] == 'red'
        result = ct.c_void_p(123)
        pixels = (ct.c_uint8*1)(0)
        code = native._lib.ci_run(native._handle, pixels, 1, 2, 2, 6, 0, None, 0, 0.25, ct.byref(result))
        assert code != 0 and not result.value
        assert b'buffer too small' in native._lib.ci_last_error()
        for threshold in [-1, float('nan'), 2]:
            with pytest.raises(RuntimeError):
                native.inference(Image.new('RGB', (2,2)), confidence_threshold=threshold)


def test_named_native_parameters(tmp_path):
    with NativeInference() as native:
        native.load_model(write_model(tmp_path/'named.onnx', parameters=True))
        image = Image.new('RGB', (2, 2), (255, 0, 0))
        expected = native.inference(image, [0, 3, 0])
        assert native.inference(image, {'length': 0, 'weight': 0, 'height': 3}) == expected
        with pytest.raises(ValueError, match='names'):
            native.inference(image, {'weight': 0})


if __name__ == '__main__':
    import sys
    directory = Path(sys.argv[1]); directory.mkdir(parents=True, exist_ok=True)
    for task in ['classification', 'detection', 'segmentation']:
        write_model(directory/f'{task}.onnx', task)
    write_model(directory/'parameters.onnx', parameters=True)
    image = Image.new('RGB', (2, 2), (255, 0, 0))
    image.save(directory/'image.png')
    (directory/'image.rgb').write_bytes(image.tobytes())
