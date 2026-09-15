"""Validation helpers for the public Cureco ONNX contract v1."""
import json
import math


def validate_session_contract(session, expected_task=None):
    metadata = session.get_modelmeta().custom_metadata_map
    if 'cureco_contract' not in metadata:
        return None
    c = json.loads(metadata['cureco_contract'])
    if not isinstance(c, dict) or type(c.get('schema_version')) is not int or c['schema_version'] != 1:
        raise ValueError('Unsupported Cureco ONNX contract version')
    task = c.get('task')
    if task not in ('classification', 'detection', 'segmentation'):
        raise ValueError('Unsupported Cureco ONNX task')
    if expected_task is not None and expected_task != task:
        raise ValueError('ONNX task mismatch')
    if 'dataset_type' in metadata and json.loads(metadata['dataset_type']) != task:
        raise ValueError('ONNX task metadata mismatch')
    if c.get('image_layout') != 'NCHW':
        raise ValueError('Unsupported ONNX image layout')
    for key, tensors in (('inputs', session.get_inputs()), ('outputs', session.get_outputs())):
        declared = c.get(key)
        if not isinstance(declared, list) or len(declared) != len(tensors):
            raise ValueError(f'ONNX {key} count mismatch')
        for item, tensor in zip(declared, tensors):
            if (not isinstance(item, dict) or item.get('name') != tensor.name
                    or item.get('shape') != tensor.shape or f"tensor({item.get('dtype')})" != tensor.type):
                raise ValueError(f'ONNX {key} signature mismatch: {tensor.name}')
    preprocessing = c.get('preprocessing')
    if preprocessing is not None:
        resize = preprocessing.get('resize') if isinstance(preprocessing, dict) else None
        if (not isinstance(resize, dict) or preprocessing.get('location') != 'outside_graph'
                or preprocessing.get('color') != 'RGB' or preprocessing.get('scale') != 1 / 255
                or resize.get('mode') != 'stretch' or resize.get('interpolation') != 'bicubic'
                or resize.get('antialias') is not True):
            raise ValueError('Unsupported ONNX preprocessing')
        image = next((i for i in session.get_inputs() if i.name == 'img_tensor'), None)
        if image is None or image.shape[2:] != resize.get('size'):
            raise ValueError('ONNX resize size mismatch')
        legacy = json.loads(metadata.get('preprocessing', '{}'))
        if not isinstance(legacy, dict):
            raise ValueError('Invalid legacy preprocessing')
        mean, std = legacy.get('normalize_mean'), legacy.get('normalize_std')
        norm = {'mean': mean, 'std': std} if mean is not None and std is not None else None
        if preprocessing.get('normalization') != norm:
            raise ValueError('ONNX normalization metadata mismatch')
        if norm is not None:
            if (not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 3 or len(std) != 3
                    or any(type(v) not in (float, int) or not math.isfinite(v) for v in mean + std)
                    or any(v <= 0 for v in std)):
                raise ValueError('Invalid ONNX normalization values')
    policy = c.get('mask_policy')
    if policy is not None:
        if (not isinstance(policy, dict) or policy.get('background_mode') not in ('implicit', 'explicit')
                or type(policy.get('ignore_index')) is not int or policy['ignore_index'] != 255):
            raise ValueError('Unsupported ONNX mask policy')
    if 'parameter_names' in metadata:
        names = json.loads(metadata['parameter_names'])
        if (not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names)
                or len(set(names)) != len(names)):
            raise ValueError('Invalid parameter_names')
        if 'num_params' in metadata:
            count = json.loads(metadata['num_params'])
            if type(count) is not int or count != len(names):
                raise ValueError('num_params mismatch')
        tensor = next((i for i in session.get_inputs() if i.name == 'param_tensor'), None)
        if tensor is not None and (len(tensor.shape) != 2 or tensor.shape[1] != len(names)):
            raise ValueError('Parameter names/dimension mismatch')
    return c
