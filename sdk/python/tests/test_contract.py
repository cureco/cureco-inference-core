import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cureco_inference_core import CurecoInference
from cureco_inference_core.contract import validate_contract, validate_classification_support


def fake_session(task='classification'):
    inputs = [SimpleNamespace(name='img_tensor', type='tensor(float)', shape=['batch', 3, 8, 12])]
    outputs = [SimpleNamespace(name='output', type='tensor(float)', shape=['batch', 2])]
    def signature(items):
        return [dict(name=t.name, shape=t.shape, dtype='float') for t in items]
    data = dict(schema_version=1, task=task, image_layout='NCHW', inputs=signature(inputs),
                outputs=signature(outputs), output_semantics=dict(output='logits', class_axis=1))
    metadata = {'cureco_contract': json.dumps(data), 'dataset_type': json.dumps(task)}
    session = SimpleNamespace(get_inputs=lambda: inputs, get_outputs=lambda: outputs,
                              get_modelmeta=lambda: SimpleNamespace(custom_metadata_map=metadata))
    return session, data, metadata


def test_legacy():
    session, _, metadata = fake_session()
    metadata.pop('cureco_contract')
    assert validate_contract(session) is None


@pytest.mark.parametrize('field,value', [('schema_version', 2), ('schema_version', True),
                                        ('task', 'detection'), ('inputs', []), ('image_layout', 'NHWC')])
def test_invalid_contract(field, value):
    session, data, metadata = fake_session()
    data[field] = value
    metadata['cureco_contract'] = json.dumps(data)
    with pytest.raises(ValueError):
        validate_contract(session)


def test_supported_classifier():
    session, data, _ = fake_session()
    validate_classification_support(session, validate_contract(session))
    data['output_semantics']['output'] = 'probabilities'
    with pytest.raises(ValueError):
        validate_classification_support(session, data)


def test_unloaded_and_backend_removed():
    with CurecoInference() as engine:
        assert engine.model_inputs == []
        assert engine.classes == []
        with pytest.raises(RuntimeError):
            engine.inference(Image.new('RGB', (8, 8)), [])
    with pytest.raises(TypeError):
        CurecoInference(backend='python')
