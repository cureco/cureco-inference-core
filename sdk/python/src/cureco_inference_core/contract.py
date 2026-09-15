"""Contract validation and executable adapter capability checks."""
from .contract_validation import validate_session_contract as validate_contract


def validate_classification_support(session, contract):
    """Reject interfaces this single-image adapter cannot execute safely."""
    if contract['task'] != 'classification':
        return  # The inference dispatcher explicitly rejects unimplemented tasks.
    image = next((item for item in session.get_inputs() if item.name == 'img_tensor'), None)
    if (image is None or image.type != 'tensor(float)' or len(image.shape) != 4
            or image.shape[1] != 3 or any(type(d) is not int or d <= 0 for d in image.shape[2:])
            or (isinstance(image.shape[0], int) and image.shape[0] != 1)):
        raise ValueError('Unsupported classification image input')
    for item in session.get_inputs():
        if item.name == 'img_tensor':
            continue
        if (item.name != 'param_tensor' or item.type != 'tensor(float)' or len(item.shape) != 2
                or type(item.shape[1]) is not int or item.shape[1] < 0
                or (isinstance(item.shape[0], int) and item.shape[0] != 1)):
            raise ValueError('Unsupported classification auxiliary input')
    outputs = session.get_outputs()
    if (len(outputs) != 1 or outputs[0].name != 'output' or outputs[0].type != 'tensor(float)' or len(outputs[0].shape) != 2
            or contract.get('output_semantics') != {'output': 'logits', 'class_axis': 1}):
        raise ValueError('Unsupported classification output')
