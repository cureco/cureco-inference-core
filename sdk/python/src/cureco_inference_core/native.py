"""C ABI v1 binding. Inference never embeds Python in the native library.

The platform wheel includes the native SDK. library_path or CURECO_NATIVE_LIBRARY
can explicitly override it. Missing libraries fail without a Python fallback.
"""
import ctypes as ct
import json
import os
import sys
from pathlib import Path
import threading

import numpy as np
from PIL import Image
from .types import InferenceResult


class NativeInference:
    def __init__(self, library_path=None):
        path = library_path or os.environ.get('CURECO_NATIVE_LIBRARY')
        if not path:
            name = 'cureco_inference.dll' if sys.platform == 'win32' else 'libcureco_inference.so'
            path = Path(__file__).resolve().parent / '_native' / name
        path = Path(path).resolve()
        if not path.is_file():
            raise RuntimeError(
                f'Native Cureco SDK not found: {path}. Install a platform wheel or '
                'build/package the native SDK. Python inference fallback is not available.')
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self._dll_directory = os.add_dll_directory(str(path.parent)) if os.name == 'nt' else None
        self.library_path = str(path)
        self._lib = ct.CDLL(self.library_path)
        lib = self._lib
        lib.ci_abi_version.restype = ct.c_uint32
        if lib.ci_abi_version() != 1:
            raise RuntimeError('Unsupported Cureco native ABI version')
        lib.ci_last_error.restype = ct.c_char_p
        lib.ci_available_providers.restype = ct.c_char_p
        lib.ci_create.argtypes = [ct.c_char_p, ct.c_char_p, ct.POINTER(ct.c_void_p)]
        lib.ci_create.restype = ct.c_int
        lib.ci_destroy.argtypes = [ct.c_void_p]
        lib.ci_destroy.restype = None
        lib.ci_model_info.argtypes = [ct.c_void_p]
        lib.ci_model_info.restype = ct.c_char_p
        lib.ci_run.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int32, ct.c_int32,
                              ct.c_size_t, ct.c_int32, ct.c_void_p, ct.c_size_t, ct.c_float,
                              ct.POINTER(ct.c_void_p)]
        lib.ci_run.restype = ct.c_int
        lib.ci_result_json.argtypes = [ct.c_void_p]
        lib.ci_result_json.restype = ct.c_char_p
        lib.ci_result_mask.argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t)]
        lib.ci_result_mask.restype = ct.POINTER(ct.c_int32)
        lib.ci_result_destroy.argtypes = [ct.c_void_p]
        lib.ci_result_destroy.restype = None
        try:
            lib.ci_read_metadata.argtypes = [ct.c_char_p, ct.POINTER(ct.c_void_p)]
            lib.ci_read_metadata.restype = ct.c_int
        except AttributeError as ex:
            raise RuntimeError('Native SDK is too old: ci_read_metadata is required') from ex

    def _read_metadata(self, model_path):
        with self._lock:
            result = ct.c_void_p()
            self._check(self._lib.ci_read_metadata(str(model_path).encode('utf-8'), ct.byref(result)))
            try:
                return json.loads(self._lib.ci_result_json(result))
            finally:
                self._lib.ci_result_destroy(result)

    def get_available_providers(self):
        raw = self._lib.ci_available_providers()
        if raw is None:
            raise RuntimeError(self._lib.ci_last_error().decode('utf-8'))
        return json.loads(raw)

    def get_current_providers(self):
        with self._lock:
            return self.model_info['configured_providers'] if self._handle else []

    def _check(self, status):
        if status:
            raise RuntimeError(self._lib.ci_last_error().decode('utf-8'))

    def load_model(self, model_path, providers=None, **options):
        if providers is not None:
            options['providers'] = providers
        with self._lock:
            new = ct.c_void_p()
            self._check(self._lib.ci_create(os.fsencode(model_path) if os.name != 'nt' else str(model_path).encode('utf-8'),
                                          json.dumps(options).encode('utf-8'), ct.byref(new)))
            self._lib.ci_destroy(self._handle)
            self._handle = new

    @property
    def model_info(self):
        with self._lock:
            if not self._handle:
                raise RuntimeError('Model is not loaded')
            return json.loads(self._lib.ci_model_info(self._handle))

    @property
    def model_type(self):
        return self.model_info['task']

    def inference(self, image, params=(), *, confidence_threshold=0.25, color='RGB') -> InferenceResult:
        with self._lock:
            return self._inference(image, params, confidence_threshold=confidence_threshold, color=color)

    def _inference(self, image, params=(), *, confidence_threshold=0.25, color='RGB'):
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert('RGB'))
            color = 'RGB'
        else:
            array = np.asarray(image)
        if array.dtype != np.uint8:
            raise ValueError('Image must contain uint8 pixels')
        if color not in ('RGB', 'BGR', 'GRAY'):
            raise ValueError('color must be RGB, BGR or GRAY')
        if not ((array.ndim == 3 and array.shape[2] == 3 and color != 'GRAY') or
                (array.ndim == 2 and color == 'GRAY')):
            raise ValueError('Image shape does not match color format')
        array = np.ascontiguousarray(array)
        info = self.model_info
        if isinstance(params, dict):
            names = info['parameter_names']
            if set(params) != set(names):
                raise ValueError('Parameter names mismatch')
            params = [params[name] for name in names]
        values = np.ascontiguousarray(params, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError('Parameters must be a vector')
        with self._lock:
            if not self._handle:
                raise RuntimeError('Model is not loaded')
            result = ct.c_void_p()
            self._check(self._lib.ci_run(self._handle, array.ctypes.data, array.nbytes,
                                       array.shape[1], array.shape[0], array.strides[0],
                                       {'RGB': 0, 'BGR': 1, 'GRAY': 2}[color],
                                       values.ctypes.data, values.size, confidence_threshold, ct.byref(result)))
            try:
                data = json.loads(self._lib.ci_result_json(result))
                if data['task'] == 'classification':
                    # Python exposes task-specific owned result objects.
                    return [{'label': item['label'], 'confidence': item['confidence']} for item in data['results']]
                if data['task'] == 'detection':
                    return data['detections']
                count = ct.c_size_t()
                ptr = self._lib.ci_result_mask(result, ct.byref(count))
                data['mask'] = np.ctypeslib.as_array(ptr, shape=(count.value,)).copy().reshape(data['height'], data['width'])
                return data
            finally:
                self._lib.ci_result_destroy(result)

    def close(self):
        with self._lock:
            self._lib.ci_destroy(self._handle)
            self._handle = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if getattr(self, '_handle', None):
            self._lib.ci_destroy(self._handle)
        directory = getattr(self, '_dll_directory', None)
        if directory is not None:
            directory.close()
