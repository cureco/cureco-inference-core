"""Python facade for the shared C++ inference core.

All model validation, preprocessing, inference and postprocessing run in C++.
There is no Python inference backend or fallback.
"""
import json
from .native import NativeInference


class CurecoInference(NativeInference):
    """Use the bundled SDK, or override it with library_path for a custom build."""

    def __init__(self, *, library_path=None):
        super().__init__(library_path=library_path)

    @property
    def idx_to_class(self):
        with self._lock:
            return self.model_info['classes'] if self._handle else {}

    @property
    def classes(self):
        return list(self.idx_to_class.values())

    @property
    def model_type(self):
        with self._lock:
            return self.model_info['task'] if self._handle else ''

    @property
    def model_contract(self):
        with self._lock:
            return self.model_info['contract'] if self._handle else None

    def _signature(self, key):
        with self._lock:
            if not self._handle:
                return []
            return [dict(name=i['name'], shape=i['shape'], type=f"tensor({i['dtype']})")
                    for i in self.model_info[key]]

    @property
    def model_inputs(self):
        return self._signature('inputs')

    @property
    def model_outputs(self):
        return self._signature('outputs')

    @staticmethod
    def read_metadata(model_path, *, library_path=None):
        """Read raw ONNX metadata in C++; no inference adapter is required."""
        with NativeInference(library_path=library_path) as engine:
            return engine._read_metadata(model_path)

    @staticmethod
    def read_classes(model_path, *, library_path=None):
        metadata = CurecoInference.read_metadata(model_path, library_path=library_path)
        return list(json.loads(metadata.get('idx_to_class', '{}')).values())
