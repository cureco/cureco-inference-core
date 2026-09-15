"""Keep native libraries next to the ctypes wrapper in a frozen application."""
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

binaries = collect_dynamic_libs('cureco_inference_core', search_patterns=['*.dll', '*.so*'])
datas = collect_data_files('cureco_inference_core', includes=['_native/licenses/*', '_native/manifest.json'])
