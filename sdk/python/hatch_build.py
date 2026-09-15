"""Build platform wheels from a prepared SDK; never a pure Python wheel."""
import json
import os
from pathlib import Path
import struct
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def runtime_files(sdk):
    manifest = json.loads((sdk / 'manifest.json').read_text(encoding='utf-8'))
    target = manifest['target']
    if manifest.get('abi_version') != 1:
        raise RuntimeError('Expected native SDK ABI v1')
    if target == 'win-x64':
        tag, machine, folder, name = 'win_amd64', 0x8664, 'bin', 'cureco_inference.dll'
    elif target.startswith('linux-x64'):
        tag, machine, folder, name = 'linux_x86_64', 62, 'lib', 'libcureco_inference.so'
    elif target.startswith(('linux-arm64', 'jetson-orin-')):
        tag, machine, folder, name = 'linux_aarch64', 183, 'lib', 'libcureco_inference.so'
    else:
        raise RuntimeError(f'Unsupported SDK target: {target}')
    runtime = sdk / folder
    files = sorted(p for p in runtime.iterdir() if p.is_file() and
                   (p.suffix == '.dll' or '.so' in p.name))
    if not (runtime / name).is_file() or not any('onnxruntime' in p.name for p in files):
        raise RuntimeError('SDK must include Cureco and matching ONNX Runtime libraries')
    for binary in files:
        data = binary.read_bytes()
        if tag == 'win_amd64':
            offset = struct.unpack_from('<I', data, 0x3c)[0]
            valid = data[:2] == b'MZ' and data[offset:offset+4] == b'PE\x00\x00'
            actual = struct.unpack_from('<H', data, offset+4)[0]
        else:
            valid = data[:6] == b'\x7fELF\x02\x01'
            actual = struct.unpack_from('<H', data, 18)[0]
        if not valid or actual != machine:
            raise RuntimeError(f'Architecture mismatch: {binary}')
    licenses = sdk / 'licenses'
    for required in ('cureco-LICENSE', 'onnxruntime-LICENSE',
                     'onnxruntime-ThirdPartyNotices.txt', 'nlohmann-json-LICENSE.MIT'):
        if not (licenses / required).is_file():
            raise RuntimeError(f'Missing third-party notice: {required}')
    return tag, files, licenses


class NativeWheelHook(BuildHookInterface):
    def initialize(self, version, build_data):
        sdk = Path(os.environ.get('CURECO_NATIVE_SDK_DIR', Path(self.root) / '../native/runtime')).resolve()
        if not (sdk / 'manifest.json').is_file():
            raise RuntimeError('Build/package the native SDK into sdk/native/runtime or set '
                               'CURECO_NATIVE_SDK_DIR. A Python-only wheel is not supported.')
        tag, files, licenses = runtime_files(sdk)
        build_data['pure_python'] = False
        build_data['tag'] = f'py3-none-{tag}'
        for file in files:
            build_data['force_include'][str(file)] = f'cureco_inference_core/_native/{file.name}'
        build_data['force_include'][str(licenses)] = 'cureco_inference_core/_native/licenses'
        build_data['force_include'][str(sdk / 'manifest.json')] = 'cureco_inference_core/_native/manifest.json'
