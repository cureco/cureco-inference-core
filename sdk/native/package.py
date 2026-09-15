"""Package an installed native SDK. Does not publish or sign."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--ort-root", type=Path, required=True)
    parser.add_argument("--json-license", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=["win-x64", "linux-x64", "linux-arm64"], required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a fresh directory")
    root = Path(__file__).resolve().parents[2]
    notices = {
        "cureco-LICENSE": root / "LICENSE",
        "onnxruntime-LICENSE": args.ort_root / "LICENSE",
        "onnxruntime-ThirdPartyNotices.txt": args.ort_root / "ThirdPartyNotices.txt",
        "nlohmann-json-LICENSE.MIT": args.json_license,
    }
    for source in notices.values():
        if not source.is_file():
            parser.error("Required license notice is missing")
    subprocess.run(["cmake", "--install", str(args.build_dir), "--config", "Release", "--prefix", str(args.output.resolve())], check=True)
    runtime = args.output / ("bin" if args.target.startswith("win-") else "lib")
    runtime.mkdir(exist_ok=True)
    for pattern in ("*.dll", "*.so*"):
        for source in (args.ort_root / "lib").glob(pattern):
            shutil.copy2(source, runtime / source.name)
    licenses = args.output / "licenses"
    licenses.mkdir()
    for name, source in notices.items():
        shutil.copy2(source, licenses / name)
    shutil.copy2(Path(__file__).with_name("README.md"), args.output / "README.md")
    (args.output / "manifest.json").write_text(json.dumps({
        "abi_version": 1, "target": args.target, "ort_sdk": args.ort_root.name
    }, indent=2), encoding="utf-8")
    print(shutil.make_archive(str(args.output), "zip", args.output))

if __name__ == "__main__":
    main()
