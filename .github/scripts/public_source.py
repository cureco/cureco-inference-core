"""Check source for accidental publication hazards; never prints secret values."""
import argparse
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
TOP = {".github", "sdk", "examples", "tests", "docs"}
FILES = {"README.md", "LICENSE", "CHANGELOG.md", "SECURITY.md", ".gitignore",
         "Directory.Build.props", "mkdocs.yml"}
FORBIDDEN = {".venv", ".git", "__pycache__", "bin", "obj", "build", "dist",
             "runtime", "artifacts", ".vs", ".pytest_cache"}
SUFFIXES = {".pfx", ".p12", ".pem", ".key", ".dll", ".exe", ".onnx", ".pyc"}
RULES = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "access-token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})"),
    "developer-path": re.compile(r"(?i)(?:[a-z]:[\\/]+Users[\\/]|/home/[A-Za-z0-9_-]+/)"),
    "private-network-url": re.compile(r"https?://(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)(?=[:/\s]|$)"),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    listing = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, check=True, stdout=subprocess.PIPE).stdout
    names = sorted({name.decode("utf-8") for name in listing.split(b"\0")
                    if name and (ROOT / name.decode("utf-8")).is_file()})
    issues = []
    for name in names:
        path = Path(name)
        if path.parts[0] not in TOP and name not in FILES:
            issues.append((name, "unexpected-top-level-file"))
        if any(p in FORBIDDEN for p in path.parts) or path.suffix.lower() in SUFFIXES or path.name.startswith(".env"):
            issues.append((name, "non-source-or-sensitive-file"))
        if (ROOT / name).is_symlink():
            issues.append((name, "symlink"))
            continue
        try:
            content = (ROOT / name).read_text(encoding="utf-8-sig")
        except UnicodeError:
            issues.append((name, "binary-source-file"))
            continue
        for label, rule in RULES.items():
            if rule.search(content):
                issues.append((name, label))
    for name, category in issues:
        print(f"{category}: {name}")
    if issues:
        raise SystemExit(1)
    print(f"PASS: {len(names)} public source files checked")
    if args.archive:
        if args.archive.exists():
            parser.error("Archive exists; choose a fresh output")
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.archive, "x", zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(ROOT / name, "cureco-inference-core/" + name)

if __name__ == "__main__":
    main()
