"""Report selected installed distribution versions without importing Paddle."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


PACKAGES = (
    "paddlepaddle-gpu", "paddlepaddle", "numpy", "opencv-python-headless",
    "opencv-python", "PyYAML", "pillow", "prettytable", "scikit-image",
    "scikit-learn", "scipy", "tqdm", "visualdl", "filelock",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result = {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "os": platform.system(), "machine": platform.machine(),
        "packages": packages, "paddle_imported": "paddle" in sys.modules,
        "scope": "Selected installed distribution metadata; not a complete dependency lock or CUDA probe.",
    }
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
