#!/usr/bin/env python3
"""Install only missing Colab extras without replacing core runtime packages."""

from __future__ import annotations

import importlib.util
import subprocess
import sys


NO_DEPS_PACKAGES = {
    "facenet_pytorch": "facenet-pytorch==2.6.0",
    "gdown": "gdown>=5.2.0",
    "seaborn": "seaborn>=0.13.0",
    "tqdm": "tqdm>=4.66.0",
}

WITH_DEPS_PACKAGES = {
    "mediapipe": "mediapipe>=0.10.0",
}

CORE_MODULES = ["cv2", "matplotlib", "numpy", "pandas", "PIL", "sklearn", "torch", "torchvision"]


def has_module(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def install_no_deps(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", package])


def install_with_deps(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def main() -> None:
    missing_core = [module for module in CORE_MODULES if not has_module(module)]
    if missing_core:
        raise SystemExit(
            "Missing expected Colab runtime packages: "
            + ", ".join(missing_core)
            + ". Use a standard Colab runtime instead of pip-replacing core dependencies."
        )

    installed = []
    for module, package in NO_DEPS_PACKAGES.items():
        if not has_module(module):
            install_no_deps(package)
            installed.append(package)

    for module, package in WITH_DEPS_PACKAGES.items():
        if not has_module(module):
            install_with_deps(package)
            installed.append(package)

    print("Installed Colab extras:" if installed else "Colab extras already available", installed)


if __name__ == "__main__":
    main()
