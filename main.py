"""Object tracking — project entry point.

Usage:
    uv run python main.py

This is a blank scaffold: PyTorch (ROCm on Linux, CUDA/CPU elsewhere) and
OpenCV are installed and ready. Put your tracker in a new module and call it
from ``main()`` below.
"""

from __future__ import annotations

import sys


def main() -> None:
    import torch

    print(f"torch  : {torch.__version__}")
    print(f"hip    : {torch.version.hip}   (ROCm/HIP build if not None)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device : {device}")
    if torch.cuda.is_available():
        print(f"gpu    : {torch.cuda.get_device_name(0)}")

    print()
    print("Blank project ready — implement your tracker in main.py.")
    print("Put input videos/frames in data/, write results to outputs/.")


if __name__ == "__main__":
    sys.exit(main())
