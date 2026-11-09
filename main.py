r"""Detection — benchmark entry point.

Usage:
    uv run python main.py --manifest data/flickrlogos32/manifest.json \\
        --models sift orb akaze --output outputs/flickrlogos32
    uv run python main.py --protocol hpatches \\
        --models sift orb akaze --output outputs/hpatches
    uv run python main.py --protocol retrieval \\
        --models sift orb akaze --output outputs/paris6k

The implementation lives in :mod:`src.pipeline`.
"""

from __future__ import annotations

import sys

from src.pipeline import main as run_benchmark


def main() -> int:
    """Run the one-shot instance detection benchmark.

    Returns:
        Process exit code.
    """
    return run_benchmark()


if __name__ == "__main__":
    sys.exit(main())
