r"""Detection — benchmark entry point.

Usage:
    uv run python main.py --manifest data/flickrlogos32/manifest.json \\
        --models sift orb akaze --output outputs/flickrlogos32
    uv run python main.py --protocol hpatches \\
        --models sift orb akaze --output outputs/hpatches
    uv run python main.py ui --port 8765

The benchmark implementation lives in :mod:`benchmark.cli`; the inspector UI in
:mod:`src.ui`.
"""

from __future__ import annotations

import sys

from osid.benchmark.cli import main as run_benchmark

UI_COMMANDS = {"ui", "serve"}


def main() -> int:
    """Dispatch to the benchmark or the inspector UI.

    Returns:
        Process exit code.
    """
    argv = sys.argv[1:]
    if argv and argv[0] in UI_COMMANDS:
        from osid.ui.cli import main as run_ui

        return run_ui(argv[1:])
    return run_benchmark()


if __name__ == "__main__":
    sys.exit(main())
