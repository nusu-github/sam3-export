"""Run the CUDA ``torch.export`` round-trip gate for every public cut."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    tests = (
        "tests/test_export_smoke.py",
        "tests/test_vision_tower_export.py",
        "tests/test_prompt_interactive_export.py",
        "tests/test_remaining_export_cuts.py",
    )
    return subprocess.call([sys.executable, "-m", "pytest", "-q", *tests], cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
