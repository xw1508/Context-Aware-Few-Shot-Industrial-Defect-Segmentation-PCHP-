#!/usr/bin/env python
"""Create a source-free review release tree."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


INCLUDE_PATHS = [
    "README_REVIEW.md",
    "requirements.txt",
    "test.py",
    "config",
    "data_list",
    "FSSD-12",
    "util",
    "dist",
]

SCRIPT_FILES = [
    "setup.sh",
    "run_test.sh",
]

IGNORE_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "build",
    "result",
    "runs",
}


def ignore_filter(_dir: str, names: list[str]) -> set[str]:
    ignored = {name for name in names if name in IGNORE_NAMES}
    ignored.update(name for name in names if name.endswith((".pyc", ".pyo")))
    return ignored


def copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, ignore=ignore_filter)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="review_release", help="Output directory")
    args = parser.parse_args()

    root = Path.cwd()
    output = root / args.output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    wheel_files = list((root / "dist").glob("pchp_private_model-*.whl"))
    if not wheel_files:
        raise SystemExit("No private wheel found. Run `bash scripts/build_private_wheel.sh` first.")

    for rel in INCLUDE_PATHS:
        src = root / rel
        if src.exists():
            copy_path(src, output / rel)

    scripts_out = output / "scripts"
    scripts_out.mkdir(parents=True, exist_ok=True)
    for script_name in SCRIPT_FILES:
        copy_path(root / "scripts" / script_name, scripts_out / script_name)

    (output / "model").mkdir(exist_ok=True)
    (output / "model" / "README.md").write_text(
        "Private model source is not included. Install `dist/pchp_private_model-*.whl`.\n",
        encoding="utf-8",
    )

    forbidden = list(output.glob("model/*.py"))
    if forbidden:
        raise SystemExit(f"Model source leaked into release: {forbidden}")

    print(output)


if __name__ == "__main__":
    main()
