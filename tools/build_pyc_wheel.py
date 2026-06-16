#!/usr/bin/env python
"""Build a pyc-only wheel for the private model package."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import py_compile
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path


def normalize_dist_name(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def wheel_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}-none-any"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def compile_to_legacy_pyc(source_file: Path, package_root: Path, wheel_root: Path) -> None:
    rel = source_file.relative_to(package_root)
    out_path = wheel_root / package_root.name / rel.with_suffix(".pyc")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    py_compile.compile(str(source_file), cfile=str(out_path), doraise=True)


def copy_orphan_pyc(package_root: Path, wheel_root: Path, allowed_modules: set[str]) -> None:
    cache_dir = package_root / "__pycache__"
    if not cache_dir.exists():
        return

    for pyc in cache_dir.glob("*.pyc"):
        module_name = pyc.name.split(".cpython-", 1)[0]
        if module_name not in allowed_modules:
            continue
        if (package_root / f"{module_name}.py").exists():
            continue
        target = wheel_root / package_root.name / f"{module_name}.pyc"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pyc, target)


def write_dist_info(wheel_root: Path, dist_info: str, name: str, version: str) -> Path:
    info_dir = wheel_root / dist_info
    info_dir.mkdir(parents=True, exist_ok=True)

    (info_dir / "METADATA").write_text(
        "\n".join(
            [
                "Metadata-Version: 2.1",
                f"Name: {name}",
                f"Version: {version}",
                "Summary: Private precompiled model package for review",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (info_dir / "WHEEL").write_text(
        "\n".join(
            [
                "Wheel-Version: 1.0",
                "Generator: tools/build_pyc_wheel.py",
                "Root-Is-Purelib: true",
                f"Tag: {wheel_tag()}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (info_dir / "top_level.txt").write_text("model\n", encoding="utf-8")
    return info_dir / "RECORD"


def write_record(wheel_root: Path, record_path: Path) -> None:
    rows = []
    for path in sorted(p for p in wheel_root.rglob("*") if p.is_file()):
        rel = path.relative_to(wheel_root).as_posix()
        if path == record_path:
            rows.append([rel, "", ""])
        else:
            rows.append([rel, hash_file(path), str(path.stat().st_size)])

    with record_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def zip_wheel(wheel_root: Path, wheel_file: Path) -> None:
    with zipfile.ZipFile(wheel_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in wheel_root.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(wheel_root).as_posix())


def build(args: argparse.Namespace) -> Path:
    package_root = Path(args.source).resolve()
    if not package_root.exists():
        raise SystemExit(f"Source package does not exist: {package_root}")
    if not (package_root / "PCHP_test.py").exists():
        raise SystemExit(f"Expected private model source at {package_root / 'PCHP_test.py'}")

    dist_dir = Path(args.dist).resolve()
    dist_dir.mkdir(parents=True, exist_ok=True)

    normalized = normalize_dist_name(args.name)
    dist_info = f"{normalized}-{args.version}.dist-info"
    wheel_name = f"{normalized}-{args.version}-{wheel_tag()}.whl"
    wheel_file = dist_dir / wheel_name

    with tempfile.TemporaryDirectory(prefix="pchp_pyc_wheel_") as temp_name:
        wheel_root = Path(temp_name) / "wheel"
        package_out = wheel_root / package_root.name
        package_out.mkdir(parents=True, exist_ok=True)

        init_source = Path(temp_name) / "__init__.py"
        init_source.write_text("", encoding="utf-8")
        py_compile.compile(str(init_source), cfile=str(package_out / "__init__.pyc"), doraise=True)

        for source_file in sorted(package_root.glob("*.py")):
            compile_to_legacy_pyc(source_file, package_root, wheel_root)

        copy_orphan_pyc(package_root, wheel_root, set(args.include_pyc_module))

        record_path = write_dist_info(wheel_root, dist_info, args.name, args.version)
        write_record(wheel_root, record_path)

        if wheel_file.exists():
            wheel_file.unlink()
        zip_wheel(wheel_root, wheel_file)

    return wheel_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Private source package directory, e.g. model")
    parser.add_argument("--dist", required=True, help="Output dist directory")
    parser.add_argument("--name", default="pchp-private-model", help="Distribution name")
    parser.add_argument("--version", default=time.strftime("%Y.%m.%d"), help="Distribution version")
    parser.add_argument(
        "--include-pyc-module",
        action="append",
        default=["transformer"],
        help="Existing __pycache__ module to include when no .py source is present.",
    )
    args = parser.parse_args()

    wheel_file = build(args)
    print(wheel_file)


if __name__ == "__main__":
    main()
