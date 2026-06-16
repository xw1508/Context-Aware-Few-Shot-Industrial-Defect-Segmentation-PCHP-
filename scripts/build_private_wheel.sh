#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python tools/build_pyc_wheel.py \
  --source model \
  --dist dist \
  --name pchp-private-model \
  --version 0.1.0

echo "Built private model wheel under dist/."
