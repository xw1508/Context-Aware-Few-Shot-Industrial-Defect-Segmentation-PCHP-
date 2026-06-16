#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-config/SSD/fold0_resnet50_test.yaml}"
EXTRA_ARGS=("${@:2}")

if [ -f ".venv/Scripts/activate" ]; then
  # Windows Git Bash
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
  # Linux/macOS
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("model.PCHP_test") is None:
    sys.exit(
        "Private model package is not installed. Run `bash scripts/setup.sh` first."
    )
PY

python test.py --config "$CONFIG_PATH" --save_vis "${EXTRA_ARGS[@]}"
