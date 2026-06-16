#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

if [ -f ".venv/Scripts/activate" ]; then
  # Windows Git Bash
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
else
  # Linux/macOS
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if ! python - <<'PY'
import ssl
print(ssl.OPENSSL_VERSION)
PY
then
  cat <<'EOF'
The selected Python cannot import the ssl module, so pip cannot download HTTPS
packages from PyPI or the PyTorch index.

Please use a Python build with SSL support, for example:
  conda create -n pchp_review python=3.9 openssl pip -y
  conda activate pchp_review
  bash scripts/setup.sh

Or install Python from python.org/Anaconda and make sure `python -c "import ssl"`
works before running this script again.
EOF
  exit 1
fi

python -m pip install --upgrade pip

# Keep the filtered file in case a reviewer adds environment comments locally.
REQ_FILE="$(mktemp)"
grep -vE '^[[:space:]]*(python|logger)==' requirements.txt > "$REQ_FILE"

if grep -qE '^torch==.*\+cu' "$REQ_FILE"; then
  python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r "$REQ_FILE"
else
  python -m pip install -r "$REQ_FILE"
fi

rm -f "$REQ_FILE"

WHEEL_FILE="$(ls dist/pchp_private_model-*.whl 2>/dev/null | head -n 1 || true)"
if [ -z "$WHEEL_FILE" ]; then
  echo "No private model wheel found in dist/."
  echo "Ask the author for dist/pchp_private_model-*.whl or run scripts/build_private_wheel.sh locally."
  exit 1
fi

python -m pip install --force-reinstall "$WHEEL_FILE"

echo "Environment is ready."
