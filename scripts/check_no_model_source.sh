#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if find model -maxdepth 1 -type f -name '*.py' | grep -q .; then
  echo "Private source files are still present under model/*.py."
  echo "Do not upload these files to the review repository."
  find model -maxdepth 1 -type f -name '*.py'
  exit 1
fi

echo "OK: no model/*.py source files found."
