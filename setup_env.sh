#!/usr/bin/env bash
# Creates .venv in this folder and installs requirements.txt (macOS / Linux).
# Afterwards activate it in YOUR shell:  source .venv/bin/activate
set -e
cd "$(dirname "$0")"
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  VIRTUAL_ENV="$PWD/.venv" uv pip install -r requirements.txt
else
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python -c "import numpy, pandas, scipy, sklearn, matplotlib, tabulate, cv2; print('environment OK')"
echo "next: source .venv/bin/activate"
