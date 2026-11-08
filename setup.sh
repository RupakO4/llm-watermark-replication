#!/usr/bin/env bash
# One-time setup: fetch the authors' code at the commit used in the report and
# install the Python dependencies into the *current* environment.
#
#   conda create -n lm-watermark python=3.10 -y && conda activate lm-watermark   (or any venv)
#   bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=82922516930c02f8aa322765defdb5863d07a00e
DEST=third_party/lm-watermarking
if [[ ! -d "$DEST/.git" ]]; then
  git clone https://github.com/jwkirchenbauer/lm-watermarking.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin || true
git -C "$DEST" checkout --quiet "$COMMIT"
echo "upstream lm-watermarking at $(git -C "$DEST" rev-parse --short HEAD)"
${PYTHON:-python} -m pip install -r requirements.txt
echo "Setup done. Next: bash run_all.sh smoke"
