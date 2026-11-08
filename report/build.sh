#!/usr/bin/env bash
# Build the report PDF from the latest analysis outputs.
set -euo pipefail
cd "$(dirname "$0")"
R="${RESULTS:-../results}"
rm -rf tables figures && cp -r "$R/tables" tables && cp -r "$R/figures" figures
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null || { tail -40 main.log; exit 1; }
echo "built report/main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages)"
