#!/usr/bin/env bash
# Execute every notebooks/exp*.ipynb top-to-bottom with outputs saved (Rule 1).
#
# Usage:  bash scripts/run_all_notebooks.sh
# Env:    PY            interpreter (default: .venv/bin/python, else python3)
#         NOTEBOOK_KERNEL kernelspec name (default: mindsense; registered if absent)
#         NB_TIMEOUT    per-notebook timeout in seconds (default: 900)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "${PY:-}" ]; then
  PYTHON="$PY"
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi
KERNEL="${NOTEBOOK_KERNEL:-mindsense}"

# Make sure the kernelspec exists and points at the chosen interpreter (idempotent).
"$PYTHON" -m ipykernel install --user --name "$KERNEL" --display-name "Python ($KERNEL)" >/dev/null

shopt -s nullglob
notebooks=(notebooks/exp*.ipynb)
if [ "${#notebooks[@]}" -eq 0 ]; then
  echo "run_all_notebooks: no notebooks/exp*.ipynb found" >&2
  exit 1
fi

for nb in "${notebooks[@]}"; do
  echo "==> executing $nb"
  "$PYTHON" -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name="$KERNEL" \
    --ExecutePreprocessor.timeout="${NB_TIMEOUT:-900}" \
    "$nb"
done
echo "run_all_notebooks: executed ${#notebooks[@]} notebook(s)"
