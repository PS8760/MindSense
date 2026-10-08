#!/usr/bin/env bash
# Lightweight notebook smoke test for CI (Section 13 / CI contract).
#
# With MINDSENSE_SYNTHETIC=1 (set by the workflow) this generates seeded
# synthetic stand-ins for every dataset with no raw files, then executes the
# two data-track notebooks end to end. Heavy training notebooks (exp03+) are
# deliberately excluded: CI only proves the data path stays green.
#
# Env:  MINDSENSE_SYNTHETIC  "1" to synthesise missing datasets (default: 0)
#       SMOKE_NOTEBOOKS      space-separated notebook stems (default: exp01 + exp02)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "${PY:-}" ]; then
  PYTHON="$PY"
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi
KERNEL="${NOTEBOOK_KERNEL:-mindsense-smoke}"
export NOTEBOOK_KERNEL="$KERNEL"

# Register a kernel for this interpreter (idempotent; CI has no pre-existing one).
"$PYTHON" -m ipykernel install --user --name "$KERNEL" --display-name "Python ($KERNEL)" >/dev/null

if [ "${MINDSENSE_SYNTHETIC:-0}" = "1" ]; then
  echo "==> generating synthetic stand-ins for missing datasets"
  "$PYTHON" scripts/make_synthetic_data.py --all-missing
fi

shopt -s nullglob
stems=${SMOKE_NOTEBOOKS:-"exp01_data_pipeline exp02_eda"}
status=0
for stem in $stems; do
  nb="notebooks/${stem}.ipynb"
  if [ ! -f "$nb" ]; then
    echo "notebook_smoke: skip (missing $nb)"
    continue
  fi
  echo "==> smoke-executing $nb"
  if ! "$PYTHON" -m jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name="$KERNEL" \
      --ExecutePreprocessor.timeout="${NB_TIMEOUT:-900}" "$nb"; then
    echo "notebook_smoke: FAILED on $nb" >&2
    status=1
  fi
done

if [ "$status" -ne 0 ]; then
  exit "$status"
fi
echo "notebook_smoke: OK"
