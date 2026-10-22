#!/usr/bin/env bash
set -euo pipefail
FTR_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FTR_OUTPUT="$FTR_DIR/runs/view_$(date +%Y%m%d_%H%M%S)_$$"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
echo "FT-R GT output: $FTR_OUTPUT"
"${ISAAC_PYTHON:-python}" "$FTR_DIR/run_isaac.py" --mode schedule --seconds 25 --output "$FTR_OUTPUT" "$@"
if [[ -f "$FTR_OUTPUT/FAILED.txt" || ! -f "$FTR_OUTPUT/result.json" ]]; then
    echo "Isaac run did not complete; inspect $FTR_OUTPUT" >&2
    exit 1
fi
