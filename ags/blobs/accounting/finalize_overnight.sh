#!/usr/bin/env bash
set -euo pipefail

repo_root="${AGT_REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
cd "$repo_root"
export PYTHONPATH="$repo_root"

root="ags/blobs/runs/certified"
required=(
  "$root/crossradii_a_linear_b20_ns200_F10_MILP/MANIFEST.json"
  "$root/crossradii_b_linear_b20_ns200_F10_MILP/MANIFEST.json"
  "$root/crossradii_c_linear_b20_ns200_F10_MILP/MANIFEST.json"
)

while true; do
  complete=1
  for manifest in "${required[@]}"; do
    if [[ ! -f "$manifest" ]] || ! grep -q '"status": "completed"' "$manifest"; then
      complete=0
    fi
  done
  [[ "$complete" -eq 1 ]] && break
  sleep 30
done

${PYTHON:-python} -u ags/blobs/accounting/run_cross_radius_comparison.py \
  --output ags/blobs/accounting/artifacts/blobs_cross_radius_pld_k20_overnight.json \
  --seeds 101 --ags-interval 1e-4 --dp-interval 1e-3
