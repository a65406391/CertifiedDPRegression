#!/usr/bin/env bash
set -euo pipefail

repo_root="${AGT_REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
cd "$repo_root"
export PYTHONPATH="$repo_root"

root="ags/blobs/runs/certified"
low_required=(
  "$root/crossradii_a_linear_b20_ns200_F10_MILP/MANIFEST.json"
  "$root/crossradii_b_linear_b20_ns200_F10_MILP/MANIFEST.json"
  "$root/crossradii_c_linear_b20_ns200_F10_MILP/MANIFEST.json"
)

wait_for_manifests() {
  while true; do
    complete=1
    for manifest in "$@"; do
      if [[ ! -f "$manifest" ]] || ! grep -q '"status": "completed"' "$manifest"; then
        complete=0
      fi
    done
    [[ "$complete" -eq 1 ]] && break
    sleep 30
  done
}

wait_for_manifests "${low_required[@]}"
${PYTHON:-python} -u ags/blobs/accounting/run_rdp_comparison.py \
  --output ags/blobs/accounting/artifacts/blobs_matched_rdp_k20_overnight.json \
  --seeds 101

high_required=(
  "$root/crossradii_high_a_linear_b20_ns200_F10_MILP/MANIFEST.json"
  "$root/crossradii_high_b_linear_b20_ns200_F10_MILP/MANIFEST.json"
)
wait_for_manifests "${high_required[@]}"
${PYTHON:-python} -u ags/blobs/accounting/run_rdp_comparison.py \
  --output ags/blobs/accounting/artifacts/blobs_matched_rdp_k200_overnight.json \
  --seeds 101
