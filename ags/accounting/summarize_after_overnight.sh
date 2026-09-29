#!/usr/bin/env bash
set -euo pipefail
repo_root="${AGT_REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
cd "$repo_root"
export PYTHONPATH="$repo_root"

required=(
  "ags/blobs/accounting/artifacts/blobs_matched_rdp_lp_complete_overnight.json"
  "ags/mnist/accounting/artifacts/pca_matched_rdp_overnight.json"
)
while true; do
  complete=1
  for artifact in "${required[@]}"; do
    [[ -f "$artifact" ]] || complete=0
  done
  [[ "$complete" -eq 1 ]] && break
  sleep 30
done
${PYTHON:-python} -u ags/accounting/summarize_overnight.py
