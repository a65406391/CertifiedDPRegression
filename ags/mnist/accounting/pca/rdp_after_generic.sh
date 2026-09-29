#!/usr/bin/env bash
set -euo pipefail

repo_root="${AGT_REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
cd "$repo_root"
export PYTHONPATH="$repo_root"
export CUDA_VISIBLE_DEVICES=0

sentinel="ags/mnist/accounting/artifacts/pca_overnight_matched/dpsgd_eps_100.json"
while [[ ! -f "$sentinel" ]]; do
  sleep 30
done

${PYTHON:-python} -u ags/mnist/accounting/pca/run_matched_rdp.py \
  --output ags/mnist/accounting/artifacts/pca_matched_rdp_overnight.json \
  --seeds 41 --device cuda
