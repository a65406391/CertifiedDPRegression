#!/usr/bin/env bash
set -euo pipefail

required=(
  ags/blobs/accounting/artifacts/blobs_matched_rdp_lp_complete_overnight.json
  ags/blobs/accounting/artifacts/blobs_public_projection_overnight.json
  ags/blobs/accounting/artifacts/blobs_poisson_dpsgd_reference_overnight.json
  ags/mnist/accounting/artifacts/pca_strong_accounting_overnight.json
  ags/mnist/accounting/artifacts/pca_early_stop_strong_corrected_overnight.json
)
while true; do
  complete=1
  for artifact in "${required[@]}"; do
    if [[ ! -s "${artifact}" ]]; then
      complete=0
    fi
  done
  if [[ "${complete}" -eq 1 ]]; then
    break
  fi
  sleep 30
done

PYTHONPATH=. ${PYTHON:-python} \
  ags/accounting/summarize_strong_accounting.py
