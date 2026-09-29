#!/usr/bin/env bash
set -euo pipefail

tail_artifact="ags/mnist/accounting/artifacts/pca_tail_average_rdp_overnight.json"
while [[ ! -s "${tail_artifact}" ]]; do
  sleep 30
done

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. ${PYTHON:-python} -u \
  ags/mnist/accounting/pca/run_strong_accounting.py \
  --device cuda:0 \
  --output ags/mnist/accounting/artifacts/pca_strong_accounting_overnight.json
