#!/usr/bin/env bash
set -euo pipefail

input="ags/blobs/accounting/artifacts/blobs_matched_rdp_lp_complete_overnight.json"
while [[ ! -s "${input}" ]]; do
  sleep 30
done

PYTHONPATH=. ${PYTHON:-python} -u \
  ags/blobs/accounting/run_public_projection.py \
  --input "${input}" \
  --output ags/blobs/accounting/artifacts/blobs_public_projection_overnight.json
