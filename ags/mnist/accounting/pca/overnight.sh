#!/usr/bin/env bash
set -euo pipefail

repo_root="${AGT_REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
cd "$repo_root"
export PYTHONPATH="$repo_root"
export CUDA_VISIBLE_DEVICES=0

artifact_dir="ags/mnist/accounting/artifacts/pca_overnight_matched"
mkdir -p "$artifact_dir"

for epsilon in 1 2 4 8 10 20 50 100; do
  ags_out="$artifact_dir/one_shot_ags_eps_${epsilon}.json"
  if [[ ! -e "$ags_out" ]]; then
    ${PYTHON:-python} -u -m ags.mnist.probes.extended_accounting release \
      --output "$ags_out" --epsilon "$epsilon" --delta 1e-5 --seeds 41 \
      --N 2000 --n-test 1000 --d 64 --batch-size 2000 --steps 40000 \
      --lr 0.05 --eta 0.01 --gamma 0.1 --device cuda
  fi

  dp_out="$artifact_dir/dpsgd_eps_${epsilon}.json"
  if [[ ! -e "$dp_out" ]]; then
    ${PYTHON:-python} -u -m ags.mnist.probes.extended_accounting dpsgd \
      --output "$dp_out" --epsilon "$epsilon" --delta 1e-5 --seeds 41 \
      --N 2000 --n-test 1000 --d 64 --batch-size 2000 --steps 40000 \
      --lr 0.05 --eta 0.01 --gamma 0.1 --device cuda
  fi
done
