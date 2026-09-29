#!/usr/bin/env bash
# The overnight run behind fig5_combined_optimized: envelopes on the GPU and baselines on the CPU in
# parallel under one wall-clock deadline, then the release and the figure on whatever finished.
#
#   HOURS=9.5 bash ags/mnist/run_optimized.sh
#
# Resumable: every envelope, every baseline cell and every release is written as it completes and
# skipped on the next start. Logs under ags/mnist/runs/optimized/logs/.
set -euo pipefail
repo_root="${AGT_REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
cd "$repo_root"
export PYTHONPATH="$repo_root"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

hours="${HOURS:-9.5}"
start=$(date +%s)
# the sweep stops starting new work one hour before the budget ends; release and plot use the rest
deadline=$(python3 -c "print($start + ($hours - 1.0) * 3600)")
logs="ags/mnist/runs/optimized/logs"
mkdir -p "$logs"

echo "start $(date)  deadline for new work $(date -d @"${deadline%.*}")  logs in $logs"
${PYTHON:-python} -u -m ags.mnist.run_optimized envelopes --stage all --device "${DEVICE:-cuda:0}" --deadline "$deadline" \
  > "$logs/envelopes.log" 2>&1 &
pid_env=$!
${PYTHON:-python} -u -m ags.mnist.run_optimized baselines --deadline "$deadline" \
  > "$logs/baselines.log" 2>&1 &
pid_base=$!
wait "$pid_env" || echo "envelopes exited non-zero; see $logs/envelopes.log"
wait "$pid_base" || echo "baselines exited non-zero; see $logs/baselines.log"

${PYTHON:-python} -u -m ags.mnist.run_optimized release > "$logs/release.log" 2>&1
${PYTHON:-python} -u -m ags.mnist.plot_optimized
${PYTHON:-python} -u -m ags.mnist.plot_optimized --beta-rule dimension_free
${PYTHON:-python} -u -m ags.mnist.plot_optimized --beta-rule pld_coordinate
${PYTHON:-python} -u -m ags.mnist.plot_optimized --certificate width
echo "done $(date)  ($(( ($(date +%s) - start) / 60 )) min)"
