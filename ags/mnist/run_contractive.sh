#!/usr/bin/env bash
# The contractive-trajectory run behind fig7 (10 classes) and fig8 (digits 0-4): sweeps, finals and
# windows on the GPU in sequence; matched, optimized and PATE baselines on the CPU as each GPU phase
# finishes; then the release and the figures for both tasks.
#
#   HOURS=9.5 bash ags/mnist/run_contractive.sh
#
# Resumable: envelopes, releases and baseline cells are written as they complete and skipped on the
# next start. Logs under ags/mnist/runs/contractive10/logs/.
set -euo pipefail
repo_root="${AGT_REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
cd "$repo_root"
export PYTHONPATH="$repo_root"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PY="${PYTHON:-python}"
hours="${HOURS:-9.5}"
start=$(date +%s)
deadline=$(python3 -c "print($start + ($hours - 1.0) * 3600)")
logs="ags/mnist/runs/contractive10/logs"
mkdir -p "$logs"
echo "start $(date)  no new work after $(date -d @"${deadline%.*}")  logs in $logs"

wait_for() {  # wait_for <marker file>
  until [ -f "$1" ]; do sleep 60; if [ "$(date +%s)" -gt "${deadline%.*}" ] && [ ! -f "$1" ]; then return 1; fi; done
}

# ---- GPU chain
(
  $PY -u -m ags.mnist.run_contractive sweep --task mnist10 --device "${DEVICE:-cuda:0}" --deadline "$deadline"
  $PY -u -m ags.mnist.run_contractive sweep --task mnist5 --device "${DEVICE:-cuda:0}" --deadline "$deadline"
  $PY -u -m ags.mnist.run_contractive final --task mnist10 --device "${DEVICE:-cuda:0}" --deadline "$deadline"
  $PY -u -m ags.mnist.run_contractive final --task mnist5 --device "${DEVICE:-cuda:0}" --deadline "$deadline"
  $PY -u -m ags.mnist.run_contractive windows --task mnist10 --device "${DEVICE:-cuda:0}" --deadline "$deadline"
  $PY -u -m ags.mnist.run_contractive windows --task mnist5 --device "${DEVICE:-cuda:0}" --deadline "$deadline"
) > "$logs/gpu.log" 2>&1 &
pid_gpu=$!

# ---- CPU chain: baselines follow each GPU phase
(
  wait_for ags/mnist/runs/contractive10/SWEEP_DONE && $PY -u -m ags.mnist.run_contractive baselines --scale sweep --task mnist10 --deadline "$deadline"
  wait_for ags/mnist/runs/contractive5/SWEEP_DONE && $PY -u -m ags.mnist.run_contractive baselines --scale sweep --task mnist5 --deadline "$deadline"
  wait_for ags/mnist/runs/contractive10/FINAL_DONE && $PY -u -m ags.mnist.run_contractive baselines --scale full --task mnist10 --deadline "$deadline"
  wait_for ags/mnist/runs/contractive5/FINAL_DONE && $PY -u -m ags.mnist.run_contractive baselines --scale full --task mnist5 --deadline "$deadline"
) > "$logs/cpu.log" 2>&1 &
pid_cpu=$!

wait "$pid_gpu" || echo "GPU chain exited non-zero; see $logs/gpu.log"
wait "$pid_cpu" || echo "CPU chain exited non-zero; see $logs/cpu.log"

for task in mnist10 mnist5; do
  $PY -u -m ags.mnist.run_contractive release --task $task > "$logs/release_$task.log" 2>&1 || echo "release $task failed"
  for v in "" "--beta-rule dimension_free" "--certificate width"; do
    $PY -m ags.mnist.plot_contractive --task $task $v || echo "plot $task $v failed"
  done
done
echo "done $(date)  ($(( ($(date +%s) - start) / 60 )) min)"
