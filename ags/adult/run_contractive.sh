#!/usr/bin/env bash
# The Adult contractive comparison behind fig10: the sweep on three workers (two on the first GPU, one
# on the second), each group's baselines on the second device as soon as the sweep has certified that
# group's finalist, then the release and the figures. Same shape as ags/imdb/run_contractive.sh.
#
#   HOURS=8 bash ags/adult/run_contractive.sh
#
# Resumable. Logs under ags/adult/runs/contractive/logs/.
set -euo pipefail
repo_root="${AGT_REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
cd "$repo_root"
export PYTHONPATH="$repo_root"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PY="${PYTHON:-python}"
hours="${HOURS:-8}"
start=$(date +%s)
deadline=$(python3 -c "print($start + ($hours - 1.0) * 3600)")
runs="ags/adult/runs/contractive"
logs="$runs/logs"
mkdir -p "$logs"
dev_sweep="${DEVICE_SWEEP:-cuda:0}"
dev_base="${DEVICE_BASELINES:-cuda:1}"
echo "start $(date)  no new work after $(date -d @"${deadline%.*}")  sweep on $dev_sweep, baselines on $dev_base, logs in $logs"

wait_for() {  # wait_for <marker file>; gives up at the deadline
  until [ -f "$1" ]; do sleep 60; if [ "$(date +%s)" -gt "${deadline%.*}" ]; then return 1; fi; done
}

$PY -u -m ags.adult.shared > "$logs/data.log" 2>&1
pids=()
i=0
for dev in ${WORKERS:-$dev_sweep $dev_sweep $dev_base}; do  # one device per worker, word-split on purpose
  $PY -u -m ags.adult.run_contractive sweep --device "$dev" --deadline "$deadline" > "$logs/sweep_$i.log" 2>&1 &
  pids+=($!)
  i=$((i + 1))
done

for group in logistic logistic_warm mlp mlp_warm; do
  (
    wait_for "$runs/$(echo "$group" | tr a-z A-Z)_DONE" \
      && $PY -u -m ags.adult.run_contractive baselines --group "$group" --device "$dev_base" --deadline "$deadline"
  ) > "$logs/baselines_$group.log" 2>&1 &
done

for pid in "${pids[@]}"; do wait "$pid" || echo "a sweep worker exited non-zero; see $logs/sweep_*.log"; done
wait

$PY -u -m ags.adult.run_contractive release > "$logs/release.log" 2>&1 || echo "release failed"
for v in "" "--group warm" "--beta-rule aggregate" "--certificate width"; do
  $PY -m ags.adult.plot_contractive $v || echo "plot $v failed"
done
echo "done $(date)  ($(( ($(date +%s) - start) / 60 )) min)"
