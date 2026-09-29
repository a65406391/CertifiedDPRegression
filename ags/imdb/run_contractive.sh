#!/usr/bin/env bash
# The IMDB contractive comparison behind fig9: the embeddings, then the sweep on three workers (two on
# the first GPU, one on the second: one certified step is 0.6-0.9 s at 20k reviews and one process
# leaves the GPU half idle); each group's baselines start on the second device as soon as the sweep
# has certified that group's finalist; then the release and the figures.
#
#   HOURS=8 bash ags/imdb/run_contractive.sh
#
# Resumable: envelopes, releases and baseline cells are written as they complete and skipped on the
# next start. Logs under ags/imdb/runs/contractive/logs/.
set -euo pipefail
repo_root="${AGT_REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
cd "$repo_root"
export PYTHONPATH="$repo_root"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PY="${PYTHON:-python}"
hours="${HOURS:-8}"
start=$(date +%s)
deadline=$(python3 -c "print($start + ($hours - 1.0) * 3600)")
runs="ags/imdb/runs/contractive"
logs="$runs/logs"
mkdir -p "$logs"
dev_sweep="${DEVICE_SWEEP:-cuda:0}"
dev_base="${DEVICE_BASELINES:-cuda:1}"
echo "start $(date)  no new work after $(date -d @"${deadline%.*}")  sweep on $dev_sweep, baselines on $dev_base, logs in $logs"

wait_for() {  # wait_for <marker file>; gives up at the deadline
  until [ -f "$1" ]; do sleep 60; if [ "$(date +%s)" -gt "${deadline%.*}" ]; then return 1; fi; done
}

# ---- the sweep: several workers share every configuration's radii through lock files (see
# run_contractive.claim); WORKERS lists one device per worker
$PY -u -m ags.imdb.run_contractive embed --device "$dev_sweep" > "$logs/embed.log" 2>&1
pids=()
i=0
for dev in ${WORKERS:-$dev_sweep $dev_sweep $dev_base}; do  # one device per worker, word-split on purpose
  $PY -u -m ags.imdb.run_contractive sweep --long --device "$dev" --deadline "$deadline" > "$logs/sweep_$i.log" 2>&1 &
  pids+=($!)
  i=$((i + 1))
done

# ---- baselines, one process per group, each waiting for its finalist
for group in logistic logistic_warm logistic_long mlp mlp_warm; do
  (
    wait_for "$runs/$(echo "$group" | tr a-z A-Z)_DONE" \
      && $PY -u -m ags.imdb.run_contractive baselines --group "$group" --device "$dev_base" --deadline "$deadline"
  ) > "$logs/baselines_$group.log" 2>&1 &
done

for pid in "${pids[@]}"; do wait "$pid" || echo "a sweep worker exited non-zero; see $logs/sweep_*.log"; done
wait

$PY -u -m ags.imdb.run_contractive release > "$logs/release.log" 2>&1 || echo "release failed"
for v in "" "--group warm" "--group long" "--beta-rule dimension_free" "--certificate width"; do
  $PY -m ags.imdb.plot_contractive $v || echo "plot $v failed"
done
echo "done $(date)  ($(( ($(date +%s) - start) / 60 )) min)"
