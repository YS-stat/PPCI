#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/reference_smoother_sensitivity}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"
PID_FILE="$OUT_ROOT/pilot_pids.txt"
: > "$PID_FILE"

launch() {
  local name="$1"
  shift
  local output_dir="$OUT_ROOT/$name"
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite $output_dir" >&2
    exit 1
  fi
  mkdir -p "$output_dir"
  nohup env PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PYTHON_BIN" "$@" --output-dir "$output_dir" \
    > "$output_dir/run.log" 2>&1 &
  echo "$! $name" >> "$PID_FILE"
}

launch income_sex1_gpu0 experiments/run_income.py \
  --seed 15100 --sexes 1 --ages 70,80,90,100 \
  --n-label 300 --n-unlab 10000 --unlab-reps 15 --label-reps 20 \
  --h-factors 1.0,1.2,1.4 \
  --lambda-factor-min 0.1 --lambda-factor-max 1000 --lambda-grid-size 41 \
  --lambda-grid-mode shrinking --tau-op 12 --tau-loc 4 --c-biases 22 \
  --bias-screens p1_label --constraint-fallback least_violation \
  --backend cuda --gpu-id "$GPU0" --workers 1 --save-replicates

launch income_sex2_gpu1 experiments/run_income.py \
  --seed 15300 --sexes 2 --ages 70,80,90,100 \
  --n-label 300 --n-unlab 10000 --unlab-reps 15 --label-reps 20 \
  --h-factors 1.0,1.2,1.4 \
  --lambda-factor-min 0.1 --lambda-factor-max 1000 --lambda-grid-size 41 \
  --lambda-grid-mode shrinking --tau-op 10 --tau-loc 3 --c-biases 15 \
  --bias-screens p1_label --constraint-fallback least_violation \
  --backend cuda --gpu-id "$GPU1" --workers 1 --save-replicates

launch blog_gpu2 experiments/run_blogfeedback.py \
  --seed 2025 --n-label 300 --n-unlab 10000 --n-x0 50 \
  --x0-indices 0,7,14,21 --unlab-reps 15 --unlab-rep-offset 3000 \
  --label-reps 20 --h-factors 0.8,0.9,1.0 \
  --lambda-factor-min 0.1 --lambda-factor-max 10000 --lambda-grid-size 81 \
  --lambda-grid-mode shrinking --tau-op 12 --tau-loc 4 --c-biases 300 \
  --bias-screens p1_label --constraint-fallback least_violation \
  --model lightgbm --model-n-jobs 4 --backend cuda --gpu-id "$GPU2" \
  --workers 1 --save-replicates

launch blog_gpu3 experiments/run_blogfeedback.py \
  --seed 2025 --n-label 300 --n-unlab 10000 --n-x0 50 \
  --x0-indices 28,35,42,49 --unlab-reps 15 --unlab-rep-offset 3000 \
  --label-reps 20 --h-factors 0.8,0.9,1.0 \
  --lambda-factor-min 0.1 --lambda-factor-max 10000 --lambda-grid-size 81 \
  --lambda-grid-mode shrinking --tau-op 12 --tau-loc 4 --c-biases 300 \
  --bias-screens p1_label --constraint-fallback least_violation \
  --model lightgbm --model-n-jobs 4 --backend cuda --gpu-id "$GPU3" \
  --workers 1 --save-replicates

sleep 5
while read -r pid name; do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "$name failed during startup; inspect $OUT_ROOT/$name/run.log" >&2
    exit 1
  fi
done < "$PID_FILE"

cat "$PID_FILE"
