#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

input=tests/accuracy/generated-mmlu-full-a770
output=tests/accuracy/generated
date_tag=${JET_A770_RUN_TAG:-20260928}
binary=${JET_A770_BINARY:-target/release/jet}

if [[ ! -f "$input/requests.jsonl" || ! -f "$input/gold.jsonl" ]]; then
  python scripts/prepare-accuracy-data.py --boolq-limit 0 --mmlu-all --ascii-json \
    --output-dir "$input"
fi

run_model() {
  local name=$1 model=$2 model_id=$3
  shift 3
  local run_dir="$output/a770-${name}-mmlu-full-${date_tag}"
  if [[ -e "$run_dir" ]]; then
    if [[ -f "$run_dir/summary.json" ]] && python - "$run_dir/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding='utf-8'))
assert summary['request_count_per_run'] == 14042
assert summary['completed_runs'] == 1
assert summary['runs'][0]['response_count'] == 14042
assert summary['runs'][0]['failures'] == 0
PY
    then
      echo "Already complete: $run_dir"
      return
    fi
    echo "Incomplete output directory exists: $run_dir" >&2
    exit 1
  fi
  python scripts/benchmark-model.py \
    --binary "$binary" --model-path "models/$model" --model-id "$model_id" \
    --input "$input/requests.jsonl" --gold "$input/gold.jsonl" \
    --output-dir "$run_dir" --runs 1 --backend vulkan --batch-requests 8 \
    --extra-args "$@" --max-sequences 2 --micro-batch 256 \
    --max-output-rows 256 --threads 8 --no-mmap
}

run_model qwen35-08b Qwen3.5-0.8B-Q8_0.gguf qwen/qwen3.5-0.8b-q8_0
run_model qwen35-2b Qwen3.5-2B-Q8_0.gguf qwen/qwen3.5-2b-q8_0
