#!/bin/sh
set -eu

model_path=${JET_MODEL_PATH:-models/Qwen3-0.6B-Q8_0.gguf}
work_dir=${JET_ACCURACY_DIR:-tests/accuracy/generated}
boolq_limit=${JET_BOOLQ_LIMIT:-256}
mmlu_per_subject=${JET_MMLU_PER_SUBJECT:-5}
backend=${JET_BACKEND:-cpu}

case "$backend" in
  cpu)
    set -- cargo run --release -p jet-cli -- judge
    ;;
  vulkan)
    set -- cargo run --release -p jet-cli --features vulkan -- judge
    ;;
  *)
    echo "run-accuracy-eval: JET_BACKEND must be cpu or vulkan" >&2
    exit 2
    ;;
esac

python3 scripts/prepare-accuracy-data.py \
  --output-dir "$work_dir" \
  --boolq-limit "$boolq_limit" \
  --mmlu-per-subject "$mmlu_per_subject"

mise exec -- "$@" \
  --backend "$backend" \
  --model-path "$model_path" \
  --model-id qwen/qwen3-0.6b-q8_0 \
  --thinking disabled \
  --input "$work_dir/requests.jsonl" \
  --output "$work_dir/responses.jsonl"

python3 scripts/evaluate-accuracy.py \
  --gold "$work_dir/gold.jsonl" \
  --responses "$work_dir/responses.jsonl" \
  --report "$work_dir/report.json"
