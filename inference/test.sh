#!/bin/bash
#
# Inference / pass@k evaluation. Wraps evaluate_model.py.
#
# Usage:
#   bash inference/test.sh <model> [--gpu-memory-util F] [--test-data D] [--prompt-type qwen|llama] [--num-samples N] [--max-length N] [--model-name NAME]
#
# <model> may be either a local checkpoint directory (absolute or relative)
# or a Hugging Face model id (e.g. "Qwen/Qwen2.5-0.5B"); HF ids are resolved
# by the inference backend (sglang) directly.
#
# Override result/checkpoint roots via environment variables:
#   CHECKPOINTS_ROOT   directory holding RL training checkpoints (default ./checkpoints)
#   MODELS_ROOT        directory holding base HF models           (default ../models)
#   RESULTS_DIR        where to write the per-question JSON       (default ./results)

set -euo pipefail

CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-./checkpoints}"
MODELS_ROOT="${MODELS_ROOT:-../models}"
RESULTS_DIR="${RESULTS_DIR:-./results}"

# Defaults; override via flags.
testdata=simplelr_qwen_level1to4@test
max_length=5120
prompt_type=qwen
num_samples=32
gpu_memory_util=0.75
temperature=0.7
model_name=""


model_path="${1:?Usage: $0 <model> [step] [--flag value ...]}"
shift


while [ $# -gt 0 ]; do
    case "$1" in
        --gpu-memory-util)  gpu_memory_util="$2";    shift 2 ;;
        --test-data)        testdata="$2";           shift 2 ;;
        --prompt-type)      prompt_type="$2";        shift 2 ;;
        --num-samples)      num_samples="$2";        shift 2 ;;
        --max-length)       max_length="$2";         shift 2 ;;
        --model-name)       model_name="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# If it looks like a local path (starts with /, ./, ../) require it to exist.
# Otherwise treat as a Hugging Face model id and let the inference backend resolve it.
case "$model_path" in
    /*|./*|../*)
        if [ ! -d "$model_path" ]; then
            echo "Model path $model_path does not exist"
            exit 1
        fi
        ;;
esac

if [ -n "$model_name" ]; then
    model_name="$model_name"
else
    model_name=$(basename "$model_path")
fi

echo "Evaluating model: $model_name"
echo "Test data:        $testdata"
echo "Prompt type:      $prompt_type"
echo "Samples/problem:  $num_samples"
echo "Max length:       $max_length"
echo "Save name:        $model_name"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
python -u "$SCRIPT_DIR/evaluate_model.py" \
    --model-path       "$model_path" \
    --tokenizer-path   "$model_path" \
    --test-data        "$testdata" \
    --num-samples      "$num_samples" \
    --max-new-tokens   "$max_length" \
    --prompt-type      "$prompt_type" \
    --save-path        "$RESULTS_DIR" \
    --model-name       "$model_name" \
    --temperature      "$temperature" \
    --gpu-memory-util  "$gpu_memory_util" \
    --use-sglang
