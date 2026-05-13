#!/bin/bash
#
# End-to-end data-augmentation pipeline:
#   1. augment.py / decompose.py   (OpenAI: similar problems / sub-problems)
#   2. generate_answer_gemini.py    (Gemini: independent answer)
#   3. extract_answer.py            (extract \boxed{} answer from Gemini output)
#   4. filter_data.py               (keep only items where Gemini agrees with model)
#   5. prepare_grad_sim_data.py     (concatenate to a JSONL for rollouts)
#
# Configure via env vars (defaults shown):
#   DATA_FILE=../data/simplelr_qwen_level1to4/test_parsed.jsonl
#                                seed problems JSONL (must contain "id"/"extra_info.index" + "question")
#   PROMPT_IDS_FILE=../unlearnable_prompt_ids/qwen_0.5b_math_level1to4_unlearnable.json
#                                JSON list of seed-problem ids to augment/decompose
#   SETTING=qwen_0.5b_math_level1to4
#   DATA_TYPE=unlearnable
#   MODE=augmented               augmented | decomposed
#   TOTAL_IDX=5                  shard count for the OpenAI step
#
# Step toggles:
#   STEP1=true|false (default true), ..., STEP5=true|false
#
# Required env (if the corresponding step is on):
#   OPENAI_API_KEY     (steps 1 and 3)
#   GEMINI_API_KEY     (step 2)

set -euo pipefail

# Run from the data_augmentation/ directory regardless of where the user
# called the script from, so the augment.py / decompose.py / etc. invocations
# below resolve correctly without `python -m`.
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"
mkdir -p ./logs ./augmented_data ./decomposed_data

DATA_FILE="${DATA_FILE:-$REPO_ROOT/data/simplelr_qwen_level1to4/test_parsed.jsonl}"
PROMPT_IDS_FILE="${PROMPT_IDS_FILE:-$REPO_ROOT/unlearnable_prompt_ids/qwen_0.5b_math_level1to4_unlearnable.json}"
SETTING="${SETTING:-qwen_0.5b_math_level1to4}"
DATA_TYPE="${DATA_TYPE:-unlearnable}"
MODE="${MODE:-augmented}"
TOTAL_IDX="${TOTAL_IDX:-5}"

STEP1="${STEP1:-true}"
STEP2="${STEP2:-true}"
STEP3="${STEP3:-true}"
STEP4="${STEP4:-true}"
STEP5="${STEP5:-true}"

OUTPUT_NAME="${MODE}_data_${SETTING}_${DATA_TYPE}"

# ---- Step 1: OpenAI generation -------------------------------------------------
if [ "$STEP1" = "true" ]; then
    if [ "$MODE" = "augmented" ]; then SCRIPT=augment.py; else SCRIPT=decompose.py; fi
    for idx in $(seq 0 $((TOTAL_IDX - 1))); do
        python -u "$SCRIPT" \
            --data-file "$DATA_FILE" \
            --prompt-ids-file "$PROMPT_IDS_FILE" \
            --output-dir "./${MODE}_data" \
            --output-name "$OUTPUT_NAME" \
            --idx "$idx" \
            --total-batch "$TOTAL_IDX" \
            > "./logs/${MODE}_${SETTING}_${DATA_TYPE}_${idx}.log" 2>&1 &
    done
    wait
    echo "Step 1 (${MODE} generation) done"
fi

# ---- Step 2: Gemini cross-validation ------------------------------------------
if [ "$STEP2" = "true" ]; then
    python -u generate_answer_gemini.py \
        --setting "$SETTING" --data-type "$DATA_TYPE" --mode "$MODE" \
        > "./logs/gemini_${SETTING}_${DATA_TYPE}_${MODE}.log" 2>&1
    echo "Step 2 (Gemini answers) done"
fi

# ---- Step 3: Extract Gemini answers -------------------------------------------
if [ "$STEP3" = "true" ]; then
    for idx in $(seq 0 $((TOTAL_IDX - 1))); do
        python -u extract_answer.py \
            --setting "$SETTING" --data-type "$DATA_TYPE" --mode "$MODE" --idx "$idx" \
            > "./logs/extract_${SETTING}_${DATA_TYPE}_${MODE}_${idx}.log" 2>&1 &
    done
    wait
    echo "Step 3 (extract Gemini answers) done"
fi

# ---- Step 4: Filter ------------------------------------------------------------
if [ "$STEP4" = "true" ]; then
    python -u filter_data.py \
        --setting "$SETTING" --data-type "$DATA_TYPE" --mode "$MODE" \
        > "./logs/filter_${SETTING}_${DATA_TYPE}_${MODE}.log" 2>&1
    echo "Step 4 (filter) done"
fi

# ---- Step 5: Prepare gradient-similarity test_parsed.jsonl --------------------
if [ "$STEP5" = "true" ]; then
    AUG="./augmented_data/all_augmented_data_${SETTING}_${DATA_TYPE}_filtered.json"
    DEC="./decomposed_data/all_decomposed_data_${SETTING}_${DATA_TYPE}_filtered.json"
    OUT="${DATA_ROOT:-$REPO_ROOT/data}/augmented_data_for_grad_sim_${SETTING}_${DATA_TYPE}/test_parsed.jsonl"
    args=(--output-file "$OUT")
    [ -f "$AUG" ] && args+=(--augmented-file "$AUG")
    [ -f "$DEC" ] && args+=(--decomposed-file "$DEC")
    python -u prepare_grad_sim_data.py "${args[@]}"
    echo "Step 5 (prepare grad-sim data) done -> $OUT"
fi

echo "All requested steps completed"
