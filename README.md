# The Unlearnability Phenomenon in RLVR for Language Models

Code release for the ICML 2026 paper: "The Unlearnability Phenomenon in RLVR for Language Models"

> **TL;DR** A substantial fraction of "hard" prompts in RLVR receive correct rollouts during GRPO training yet show no improvement in pass rate. We call these *unlearnable* examples and show is't likely fundamental representation issue with gradient analysis. 

The walkthrough below reproduces the **Qwen2.5-0.5B + MATH levels 1–4** configuration.

## Hardware & dependencies

- **GPUs:** ≥2 GPUs for training (the default driver allocates `--num-gpus 4`),
  ≥1 for inference / gradient analysis.
- **Python env:** `pip install -r requirements.txt`

## Environment variables

| Variable | Used by | Default | Notes |
|---|---|---|---|
| `MODELS_ROOT` | training, inference | `../models` | base HF models live here |
| `DATA_ROOT` | inference, data-aug | `./data` | parquet/JSONL datasets |
| `CHECKPOINTS_ROOT` | training, inference | `./checkpoints` | RL training output |
| `RESULTS_DIR` | inference | `./results` | per-question rollout JSONs |
| `GRAD_ANALYSIS_OUTPUT_DIR` | gradient analysis | `./grpo_gradient_analysis_output` | per-prompt gradient `.npz` |
| `HDFS_DATA_PATH` | training | — (required) | parquet `<dataset>/{train,test}.parquet` lives here |
| `PROJECT_NAME` | training | — (required) | wandb project name |
| `WORKING_DIR` | training | — (required) | absolute path uploaded as Ray runtime working dir |
| `WANDB_API_KEY` | training | optional | otherwise wandb runs offline |
| `OPENAI_API_KEY` | data augmentation | — (required for steps 1, 3) | |
| `GEMINI_API_KEY` | data augmentation | — (required for step 2) | |

## Models and Data

- **Base model:** [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B)
  → place at `${MODELS_ROOT}/Qwen2.5-0.5B/`.
- **Training data:** [`hkust-nlp/SimpleRL-Zoo-Data`](https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data).
  → place at `${DATA_ROOT}/SimpleRL-Zoo-Data/`


## Layout

```
├── train.sh / train_baseline.sh       GRPO training entry points
├── verl/                              vendored RL framework (Apache-2.0)
├── inference/                         pass@k evaluation
├── classification/                    unlearnability partition
├── data_augmentation/                 similar-problem + sub-problem generation
├── gradient_analysis/                 per-prompt GRPO gradients + cosine similarity
├── data/                              shipped paper artifacts (filtered augmentations)
└── common/                            shared utilities (math verifier, OpenAI client)
```

## How to Run

### 1. (Optional) Identify Unlearnable Examples

**(1) Running your own GRPO training**
Train for 3 independent runs with different random seeds, evaluate each (plus the
initial model), aggregate, then run the same `finalize_*` command:

```bash
# (a) Train.
export HDFS_DATA_PATH=/abs/path/to/parquet/root # where train.parquet and test.parquet locate
export PROJECT_NAME=unlearnability_rlvr
export WORKING_DIR=$(pwd)
for seed in 1 42 37; do bash train_baseline.sh "$seed"; done

# (b) Evaluate each checkpoint at the chosen step (e.g. step 120).
#     Pass --model-name so the result filename starts with the model-name
#     you'll feed into the aggregator below.
bash inference/test.sh /abs/path/to/qwen_0.5b_baseline_seed1/global_step_120/actor/huggingface \
    --num-samples 32 --test-data simplelr_qwen_level1to4@test \
    --model-name qwen_0.5b_run1
bash inference/test.sh /abs/path/to/qwen_0.5b_baseline_seed42/global_step_120/actor/huggingface \
    --num-samples 32 --test-data simplelr_qwen_level1to4@test \
    --model-name qwen_0.5b_run2
bash inference/test.sh /abs/path/to/qwen_0.5b_baseline_seed37/global_step_120/actor/huggingface \
    --num-samples 32 --test-data simplelr_qwen_level1to4@test \
    --model-name qwen_0.5b_run3
bash inference/test.sh Qwen/Qwen2.5-0.5B \
    --num-samples 32 --test-data simplelr_qwen_level1to4@test \
    --model-name qwen_0.5b_initial
```

**(2) Get unlearnable examples**

Given each training run's prompt reward stats during training (see [`./example_data/prompt_reward_stats.csv`](./example_data/prompt_reward_stats.csv) as a real example for Qwen0.5B + simplelr_qwen_level1to4), test results and the initial model's test results,
this step partitions all evaluated prompts into four groups at threshold
`--threshold` (= τ):

- `{output_prefix}_unlearnable.json` — pass@1 ≤ τ in every run (intersection), with verifier-failure prompts removed.
- `{output_prefix}_learnable.json`   — initial pass@1 ≤ τ and final pass@1 > τ in **every** run (intersection of per-run learnable sets).
- `{output_prefix}_no_reward.json`   — union of prompts that received zero positive reward in any run (treated as verifier failures, not learning failures).
- `{output_prefix}_easy.json`        — initial pass@1 > τ (not used downstream by the unlearnability analysis, but useful as a control group for gradient similarity).

```bash
python classification/finalize_unlearnable_example_id.py \
    --run          qwen_0.5b_run1=results/qwen_0.5b_run1_simplelr_qwen_level1to4*.json \
    --run          qwen_0.5b_run2=results/qwen_0.5b_run2_simplelr_qwen_level1to4*.json \
    --run          qwen_0.5b_run3=results/qwen_0.5b_run3_simplelr_qwen_level1to4*.json \
    --reward-stats qwen_0.5b_run1=checkpoints/qwen_0.5b_run1/prompt_reward_stats.csv \
    --reward-stats qwen_0.5b_run2=checkpoints/qwen_0.5b_run2/prompt_reward_stats.csv \
    --reward-stats qwen_0.5b_run3=checkpoints/qwen_0.5b_run3/prompt_reward_stats.csv \
    --initial-run  qwen_0.5b_initial=results/qwen_0.5b_initial_simplelr_qwen_level1to4*.json \
    --threshold    0.1 \
    --output-dir   ./example_ids \
    --output-prefix qwen_0.5b_math_level1to4
```

Writes `example_ids/qwen_0.5b_math_level1to4_{unlearnable,learnable,no_reward,easy}.json`.

### 2. Gradient Analysis

After classification, sample 100 prompts from each of `easy`, `learnable`,
and `unlearnable`, then compute pairwise GRPO-gradient cosine similarity at
the initial checkpoint.
If you skip previous step, you can simply use data in [`example_data/example_ids`](./example_data/example_ids)

**(1) Sample prompts and generate rollouts**

```bash
# (a) Sample 100 prompt-ids from each group and write a filtered test JSONL.
#     Prereq: inference/test.sh has run on simplelr_qwen_level1to4@test at
#     least once so ./data/.../test_parsed.jsonl exists.
# Replace `example_ids` below with `./example_data/example_ids` to use the shipped data directly.
python gradient_analysis/sample_grad_sim_prompts.py \
    --example-ids-dir    example_ids \
    --example-ids-prefix qwen_0.5b_math_level1to4 \
    --groups             easy learnable unlearnable \
    --n-per-group        100 \
    --source-test-jsonl  ./data/simplelr_qwen_level1to4/test_parsed.jsonl \
    --output-jsonl       ./data/grad_sim/test.jsonl

# (b) Generate ~1k rollouts per sampled prompt at the initial checkpoint.
# (inference/test.sh hardcodes temperature=0.7; call evaluate_model.py
# directly if you need a different temperature.)
bash inference/test.sh Qwen/Qwen2.5-0.5B \
    --test-data ./data/grad_sim/test.jsonl \
    --num-samples 1000 \
    --model-name qwen_0.5b_initial_grad_sim
```

**(2) Per-prompt gradients and pairwise similarity**

```bash
# (a) Per-prompt GRPO gradients on correct rollouts only (LoRA keeps the
#     .npz files manageable). Rollouts are already restricted to the
#     sampled prompts via the filtered test JSONL above.
ROLLOUTS_FILE='./results/qwen_0.5b_initial_grad_sim_*.json' \
MODEL_PATH=Qwen/Qwen2.5-0.5B \
OUTPUT_DIR=./grpo_gradient_analysis_output/qwen_0.5b_demo \
bash gradient_analysis/run_gradient.sh 0 300 --pos_only --lora

# (b) Pairwise cosine similarity across the per-prompt gradients.
bash gradient_analysis/cal_similarity.sh qwen_0.5b_demo 0 100000

# (c) Block-ordered heatmap (groups: easy / learnable / unlearnable).
python gradient_analysis/plot_grad_sim_heatmap.py \
    --similarity-files   './grpo_gradient_analysis_output/qwen_0.5b_demo/qid_cosine_similarity_pair_dict_pos_*.json' \
    --example-ids-dir    example_ids \
    --example-ids-prefix qwen_0.5b_math_level1to4 \
    --groups             easy learnable unlearnable \
    --output             figures/grad_sim_heatmap.png
```

Writes `${OUTPUT_DIR}/average_gradients_prompt_correct_<qid>.npz` per
sampled prompt, `qid_cosine_similarity_pair_dict_*.json` containing the
pairwise similarities, and `figures/grad_sim_heatmap.png` (the block-
structured heatmap from the paper).

### 3. Data Augmentation

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...

# Augmentation: similar problems for unlearnable prompts.
# Defaults point at ./data/... and ./unlearnable_prompt_ids/...
# (relative to the repo root); override DATA_FILE / PROMPT_IDS_FILE if needed.
SETTING=qwen_0.5b_math_level1to4 \
DATA_TYPE=unlearnable \
MODE=augmented \
bash data_augmentation/pipeline.sh

# MODE
# augmented: similar problems
# decomposition: same prompts, sub-problems instead of similar problems.
MODE=decomposed bash data_augmentation/pipeline.sh
```

Each invocation runs five stages (toggle individually with `STEP1=true|false …`):
generate via OpenAI → cross-validate via Gemini → extract Gemini's
boxed answer → keep only items the two models agree on → assemble a
JSONL ready for rollouts at
`${DATA_ROOT}/augmented_data_for_grad_sim_<setting>_<data_type>/test_parsed.jsonl`.

We include the generated augment data used in our experiment in `./example_data`.


### 4. Hypothesis-elimination Experiments

Section 4 in the paper.

**Oversample and replay positive rollouts**

```bash
bash train.sh --model_name Qwen2.5-0.5B --max_response_length 5120 \
    --train_batch_size 256 --rollout_n 8 --learning_rate 5e-7 \
    --kl_loss_coef 0.0001 --entropy_coefficient 0.001 \
    --rollout_tp 2 --n_gpu_per_node 4 --seed 1 \
    --replay --replay_n 1 --oversample_n 64
```

**Clip-higher**

```bash
bash train.sh --model_name Qwen2.5-0.5B --max_response_length 5120 \
    --train_batch_size 256 --rollout_n 8 --learning_rate 5e-7 \
    --entropy_coefficient 0.001 \
    --rollout_tp 2 --n_gpu_per_node 4 --seed 1 \
    --low_clip_ratio 0.2 --high_clip_ratio 0.28
```

**KL removed**

```bash
bash train.sh --model_name Qwen2.5-0.5B --max_response_length 5120 \
    --train_batch_size 256 --rollout_n 8 --learning_rate 5e-7 \
    --entropy_coefficient 0.001 \
    --rollout_tp 2 --n_gpu_per_node 4 --seed 1 \
    --no_kl_loss
```



## Other configurations

The same scripts cover the other two paper configurations; only the model,
training data, and learning rate change.

| Config | Base model | Training data | LR | Max resp. |
|---|---|---|---|---|
| Qwen 0.5B (walkthrough) | `Qwen/Qwen2.5-0.5B` | `simplelr_qwen_level1to4` | 5e-7 | 5120 |
| Llama 3.2 3B Instruct | `meta-llama/Llama-3.2-3B-Instruct` | `simplelr_qwen_level3to5_strip_template` | 5e-7 | 5120 |
| Qwen 2.5 3B | `Qwen/Qwen2.5-3B` | `deepscaler` | 1e-6 | 8192 |

> Note: For Llama-instruct, strip the qwen template in the original data and the training script automatically wraps with llama template. For inference, pass `--prompt-type llama` to `inference/test.sh` for the Llama run. 



## Citation

