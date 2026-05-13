#!/bin/bash
#
# GRPO baseline training entrypoint for Qwen2.5-0.5B on SimpleRL-Zoo MATH
# levels 1-4. Pass a seed as $1.
#
# Required environment variables:
#   HDFS_DATA_PATH   directory containing $DATASET_NAME/{train,test}.parquet
#   PROJECT_NAME     wandb project name
#   WORKING_DIR      directory uploaded as the Ray runtime working dir
#                    (typically the absolute path to this repo)
#
# Optional:
#   WANDB_API_KEY    if set, used by wandb (otherwise wandb runs offline)
#   MODELS_ROOT      defaults to ../models; expected to contain Qwen2.5-0.5B/
#   CHECKPOINTS_ROOT defaults to ./checkpoints
#   LOGS_ROOT        defaults to ./logs

set -euo pipefail

seed="${1:?Usage: $0 <seed>}"

HDFS_DATA_PATH="${HDFS_DATA_PATH:?HDFS_DATA_PATH must be set (directory containing <dataset>/{train,test}.parquet)}"
PROJECT_NAME="${PROJECT_NAME:?PROJECT_NAME must be set (wandb project)}"
WORKING_DIR="${WORKING_DIR:?WORKING_DIR must be set (absolute path uploaded as Ray runtime working dir)}"

export HDFS_DATA_PATH PROJECT_NAME WORKING_DIR

export RUN_NAME=qwen_0.5b_baseline
export DATASET_NAME=simplelr_qwen_level1to4
export VLLM_ATTENTION_BACKEND=XFORMERS

#===============================================
# Ray setup with cleanup on exit
#===============================================
cleanup_on_exit() {
    echo "Cleaning up Ray cluster..."
    ray stop --force || true
    pkill -f ray || true
    rm -rf /dev/shm/ray* /tmp/ray* || true
}
trap cleanup_on_exit EXIT INT TERM

echo "Starting Ray cluster..."
ray start --head --num-gpus 4 --num-cpus 64 --node-ip-address 0.0.0.0
export ARNOLD_WORKER_NUM=1   # number of nodes

batch_size=256

HYDRA_FULL_ERROR=1 bash train.sh \
    --model_name Qwen2.5-0.5B \
    --max_response_length 5120 \
    --train_batch_size "$batch_size" \
    --rollout_n 8 \
    --kl_loss_coef 0.0001 \
    --entropy_coefficient 0.001 \
    --rollout_gpu_memory_util 0.75 \
    --rollout_tp 2 \
    --save_freq 20 \
    --micro_rollout_batch_size 1024 \
    --total_epochs 50 \
    --rollout_name sglang \
    --learning_rate 5e-7 \
    --ppo_micro_batch_size 32 \
    --log_prob_micro_batch_size 128 \
    --n_gpu_per_node 4 \
    --resume_data_state True \
    --seed "$seed" \
    --ppo_mini_batch_size 64
