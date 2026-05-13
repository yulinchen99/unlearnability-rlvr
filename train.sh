#! /bin/bash

# USER_ENV=`whoami`
# set -x
export NCCL_DEBUG=DEBUG
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEDUP_LOGS=1

export VLLM_ATTENTION_BACKEND=XFORMERS

# default models_root, checkpoints_root, logs_root
# read from env first, if not set, use default
MODELS_ROOT=${MODELS_ROOT:-../models}
CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT:-./checkpoints}
LOGS_ROOT=${LOGS_ROOT:-./logs}

# Default values
TRAIN_BATCH_SIZE=256
VAL_BATCH_SIZE=1000
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=5120
LEARNING_RATE=5e-7
PPO_MINI_BATCH_SIZE=64
# per GPU
PPO_MICRO_BATCH_SIZE=2
CLIP_RATIO=0.2
# KL_LOSS_COEF: GRPO actor-loss KL regularizer (-> actor_rollout_ref.actor.kl_loss_coef)
KL_LOSS_COEF=0.001
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
LOG_PROB_MICRO_BATCH_SIZE=8
ROLLOUT_N=8
# KL_COEF: reward-side KL controller coef, used only when algorithm.use_kl_in_reward=True
# (-> algorithm.kl_ctrl.kl_coef). Distinct from KL_LOSS_COEF above.
KL_COEF=0.001
TOTAL_STEPS=100
DATASET_NAME=$DATASET_NAME
ROLLOUT_GPU_MEMORY_UTIL=0.6
MODEL_NAME=Qwen2.5-0.5B
SAVE_FREQ=20
TEST_FREQ=5
REMOVE_CLIP=False
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2
MICRO_ROLLOUT_BATCH_SIZE=1024
REMOVE_PREVIOUS_CKPT=False
GRAD_CLIP=1.0
# ENTROPY_CTRL_TYPE selects the entropy-loss coefficient controller:
#   fixed           -> coef = ENTROPY_COEFFICIENT   (only knob that matters here)
#   instance_linear -> coef = max(ENTROPY_MIN_COEFFICIENT, ENTROPY_BETA * pass@1)
#   instance_power  -> coef = max(ENTROPY_MIN_COEFFICIENT, ENTROPY_BETA * pass@1**2)
# With the default 'fixed' controller, ENTROPY_MIN_COEFFICIENT and ENTROPY_BETA are inert.
ENTROPY_CTRL_TYPE="fixed"
ENTROPY_COEFFICIENT=0.001
ENTROPY_MIN_COEFFICIENT=0.001
ENTROPY_BETA=0.01
N_GPU_PER_NODE=2
TOTAL_EPOCHS=20
RESUME_DATA_STATE=False
ROLLOUT_NAME=sglang
ARNOLD_WORKER_NUM=1
SEED=1

# ===== for hypothesis testing =====

# oversample with replay
OVERSAMPLE_N=-1
REPLAY=False
REPLAY_N=1
KEEP_OLD_ADVANTAGE_BEFORE_DOWNSAMPLE=False

# clip-higher
LOW_CLIP_RATIO=0.2
HIGH_CLIP_RATIO=0.2

# kl ablation
USE_KL_LOSS=True


KEEP_ALL_NEGATIVE=False

generate_suffix() {
  local suffix=""
  local dataset_provided=false
  local model_provided=false
  local suffix_provided=false

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --train_batch_size) suffix+="_batch$2"; shift 2 ;;
      --val_batch_size) suffix+="_valbatch$2"; shift 2 ;;
      --max_prompt_length) suffix+="_max_prompt$2"; shift 2 ;;
      --max_response_length) suffix+="_max_response$2"; shift 2 ;;
      --learning_rate) suffix+="_lr$2"; shift 2 ;;
      --ppo_mini_batch_size) suffix+="_ppomini$2"; shift 2 ;;
      --ppo_micro_batch_size) shift 2 ;;
      --kl_loss_coef) suffix+="_klcoef$2"; shift 2 ;;
      --entropy_coefficient) suffix+="_entcoef$2"; shift 2 ;;
      --entropy_min_coefficient) suffix+="_entmincoef$2"; shift 2 ;;
      --entropy_beta) suffix+="_entbeta$2"; shift 2 ;;
      --entropy_ctrl_type) suffix+="_entctrltype$2"; shift 2 ;;
      --clip_ratio) suffix+="_clipratio$2"; shift 2 ;;
      --kl_loss_type) suffix+="_kltype$2"; shift 2 ;;
      --temperature) suffix+="_temp$2"; shift 2 ;;
      --log_prob_micro_batch_size) suffix+="_logprobbatch256"; shift 2 ;;
      --rollout_n) suffix+="_rollout$2"; shift 2 ;;
      --kl_coef) suffix+="_klcontrol$2"; shift 2 ;;
      --total_training_steps) suffix+="_steps$2"; shift 2 ;;
      --rollout_gpu_memory_util) shift 2 ;;
      --dataset_name) suffix+="_$2"; dataset_provided=true; shift 2 ;;
      --model_name) suffix+="_$2"; model_provided=true; shift 2 ;;
      --remove_clip) suffix+="_remove_clip$2"; shift 2 ;;
      --suffix) input_suffix="$2"; suffix_provided=true; shift 2 ;;
      --no_kl_loss) suffix+="_nokl"; shift 1 ;;
      --grad_clip) suffix+="_gradclip$2"; shift 2 ;;
      --low_clip_ratio) suffix+="_lowclip$2"; shift 2 ;;
      --high_clip_ratio) suffix+="_highclip$2"; shift 2 ;;
      --seed) suffix+="_seed$2"; shift 2 ;;
      --oversample_n) suffix+="_oversample$2"; shift 2 ;;
      --replay) suffix+="_replay"; shift 1 ;;
      --replay_n) suffix+="_n$2"; shift 2 ;;
      --keep_old_advantage_before_downsample) suffix+="_keepoldadvantage"; shift 1 ;;
      --keep_all_negative) suffix+="_keepallnegative"; shift 1 ;;
      *) shift ;;
    esac
  done

  if [ "$dataset_provided" = false ]; then
    suffix+="_$DATASET_NAME"
  fi

  if [ "$model_provided" = false ]; then
    suffix+="_$MODEL_NAME"
  fi

  if [ "$suffix_provided" = true ]; then
    suffix+="_$input_suffix"
  fi
  
  echo "$suffix"
}

echo "Arguments received: $@"

# Generate a unique suffix based on the input arguments
SUFFIX=$(generate_suffix "$@")
RUN_NAME="$RUN_NAME$SUFFIX"
# replace the / with _
RUN_NAME=${RUN_NAME//\//_}
LOG_FILE_PATH="./logs/$RUN_NAME.log"
mkdir -p ./logs

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  echo "Processing: $1"
  case "$1" in
    --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --val_batch_size) VAL_BATCH_SIZE="$2"; shift 2 ;;
    --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
    --ppo_micro_batch_size) PPO_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
    --entropy_coefficient) ENTROPY_COEFFICIENT="$2"; shift 2 ;;
    --entropy_min_coefficient) ENTROPY_MIN_COEFFICIENT="$2"; shift 2 ;;
    --entropy_beta) ENTROPY_BETA="$2"; shift 2 ;;
    --entropy_ctrl_type) ENTROPY_CTRL_TYPE="$2"; shift 2 ;;
    --clip_ratio) CLIP_RATIO="$2"; shift 2 ;;
    --kl_loss_type) KL_LOSS_TYPE="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --log_prob_micro_batch_size) LOG_PROB_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
    --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
    --rollout_tp) ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="$2"; shift 2 ;;
    --micro_rollout_batch_size) MICRO_ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
    --kl_coef) KL_COEF="$2"; shift 2 ;;
    --total_training_steps) TOTAL_STEPS="$2"; shift 2 ;;
    --dataset_name) DATASET_NAME="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --save_freq) SAVE_FREQ="$2"; shift 2 ;;
    --test_freq) TEST_FREQ="$2"; shift 2 ;;
    --remove_clip) REMOVE_CLIP="$2"; shift 2 ;;
    --remove_previous_ckpt) REMOVE_PREVIOUS_CKPT="$2"; shift 2 ;;
    --suffix) SUFFIX="$2"; shift 2 ;;
    --n_gpu_per_node) N_GPU_PER_NODE="$2"; shift 2 ;;
    --resume_data_state) RESUME_DATA_STATE="$2"; shift 2 ;;
    --grad_clip) GRAD_CLIP="$2"; shift 2 ;;
    --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
    --rollout_name) ROLLOUT_NAME="$2"; shift 2 ;;
    --no_kl_loss) USE_KL_LOSS=False; shift 1 ;;
    --low_clip_ratio) LOW_CLIP_RATIO="$2"; shift 2 ;;
    --high_clip_ratio) HIGH_CLIP_RATIO="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --oversample_n) OVERSAMPLE_N="$2"; shift 2 ;;
    --replay) REPLAY=True; shift 1 ;;
    --replay_n) REPLAY_N="$2"; shift 2 ;;
    --keep_old_advantage_before_downsample) KEEP_OLD_ADVANTAGE_BEFORE_DOWNSAMPLE=True; shift 1 ;;
    --keep_all_negative) KEEP_ALL_NEGATIVE=True; shift 1 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Training with the following parameters:"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Val Batch Size: $VAL_BATCH_SIZE" 
echo "Max Prompt Length: $MAX_PROMPT_LENGTH" 
echo "Max Response Length: $MAX_RESPONSE_LENGTH" 
echo "Learning Rate: $LEARNING_RATE" 
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE" 
echo "PPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE" 
echo "Micro Rollout Batch Size: $MICRO_ROLLOUT_BATCH_SIZE"
echo "KL Loss Coefficient: $KL_LOSS_COEF" 
echo "KL Loss Type: $KL_LOSS_TYPE" 
echo "Temperature: $TEMPERATURE" 
echo "Rollout N: $ROLLOUT_N" 
echo "KL Coefficient: $KL_COEF" 
echo "Total Training Steps: $TOTAL_STEPS"
echo "Dataset Name: $DATASET_NAME"
echo "Model Name: $MODEL_NAME"
echo "Remove Clip: $REMOVE_CLIP"
echo "Remove Previous Ckpt: $REMOVE_PREVIOUS_CKPT"
echo "LOG FILE PATH: $LOG_FILE_PATH"
echo "Entropy Coefficient: $ENTROPY_COEFFICIENT"
echo "Entropy Min Coefficient: $ENTROPY_MIN_COEFFICIENT"
echo "Entropy Beta: $ENTROPY_BETA"
echo "Entropy Control Type: $ENTROPY_CTRL_TYPE"
echo "Resume Data State: $RESUME_DATA_STATE"
echo "Grad Clip: $GRAD_CLIP"
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Use KL Loss: $USE_KL_LOSS"
echo "Low Clip Ratio: $LOW_CLIP_RATIO"
echo "High Clip Ratio: $HIGH_CLIP_RATIO"
echo "Seed: $SEED"
echo "Oversample N: $OVERSAMPLE_N"
echo "Replay: $REPLAY"
echo "Replay N: $REPLAY_N"
echo "Keep Old Advantage Before Downsample: $KEEP_OLD_ADVANTAGE_BEFORE_DOWNSAMPLE"
echo "Keep All Negative: $KEEP_ALL_NEGATIVE"
max_num_batched_tokens=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH + 1000)
echo -e "Training with the following parameters:\nTrain Batch Size: $TRAIN_BATCH_SIZE\nVal Batch Size: $VAL_BATCH_SIZE\nMax Prompt Length: $MAX_PROMPT_LENGTH\nMax Response Length: $MAX_RESPONSE_LENGTH\nLearning Rate: $LEARNING_RATE\nPPO Mini Batch Size: $PPO_MINI_BATCH_SIZE\nPPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE\nKL Loss Coefficient: $KL_LOSS_COEF\nKL Loss Type: $KL_LOSS_TYPE\nTemperature: $TEMPERATURE\nRollout N: $ROLLOUT_N\nKL Coefficient: $KL_COEF\nTotal Training Steps: $TOTAL_STEPS\nDataset Name: $DATASET_NAME\nModel Name: $MODEL_NAME\nEntropy Coefficient: $ENTROPY_COEFFICIENT\nEntropy Min Coefficient: $ENTROPY_MIN_COEFFICIENT\nEntropy Beta: $ENTROPY_BETA\nEntropy Control Type: $ENTROPY_CTRL_TYPE\nGrad Clip: $GRAD_CLIP\nLow Clip Ratio: $LOW_CLIP_RATIO\nHigh Clip Ratio: $HIGH_CLIP_RATIO\nKeep All Negative: $KEEP_ALL_NEGATIVE"


echo "Running Python script..."
ray job submit --address=localhost:6379 \
  --entrypoint-num-cpus=1 \
  --runtime-env-json='{
        "working_dir": "'${WORKING_DIR}'",
        "excludes": [
          ".git/objects/pack/pack-78430d62e131e6388668e283065df0b5c54aaaa5.pack",
          ".git/**"
        ],
        "env_vars": {
          "http_proxy": "",
          "https_proxy": ""
        }
    }' \
  -- python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.en_ctrl.type=$ENTROPY_CTRL_TYPE \
  actor_rollout_ref.actor.en_ctrl.en_coef=$ENTROPY_COEFFICIENT \
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFICIENT \
  actor_rollout_ref.actor.en_ctrl.min_en_coef=$ENTROPY_MIN_COEFFICIENT \
  actor_rollout_ref.actor.en_ctrl.beta=$ENTROPY_BETA \
  data.seed=$SEED \
  data.train_files=$HDFS_DATA_PATH/$DATASET_NAME/train.parquet \
  data.val_files=$HDFS_DATA_PATH/$DATASET_NAME/test.parquet \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=$VAL_BATCH_SIZE \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.replay=$REPLAY \
  data.replay_n=$REPLAY_N \
  data.keep_old_advantage_before_downsample=$KEEP_OLD_ADVANTAGE_BEFORE_DOWNSAMPLE \
  data.keep_all_negative=$KEEP_ALL_NEGATIVE \
  actor_rollout_ref.model.path=$MODELS_ROOT/$MODEL_NAME \
  actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
  actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
  actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
  actor_rollout_ref.actor.low_clip_ratio=$LOW_CLIP_RATIO \
  actor_rollout_ref.actor.high_clip_ratio=$HIGH_CLIP_RATIO \
  actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
  actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.temperature=$TEMPERATURE \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
  actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.oversample_n=$OVERSAMPLE_N \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
  actor_rollout_ref.rollout.micro_rollout_batch_size=$MICRO_ROLLOUT_BATCH_SIZE \
  actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=$KL_COEF \
  critic.ppo_micro_batch_size_per_gpu=4 \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$PROJECT_NAME \
  trainer.remove_previous_ckpt=$REMOVE_PREVIOUS_CKPT \
  trainer.experiment_name=$RUN_NAME \
  trainer.n_gpus_per_node=$N_GPU_PER_NODE \
  trainer.nnodes=$ARNOLD_WORKER_NUM \
  trainer.remove_clip=$REMOVE_CLIP \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.default_local_dir=$CHECKPOINTS_ROOT/$RUN_NAME \
  trainer.resume_data_state=$RESUME_DATA_STATE \
  trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee -a $LOG_FILE_PATH
