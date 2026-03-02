#!/bin/bash

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/weka/kuehne/kqr867/miniconda3/envs/minicpm/bin/python}
TORCHRUN_BIN=${TORCHRUN_BIN:-/weka/kuehne/kqr867/miniconda3/envs/minicpm/bin/torchrun}
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6011}

MODEL=${MODEL:-openbmb/MiniCPM-o-4_5}
SFT_ADAPTER=${SFT_ADAPTER:-/weka/kuehne/kqr867/code/MiniCPM-o/finetune/output/output_minicpmo45_av_lora_reasoning_full_8gpu_1epoch_video}
DATA=${DATA:-/weka/kuehne/kqr867/datasets/AVQA/train_qa_cleaned.data}
EVAL_DATA=${EVAL_DATA:-}
OUTPUT_DIR=${OUTPUT_DIR:-output/output_minicpmo45_av_grpo_smoke}
REPORT_TO=${REPORT_TO:-none}
RUN_NAME=${RUN_NAME:-}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}
WANDB_PROJECT=${WANDB_PROJECT:-minicpm-av}
WANDB_ENTITY=${WANDB_ENTITY:-}

NUM_VIDEO_FRAMES=${NUM_VIDEO_FRAMES:-8}
MAX_SLICE_NUMS=${MAX_SLICE_NUMS:-1}
MAX_SAMPLES=${MAX_SAMPLES:-20}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-}
REWARD_FUNCTIONS=${REWARD_FUNCTIONS:-"accuracy format reasoning_length"}
LOSS_TYPE=${LOSS_TYPE:-dapo}
SCALE_REWARDS=${SCALE_REWARDS:-}
LOG_COMPLETIONS=${LOG_COMPLETIONS:-false}
WANDB_NUM_LOGGED_COMPLETIONS=${WANDB_NUM_LOGGED_COMPLETIONS:-8}

PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
STEPS_PER_GENERATION=${STEPS_PER_GENERATION:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-96}
MAX_STEPS=${MAX_STEPS:-2}
SAVE_STEPS=${SAVE_STEPS:-2}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
LOGGING_STEPS=${LOGGING_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
BETA=${BETA:-0.04}
TEMPERATURE=${TEMPERATURE:-1.0}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
USE_LORA_FOR_RL=${USE_LORA_FOR_RL:-true}
MERGE_SFT_ADAPTER=${MERGE_SFT_ADAPTER:-true}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

EXTRA_ARGS=()
if [ -n "$RUN_NAME" ]; then
    EXTRA_ARGS+=(--run_name "$RUN_NAME")
fi
if [ -n "$DEEPSPEED_CONFIG" ]; then
    EXTRA_ARGS+=(--deepspeed "$DEEPSPEED_CONFIG")
fi
if [ -n "$EVAL_DATA" ]; then
    EXTRA_ARGS+=(--eval_data_path "$EVAL_DATA" --do_eval --eval_strategy steps --eval_steps "$SAVE_STEPS")
fi
if [ -n "$MAX_EVAL_SAMPLES" ]; then
    EXTRA_ARGS+=(--max_eval_samples "$MAX_EVAL_SAMPLES")
fi
if [ -n "$SCALE_REWARDS" ]; then
    EXTRA_ARGS+=(--scale_rewards "$SCALE_REWARDS")
fi

if [ "$REPORT_TO" = "wandb" ]; then
    export WANDB_PROJECT
    if [ -n "$WANDB_ENTITY" ]; then
        export WANDB_ENTITY
    fi
fi

$TORCHRUN_BIN $DISTRIBUTED_ARGS finetune_grpo_av.py \
    --base_model_name_or_path "$MODEL" \
    --sft_adapter_path "$SFT_ADAPTER" \
    --use_lora_for_rl "$USE_LORA_FOR_RL" \
    --merge_sft_adapter "$MERGE_SFT_ADAPTER" \
    --data_path "$DATA" \
    --max_samples "$MAX_SAMPLES" \
    --num_video_frames "$NUM_VIDEO_FRAMES" \
    --max_slice_nums "$MAX_SLICE_NUMS" \
    --reward_functions "$REWARD_FUNCTIONS" \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$OUTPUT_DIR" \
    --report_to "$REPORT_TO" \
    --log_completions "$LOG_COMPLETIONS" \
    --wandb_num_logged_completions "$WANDB_NUM_LOGGED_COMPLETIONS" \
    --bf16 true \
    --fp16 false \
    --do_train \
    --save_only_model true \
    --remove_unused_columns false \
    --loss_type "$LOSS_TYPE" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --num_generations "$NUM_GENERATIONS" \
    --steps_per_generation "$STEPS_PER_GENERATION" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --max_steps "$MAX_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --logging_steps "$LOGGING_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --beta "$BETA" \
    --temperature "$TEMPERATURE" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    "${EXTRA_ARGS[@]}"
