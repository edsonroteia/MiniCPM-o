#!/bin/bash

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/weka/kuehne/kqr867/miniconda3/envs/minicpm/bin/python}
TORCHRUN_BIN=${TORCHRUN_BIN:-/weka/kuehne/kqr867/miniconda3/envs/minicpm/bin/torchrun}
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6003}

MODEL=${MODEL:-openbmb/MiniCPM-o-4_5}
DATA=${DATA:-sample_data/av_smoke/train.json}
EVAL_DATA=${EVAL_DATA:-sample_data/av_smoke/eval.json}
LLM_TYPE=${LLM_TYPE:-qwen}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
MAX_SLICE_NUMS=${MAX_SLICE_NUMS:-1}
OUTPUT_DIR=${OUTPUT_DIR:-output/output_minicpmo45_av_lora}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}
REPORT_TO=${REPORT_TO:-none}
RUN_NAME=${RUN_NAME:-}
WANDB_NUM_EVAL_EXAMPLES=${WANDB_NUM_EVAL_EXAMPLES:-2}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
MAX_STEPS=${MAX_STEPS:-20}
EVAL_STEPS=${EVAL_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-10}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
ADAM_BETA2=${ADAM_BETA2:-0.95}
WARMUP_RATIO=${WARMUP_RATIO:-0.01}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
LOGGING_STEPS=${LOGGING_STEPS:-1}
TUNE_VISION=${TUNE_VISION:-false}
TUNE_LLM=${TUNE_LLM:-false}
TUNE_AUDIO=${TUNE_AUDIO:-false}
INIT_AUDIO=${INIT_AUDIO:-true}
USE_LORA=${USE_LORA:-true}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

EXTRA_ARGS=()
if [ -n "$DEEPSPEED_CONFIG" ]; then
    EXTRA_ARGS+=(--deepspeed "$DEEPSPEED_CONFIG")
fi
if [ -n "$RUN_NAME" ]; then
    EXTRA_ARGS+=(--run_name "$RUN_NAME")
fi

$TORCHRUN_BIN $DISTRIBUTED_ARGS finetune_av.py \
    --model_name_or_path "$MODEL" \
    --llm_type "$LLM_TYPE" \
    --data_path "$DATA" \
    --eval_data_path "$EVAL_DATA" \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --do_eval \
    --tune_vision "$TUNE_VISION" \
    --tune_llm "$TUNE_LLM" \
    --tune_audio "$TUNE_AUDIO" \
    --init_audio "$INIT_AUDIO" \
    --use_lora "$USE_LORA" \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --max_slice_nums "$MAX_SLICE_NUMS" \
    --max_steps "$MAX_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$OUTPUT_DIR" \
    --logging_strategy "steps" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --eval_strategy "steps" \
    --save_strategy "steps" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --adam_beta2 "$ADAM_BETA2" \
    --warmup_ratio "$WARMUP_RATIO" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --logging_steps "$LOGGING_STEPS" \
    --gradient_checkpointing true \
    --wandb_num_eval_examples "$WANDB_NUM_EVAL_EXAMPLES" \
    "${EXTRA_ARGS[@]}" \
    --report_to "$REPORT_TO"
