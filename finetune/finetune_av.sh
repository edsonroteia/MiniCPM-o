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
OUTPUT_DIR=${OUTPUT_DIR:-output/output_minicpmo45_av_lora}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}
REPORT_TO=${REPORT_TO:-none}
RUN_NAME=${RUN_NAME:-}
WANDB_NUM_EVAL_EXAMPLES=${WANDB_NUM_EVAL_EXAMPLES:-2}

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
    --tune_vision false \
    --tune_llm false \
    --tune_audio false \
    --init_audio true \
    --use_lora true \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --max_slice_nums 1 \
    --max_steps 20 \
    --eval_steps 10 \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$OUTPUT_DIR" \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --wandb_num_eval_examples "$WANDB_NUM_EVAL_EXAMPLES" \
    "${EXTRA_ARGS[@]}" \
    --report_to "$REPORT_TO"
