#!/bin/bash
#SBATCH --job-name=minicpmo45-grpo-fullft-sftD
#SBATCH --partition=h100-ferranti
#SBATCH --reservation=kqr867_3
#SBATCH --nodelist=mlcbm004
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out

set -eo pipefail

# GCC 11 needed for DeepSpeed JIT compilation of FusedAdam
source scl_source enable gcc-toolset-11

set -u

cd /weka/kuehne/kqr867/code/MiniCPM-o/finetune

# GRPO full fine-tuning on SFT-D (text answers, lr=1e-6)
SFT_OUTPUT_DIR="output/output_minicpmo45_av_fullft_reasoning_full_8gpu_1epoch_video_textanswer"
SFT_CHECKPOINT=$(ls -d "${SFT_OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$SFT_CHECKPOINT" ]; then
    echo "ERROR: No SFT checkpoint found in ${SFT_OUTPUT_DIR}" >&2
    exit 1
fi
echo "Using SFT checkpoint: ${SFT_CHECKPOINT}"

export GPUS_PER_NODE=8
export NNODES=1

# Full-FT GRPO: no LoRA
export USE_LORA_FOR_RL=false
export MERGE_SFT_ADAPTER=false
export SFT_ADAPTER="${SFT_CHECKPOINT}"
export DATA=/weka/kuehne/kqr867/datasets/AVQA/train_qa_cleaned.data
export OUTPUT_DIR="output/output_minicpmo45_av_grpo_fullft_sftD"

# DeepSpeed ZeRO-2
export DEEPSPEED_CONFIG=ds_config_zero2.json

# GRPO hyperparameters (same as baseline GRPO)
export LEARNING_RATE=5e-7
export BETA=0.0
export REWARD_FUNCTIONS="accuracy format"
export MAX_COMPLETION_LENGTH=384
export NUM_GENERATIONS=4
export LOSS_TYPE=dapo
export MAX_SAMPLES=0
export MAX_STEPS=200
export SAVE_STEPS=50
export SAVE_TOTAL_LIMIT=5
export LOGGING_STEPS=1
export TEMPERATURE=1.0
export GRADIENT_ACCUMULATION_STEPS=4
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export PER_DEVICE_EVAL_BATCH_SIZE=4
export STEPS_PER_GENERATION=4
export WARMUP_RATIO=0.03
export LR_SCHEDULER_TYPE=cosine
export DATALOADER_NUM_WORKERS=0

# W&B logging
export REPORT_TO=wandb
export RUN_NAME="grpo-fullft-sftD-8gpu"
export LOG_COMPLETIONS=true
export WANDB_NUM_LOGGED_COMPLETIONS=8

bash finetune_grpo_av.sh
