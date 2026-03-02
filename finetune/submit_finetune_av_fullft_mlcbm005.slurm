#!/bin/bash
#SBATCH --job-name=minicpmo45-av-fullft-5e6
#SBATCH --partition=h100-ferranti
#SBATCH --exclude=mlcbm012
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out

set -eo pipefail

# GCC 11 needed for DeepSpeed JIT compilation of FusedAdam
source scl_source enable gcc-toolset-11

set -u

cd /weka/kuehne/kqr867/code/MiniCPM-o/finetune

# Full fine-tuning: same config as the 8-frame LoRA run (job 325783)
# but with LoRA disabled, LLM tuning enabled, and lr=5e-6
export GPUS_PER_NODE=2
export NNODES=1
export DATA="sample_data/av_reasoning_full_video/train.json"
export EVAL_DATA="sample_data/av_reasoning_full_video/eval.json"
export OUTPUT_DIR="output/output_minicpmo45_av_fullft_reasoning_full_8gpu_1epoch_video_5e6"

# Full FT flags
export USE_LORA=false
export TUNE_LLM=true
export TUNE_VISION=false
export TUNE_AUDIO=false
export INIT_AUDIO=true

# DeepSpeed ZeRO-2 for optimizer/gradient sharding
export DEEPSPEED_CONFIG=ds_config_zero2.json

# Hyperparams matching the 8-frame LoRA run, except lr
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export PER_DEVICE_EVAL_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=16
export MAX_STEPS=572
export EVAL_STEPS=100
export SAVE_STEPS=100
export SAVE_TOTAL_LIMIT=6
export LEARNING_RATE=5e-6
export WEIGHT_DECAY=0.1
export ADAM_BETA2=0.95
export WARMUP_RATIO=0.03
export LR_SCHEDULER_TYPE=cosine
export LOGGING_STEPS=10
export DATALOADER_NUM_WORKERS=4

# W&B logging
export REPORT_TO=wandb
export RUN_NAME="fullft-8f-5e6-8gpu"
export WANDB_NUM_EVAL_EXAMPLES=2

bash finetune_av.sh
