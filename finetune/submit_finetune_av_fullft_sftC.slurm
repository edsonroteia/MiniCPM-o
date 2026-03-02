#!/bin/bash
#SBATCH --job-name=minicpmo45-av-fullft-sftC
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

# SFT-C: Tune LLM + Vision + Audio (lr=1e-6)
# Rationale: unfreeze all encoders for co-adaptation of visual, auditory, and
# language representations. Compared to SFT-B (LLM+audio), tests whether
# vision tuning adds value.
export GPUS_PER_NODE=8
export NNODES=1
export DATA="sample_data/av_reasoning_full_video/train.json"
export EVAL_DATA="sample_data/av_reasoning_full_video/eval.json"
export OUTPUT_DIR="output/output_minicpmo45_av_fullft_reasoning_full_8gpu_1epoch_video_all"

# Full FT flags — tune everything
export USE_LORA=false
export TUNE_LLM=true
export TUNE_VISION=true
export TUNE_AUDIO=true
export INIT_AUDIO=true

# DeepSpeed ZeRO-2 for optimizer/gradient sharding
export DEEPSPEED_CONFIG=ds_config_zero2.json

# Hyperparams — same as baseline (lr=1e-6)
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export PER_DEVICE_EVAL_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=4
export MAX_STEPS=572
export EVAL_STEPS=100
export SAVE_STEPS=100
export SAVE_TOTAL_LIMIT=6
export LEARNING_RATE=1e-6
export WEIGHT_DECAY=0.1
export ADAM_BETA2=0.95
export WARMUP_RATIO=0.03
export LR_SCHEDULER_TYPE=cosine
export LOGGING_STEPS=10
export DATALOADER_NUM_WORKERS=4

# W&B logging
export REPORT_TO=wandb
export RUN_NAME="fullft-8f-1e6-llm-vision-audio-8gpu"
export WANDB_NUM_EVAL_EXAMPLES=2

bash finetune_av.sh
