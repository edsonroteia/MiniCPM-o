#!/bin/bash
# Master pipeline script: chains SFT experiments + GRPO full-FT on mlcbm004
# All jobs use --dependency=afterany so the chain continues even if one SFT fails.
#
# Timeline (approximate):
#   Hour 0-1: SFT baseline (job 326475, already running)
#   Hour 1-2: SFT-A (lr=1e-5)
#   Hour 2-3: SFT-B (LLM + audio)
#   Hour 3-4: SFT-C (LLM + vision + audio)
#   Hour 4-9: GRPO full-FT (on baseline lr=1e-6 checkpoint)

set -euo pipefail

cd /weka/kuehne/kqr867/code/MiniCPM-o/finetune

BASELINE_JOB=${1:-326475}

echo "Chaining pipeline after baseline job ${BASELINE_JOB}"

# SFT-A: lr=1e-5
JOB_A=$(sbatch --dependency=afterany:${BASELINE_JOB} submit_finetune_av_fullft_sftA.slurm | awk '{print $4}')
echo "SFT-A submitted: job ${JOB_A} (lr=1e-5)"

# SFT-B: LLM + audio
JOB_B=$(sbatch --dependency=afterany:${JOB_A} submit_finetune_av_fullft_sftB.slurm | awk '{print $4}')
echo "SFT-B submitted: job ${JOB_B} (LLM+audio)"

# SFT-C: LLM + vision + audio
JOB_C=$(sbatch --dependency=afterany:${JOB_B} submit_finetune_av_fullft_sftC.slurm | awk '{print $4}')
echo "SFT-C submitted: job ${JOB_C} (LLM+vision+audio)"

# SFT-D: text answers (50% letter -> choice text)
JOB_D=$(sbatch --dependency=afterany:${JOB_C} submit_finetune_av_fullft_sftD.slurm | awk '{print $4}')
echo "SFT-D submitted: job ${JOB_D} (text answers)"

# GRPO full-FT (on baseline lr=1e-6 checkpoint)
JOB_GRPO=$(sbatch --dependency=afterany:${JOB_D} submit_finetune_grpo_av_fullft.slurm | awk '{print $4}')
echo "GRPO full-FT submitted: job ${JOB_GRPO}"

echo ""
echo "Pipeline: baseline=${BASELINE_JOB} -> SFT-A=${JOB_A} -> SFT-B=${JOB_B} -> SFT-C=${JOB_C} -> SFT-D=${JOB_D} -> GRPO=${JOB_GRPO}"
echo "Monitor: squeue -u \$USER"
