#!/bin/bash
#SBATCH --job-name=minicpmo45-av-grpo
#SBATCH --partition=h100-ferranti
#SBATCH --reservation=kqr867_3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /weka/kuehne/kqr867/code/MiniCPM-o/finetune
bash finetune_grpo_av.sh
