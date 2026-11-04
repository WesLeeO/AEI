#!/bin/bash
# Full pipeline on one GPU (ETH Euler / any SLURM cluster).
#   sbatch --export=MODEL=Qwen/Qwen2.5-1.5B-Instruct,TAG=1.5b slurm/pipeline.sbatch
#   sbatch --export=MODEL=Qwen/Qwen2.5-7B-Instruct,TAG=7b --gres=gpumem:80g slurm/pipeline.sbatch
#SBATCH --job-name=ddx
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
TAG=${TAG:-1.5b}
cd "$(dirname "$0")/.."
mkdir -p logs results ckpt

# Euler: module load stack/2024-06 python_cuda/3.11.6 ; then a venv with requirements.txt
source ../.venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}

# 1) non-LLM baselines + expert dialogues (CPU, minutes)
[ -f results/baselines_test.json ] || python baselines.py --mode eval --n 2000
[ -f data/sft_expert.jsonl ] || python baselines.py --mode make_sft --n 20000

# 2) zero-shot LLM (with and without the question list in the prompt)
python evaluate.py --model "$MODEL" --name "zeroshot_${TAG}" --n 1000
python evaluate.py --model "$MODEL" --name "zeroshot_menu_${TAG}" --n 1000 --question_menu

# 3) SFT (filtered BC on expert dialogues)
python train_sft.py --model "$MODEL" --data data/sft_expert.jsonl --out "ckpt/sft_${TAG}"
python evaluate.py --model "$MODEL" --adapter "ckpt/sft_${TAG}" --name "sft_${TAG}" --n 1000

# 4) RL from the SFT policy
python train_rl.py --algo ppo  --model "$MODEL" --adapter "ckpt/sft_${TAG}" --out "ckpt/ppo_${TAG}"
python evaluate.py --model "$MODEL" --adapter "ckpt/ppo_${TAG}/best" --name "ppo_${TAG}" --n 1000
python train_rl.py --algo grpo --model "$MODEL" --adapter "ckpt/sft_${TAG}" --out "ckpt/grpo_${TAG}"
python evaluate.py --model "$MODEL" --adapter "ckpt/grpo_${TAG}/best" --name "grpo_${TAG}" --n 1000

python plot.py
