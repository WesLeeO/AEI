#!/bin/bash
# Full pipeline on one GPU (ETH Euler / any SLURM cluster). Submit from the repo root.
#   easy setting (original DDXPlus questions, single diagnosis):
#     sbatch --export=ALL,MODEL=Qwen/Qwen2.5-1.5B-Instruct,TAG=1.5b slurm/pipeline.sbatch
#   hard setting (yes/no questions only, ranked top-3 differential):
#     sbatch --export=ALL,MODEL=Qwen/Qwen2.5-1.5B-Instruct,TAG=1.5b,SETTING=hard slurm/pipeline.sbatch
#   bigger model:
#     sbatch --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct,TAG=7b --gres=gpumem:80g slurm/pipeline.sbatch
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
SETTING=${SETTING:-easy}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs results ckpt

if [ "$SETTING" = hard ]; then
  ENV_FLAGS="--binary_questions --differential"
  SUFFIX="_binary_ddx"
else
  ENV_FLAGS=""
  SUFFIX=""
fi
NAME="${TAG}${SUFFIX}"

# Euler: module load stack/2024-06 python_cuda/3.11.6 ; then a venv with requirements.txt
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}

# 1) non-LLM baselines + expert dialogues (CPU, minutes)
[ -f "results/baselines_test${SUFFIX}.json" ] || python baselines.py --mode eval --n 2000 $ENV_FLAGS
[ -f "data/sft_expert${SUFFIX}.jsonl" ] || python baselines.py --mode make_sft --n 20000 $ENV_FLAGS

# 2) zero-shot LLM (with and without the question list in the prompt)
python evaluate.py --model "$MODEL" --name "zeroshot_${NAME}" --n 1000 $ENV_FLAGS
python evaluate.py --model "$MODEL" --name "zeroshot_menu_${NAME}" --n 1000 --question_menu $ENV_FLAGS

# 3) SFT (filtered BC on expert dialogues)
python train_sft.py --model "$MODEL" --data "data/sft_expert${SUFFIX}.jsonl" --out "ckpt/sft_${NAME}" $ENV_FLAGS
python evaluate.py --model "$MODEL" --adapter "ckpt/sft_${NAME}" --name "sft_${NAME}" --n 1000 $ENV_FLAGS

# 4) RL from the SFT policy
python train_rl.py --algo ppo  --model "$MODEL" --adapter "ckpt/sft_${NAME}" --out "ckpt/ppo_${NAME}" $ENV_FLAGS
python evaluate.py --model "$MODEL" --adapter "ckpt/ppo_${NAME}/best" --name "ppo_${NAME}" --n 1000 $ENV_FLAGS
python train_rl.py --algo grpo --model "$MODEL" --adapter "ckpt/sft_${NAME}" --out "ckpt/grpo_${NAME}" $ENV_FLAGS
python evaluate.py --model "$MODEL" --adapter "ckpt/grpo_${NAME}/best" --name "grpo_${NAME}" --n 1000 $ENV_FLAGS

python plot.py
