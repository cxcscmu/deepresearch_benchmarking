#!/bin/bash
#SBATCH --job-name=researchy
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00


export HF_HOME=/data/group_data/cx_group/query_generation_data/hf_cache/

eval "$(conda shell.bash hook)"
conda activate hf-deepsearch

set -o allexport
source keys.env
set +o allexport

python run_researchy_threaded_custom.py --model-id gpt-4o-mini --concurrency 32
