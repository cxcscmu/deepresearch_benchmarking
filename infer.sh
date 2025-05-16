#!/bin/bash
#
#SBATCH --job-name=searchr1_infer
#SBATCH --output=logs/searchr1_infer_%j.out
#SBATCH --error=logs/searchr1_infer_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00

# Activate your conda env
source activate searchr1

# Export your SERPer key
export SERPER_API_KEY="<Your-API-Key>"

# Run the batch inference
python -u infer_batch.py \
  --input_file  path/to/your/queries/jsonl \
  --output_file output/file/path \
  --search_url   https://google.serper.dev/search \
  --k            3
