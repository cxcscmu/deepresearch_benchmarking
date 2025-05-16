#!/bin/bash
#
#SBATCH --job-name=searchr1_infer
#SBATCH --output=logs/searchr1_infer_test.out
#SBATCH --error=logs/searchr1_infer_test.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00

source activate searchr1

# Run the batch inference against your ClueWeb22 API
python -u infer_batch_our_api.py \
  --input_file   path/to/your/queries/jsonl \
  --output_file  output/file/path \
  --search_url   URL for ClueWeb22 \
  --k             3
