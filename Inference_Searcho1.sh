#!/bin/bash
#SBATCH --job-name=searcho1_infer
#SBATCH --output=logs/searcho1_infer_serper.out
#SBATCH --error=logs/searcho1_infer_serper.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4
#SBATCH --time=11:00:00

source activate search_o1

export SERPER_API_KEY="<Your-Serper-API-Key>"

python -u run_search_o1_serper.py \
  --input_file      /path/to/custom/queries/ \
  --serper_api_url  https://google.serper.dev/search \
  --model_path      Qwen/QwQ-32B-Preview \
  --max_search_limit 5 \
  --max_turn        10 \
  --top_k           5 \
  --max_doc_len     3000 \
  --use_jina        True \
  --jina_api_key    "<Your-JINA-API-Key>"