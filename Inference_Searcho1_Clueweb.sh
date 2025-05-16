#!/bin/bash
#SBATCH --job-name=searcho1_infer
#SBATCH --output=logs/searcho1_infer_custom.out
#SBATCH --error=logs/searcho1_infer_custom.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

# activate your conda env
source activate search_o1

# point this at your custom search API base URL
export SEARCH_URL="<custom URL for ClueWeb or other APIs>"

python -u run_search_o1_clueweb22.py \
  --input_file      /path/to/custom/queries/ \
  --search_url      $SEARCH_URL \
  --model_path      Qwen/QwQ-32B-Preview \
  --max_search_limit 5 \
  --max_turn        10 \
  --top_k           5 \
  --max_doc_len     3000 \
  --use_jina        True \
  --jina_api_key    "<Your-JINA-API-Key>"
