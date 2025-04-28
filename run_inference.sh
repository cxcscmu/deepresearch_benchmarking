#!/bin/bash
#SBATCH --job-name=last # Updated job name for rerun
#SBATCH --output=logs/%x-%j.out       # Log file for this specific rerun job
#SBATCH --error=logs/%x-%j.err        # Error file for this specific rerun job
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-12:00:00            # Adjust time based on expected runtime for ~4 combinations
#SBATCH --gpus=1

# =======================
# Environment Setup
# =======================
echo "Setting up environment..."
eval "$(conda shell.bash hook)"
conda activate rag || { echo "Failed to activate conda environment"; exit 1; }
export PYTHONPATH=.
echo "Environment setup complete."

export OPENAI_API_KEY="OpenAIKey"
export TAVILY_API_KEY="TavilyKey"


python test.py -i queries/new_154.jsonl
