#!/bin/bash
#SBATCH --job-name=eval-%1            # Job name will include the task, e.g., "eval-high_to_high"
#SBATCH --partition=standard          # The partition to run on
#SBATCH --nodes=1                     # We need one node
#SBATCH --ntasks-per-node=1           # One task
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=8             # Request 8 CPU cores
#SBATCH --mem=32G                     # Request 32 GB of RAM
#SBATCH --output=results/%1-%j.out    # Output file will be named after the task, e.g., "high_to_high-12345.out"

# --- Your commands start here ---

# This line takes the first argument you provide to the script (e.g., "high_to_high")
# and saves it in a variable called TASK_NAME.
TASK_NAME=$1

echo "Starting evaluation for task: $TASK_NAME"
echo "Job started on $(hostname) at $(date)"

# 1. Activate your Conda environment
source ~/miniconda3/bin/activate llm-eval-env

# 2. Run the evaluation
#    Notice how we use the $TASK_NAME variable to tell the harness which task to run
#    and how to name the final results file.
srun lm_eval \
    --model hf \
    --model_args pretrained=google/gemma-2b \
    --tasks $TASK_NAME \
    --include_path . \
    --device cuda:0 \
    --output_path results/${TASK_NAME}_results.json

echo "Job finished at $(date)"