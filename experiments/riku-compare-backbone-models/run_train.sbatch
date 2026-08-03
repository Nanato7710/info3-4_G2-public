#!/bin/bash
#SBATCH --job-name=riku-backbone-graph
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

experiment_dir="experiments/riku-compare-backbone-models"
project_dir="$SLURM_SUBMIT_DIR"
script_dir="$project_dir/$experiment_dir"

mkdir -p "$project_dir/logs"
mkdir -p "$script_dir/figures"

date
cd "$project_dir" || exit 1

uv run python "$script_dir/glaf.py"

date
