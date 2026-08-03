#!/bin/bash
#SBATCH --job-name takuma_asl_gradual_unfreeze
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1
#SBATCH --time 05:00:00

experiment_dir="experiments/takuma-asl_gradual_unfreeze"
project_dir="$SLURM_SUBMIT_DIR"
script_dir="$project_dir/$experiment_dir"

date
cd "$project_dir"
uv run python "$script_dir/run_exp.py"
uv run python "$script_dir/analyze.py"
uv run python "$script_dir/make_report.py"
date
