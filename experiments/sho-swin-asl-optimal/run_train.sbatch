#!/bin/bash
#SBATCH --job-name sho-swin-asl-optimal
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1
#SBATCH --time 05:00:00

# 環境変数 CONFIG_PATH がなければデフォルトを使用
: "${CONFIG_PATH:=experiments/template/config.yaml}"

# config の親ディレクトリを実験ディレクトリとみなす
experiment_dir=$(dirname "$CONFIG_PATH")
project_dir="$SLURM_SUBMIT_DIR"

date
cd "$project_dir"

# 引数として CONFIG_PATH を各スクリプトに渡す
# 各スクリプトで --config を受け取れるようにしておく必要があります
uv run python "$experiment_dir/run_exp.py" --config "$CONFIG_PATH"
uv run python "$experiment_dir/analyze.py" --config "$CONFIG_PATH"
uv run python "$experiment_dir/make_report.py" --config "$CONFIG_PATH"

date