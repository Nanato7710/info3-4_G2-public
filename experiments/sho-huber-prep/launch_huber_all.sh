#!/bin/bash

# 実験のディレクトリを変数にする
EXPERIMENT_DIR="experiments/sho-huber-prep"
CONFIG_DIR="$EXPERIMENT_DIR"
SBATCH_FILE="$EXPERIMENT_DIR/run_train.sbatch"

configs=("config_d015.yaml" "config_d020.yaml" "config_d025.yaml" "config_d030.yaml" "config_d035.yaml" "config_d040.yaml")

for config in "${configs[@]}"; do
    echo "-----------------------------------------"
    echo "Starting experiment with $config"
    echo "-----------------------------------------"
    
    # --wait をつけると、このジョブが完了するまで次のループに進まない
    sbatch --wait --job-name="huber-${config%.yaml}" "$SBATCH_FILE" "$CONFIG_DIR/$config"
    
    echo "Finished $config. Moving to next..."
done

echo "全ての実験が完了しました。"