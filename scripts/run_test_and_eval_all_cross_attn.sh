#!/bin/bash

# Set the full path to the checkpoint directory manually
CHECKPOINT_DIR="/home/stud/xiji/SC-Depth_Anything/checkpoints/cross_attn_nusc_6Hz/06-17-23:09"

# Derived paths
CKPT_BASE_DIR="${CHECKPOINT_DIR%%/checkpoints/*}"
BASE_DIR="/home/stud/xiji/SC-Depth_Anything"
COMPARATION_DIR="$BASE_DIR/comparation"
DATASET_DIR="/home/stud/xiji/storage/slurm/nuscenes_416_256"

# Scripts
TEST_SCRIPT="$BASE_DIR/test_vo_nusc.py"
EVAL_SCRIPT="$BASE_DIR/nusc_eval/eval_odom.py"

# Image dimensions
IMG_HEIGHT=256
IMG_WIDTH=416

echo "Scanning for checkpoints in: $CHECKPOINT_DIR"
echo "--------------------------------------------"

# Loop through checkpoints
find "$CHECKPOINT_DIR" -name "model_checkpoint_*.pth.tar" | sort -V | while read -r ckpt_path; do
    if [ ! -f "$ckpt_path" ]; then
        echo "No checkpoints found at: $ckpt_path"
        continue
    fi

    ckpt_file=$(basename "$ckpt_path")
    ckpt_name="${ckpt_file%.pth.tar}"
    relative_path="${ckpt_path#$CKPT_BASE_DIR/checkpoints/}"      # Get <model>/<timestamp>/exp_pose_checkpoint_5.pth.tar
    result_subpath="${relative_path%.pth.tar}"               # Remove .pth.tar
    result_dir="$COMPARATION_DIR/checkpoints/$result_subpath"

    # echo "=== Running test_vo_nusc.py for $ckpt_path ==="
    # python "$TEST_SCRIPT" \
    #     --interval 2 \
    #     --pretrained-model "$ckpt_path" \
    #     --img-height "$IMG_HEIGHT" \
    #     --img-width "$IMG_WIDTH" \
    #     --dataset-dir "$DATASET_DIR" \
    #     --output-dir "$COMPARATION_DIR"

    echo "=== Running eval_odom.py on $result_dir ==="
    python "$EVAL_SCRIPT" --result "$result_dir" --align '7dof' --interval 2

    echo ""
done
