KITTI="storage/kitti_odometry"
NUSC="storage/nuscenes_original_size"
WAYMO="storage/waymo_original_size/waymo_original_size_val"

INTERVAL=1 # default
WINDOW_LENGTH=8
SLIDING_STEP=7
EVAL_INTERVAL=1 # default
RESCALE=1 # default
ALIGN="7dof" # default


WINDOW_LENGTH=(8 250 250)
SLIDING_STEP=(7 249 249)


DATASET_DIRS=(
    "$KITTI"
    "$NUSC"
    "$WAYMO"
)

DATASETS=(
    "kitti"
    "nusc"
    "waymo"
)

# Folder names produced by inference
EVAL_NAMES=(
    "kitti_odometry"
    "nusc"
    "waymo"
)

RESULT_ROOT="eval_output/${CKPT_DIR%.ckpt}"

echo "start inference"
echo "Ckpt dir: $CKPT_DIR"
echo "Result root: $RESULT_ROOT"

for i in "${!DATASET_DIRS[@]}"; do
    dataset_dir="${DATASET_DIRS[$i]}"
    dataset="${DATASETS[$i]}"
    

    echo "Running inference on dataset: $dataset"
    echo "Dataset dir: $dataset_dir"

    python inference/traj_vis_loc_and_global.py \
        --interval "$INTERVAL" \
        --dataset-dir "$dataset_dir" \
        --pretrained-model "$CKPT_DIR" \
        --window-length "${WINDOW_LENGTH[$i]}" \
        --sliding-step "${SLIDING_STEP[$i]}" \
        --eval-interval "$EVAL_INTERVAL"

    echo "Inference finished for dataset: $dataset"
done

echo "Inference finished for all datasets"
echo "start evaluation"

for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    eval_name="${EVAL_NAMES[$i]}"

    eval_dir="$RESULT_ROOT/$eval_name"
    poses_dir="$eval_dir/len${WINDOW_LENGTH[$i]}_step${SLIDING_STEP[$i]}_evalInterval${EVAL_INTERVAL}_rescale${RESCALE}/predicted_poses"

    echo "Evaluating dataset: $dataset"
    echo "Poses dir: $poses_dir"

    python -m eval.eval_global_odom \
        --interval "$INTERVAL" \
        --result "$poses_dir" \
        --align "$ALIGN" \
        --dataset "$dataset"

    echo "Evaluation finished for dataset: $dataset"
done

echo "Evaluation finished for all datasets"