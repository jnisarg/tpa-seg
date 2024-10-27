#!/bin/bash

# Prevent unwanted updates
export NO_ALBUMENTATIONS_UPDATE=1

###################
# Configuration   #
###################

# Experiment Settings
readonly EXP_DIR="experiments/exp002"
readonly CHECKPOINT_PATH="${EXP_DIR}/checkpoints/best/best_checkpoint_e92_loss_0.0303_iou_0.9297.pth"
readonly OUTPUT_DIR="${EXP_DIR}/results"

# Model Architecture
readonly N_CLASSES=4
readonly CLASS_NAMES=("background" "trailer" "trailer_bar" "trailer_ball")
readonly WIDTH_LIST=(8 16 32 64 128)
readonly DEPTH_LIST=(1 2 2 2 2)
readonly HEAD_WIDTH=32
readonly HEAD_DEPTH=1

# Model Components
readonly NORM_TYPE="bn2d"
readonly ACT_TYPE="relu6"

# Evaluation Parameters
readonly BATCH_SIZE=1
readonly NUM_WORKERS=4
readonly SAVE_PREDICTIONS=true
readonly DEVICE="cuda"
readonly TOP_K_PERCENT=0.8

###################
# Main Function   #
###################

main() {
    # Build command array (safer than string concatenation)
    local cmd=(
        python tools/eval.py
        --checkpoint-path "${CHECKPOINT_PATH}"
        --n-classes "${N_CLASSES}"
        --class-names "${CLASS_NAMES[@]}"
        --width-list "${WIDTH_LIST[@]}"
        --depth-list "${DEPTH_LIST[@]}"
        --head-width "${HEAD_WIDTH}"
        --head-depth "${HEAD_DEPTH}"
        --batch-size "${BATCH_SIZE}"
        --num-workers "${NUM_WORKERS}"
        --output-dir "${OUTPUT_DIR}"
        --norm "${NORM_TYPE}"
        --act "${ACT_TYPE}"
        --device "${DEVICE}"
        --top_k_percent "${TOP_K_PERCENT}"
    )

    # Add optional flags
    if [[ "${SAVE_PREDICTIONS}" == true ]]; then
        cmd+=(--save-predictions)
    fi

    # Execute the command
    "${cmd[@]}"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi