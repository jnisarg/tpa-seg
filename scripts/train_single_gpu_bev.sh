#!/bin/bash

# Prevent unwanted updates
export NO_ALBUMENTATIONS_UPDATE=1

#########################
# Configuration Options #
#########################

# Feature Flags
readonly ENABLE_RICH_LOGGING=false
readonly ENABLE_RESET_EXP=true

# Experiment Settings
readonly EXP_NAME="exp003-all"
readonly BASE_DIR="experiments"

# Model Architecture
readonly N_CLASSES=4
readonly CLASS_NAMES=("background" "trailer" "trailer_bar" "trailer_ball")
readonly WIDTH_LIST=(8 16 32 64 128)
readonly DEPTH_LIST=(1 2 2 2 2)
readonly HEAD_WIDTH=32
readonly HEAD_DEPTH=1

# Training Parameters
readonly EPOCHS=100
readonly BATCH_SIZE=8
readonly VAL_BATCH_SIZE=1
readonly NUM_WORKERS=4

# Model Components
readonly NORM_TYPE="bn2d"
readonly ACT_TYPE="relu6"

# Optimizer Settings
readonly OPTIMIZER="adamw"
readonly LEARNING_RATE=0.001
readonly MOMENTUM=0.9
readonly WEIGHT_DECAY=0.01
readonly BETA1=0.9
readonly BETA2=0.999
readonly ENABLE_NESTEROV=false
readonly ENABLE_AMSGRAD=false

# Learning Rate Scheduler
readonly LR_SCHEDULER="cosine"
readonly MIN_LR=0.0
readonly WARMUP_STEPS=0
readonly UPDATE_FREQ="step"
readonly POLY_POWER=0.9

# Checkpointing
readonly SAVE_FREQ=1
readonly KEEP_LAST_N=5

# Miscellaneous
readonly LOG_INTERVAL=100
readonly DEVICE="cuda"
readonly SEED=42
readonly TOP_K_PERCENT=0.8

###################
# Helper Functions #
###################

build_feature_flags() {
    local flags=()
    
    [[ "${ENABLE_RICH_LOGGING}" == true ]] && flags+=(--rich-logging)
    [[ "${ENABLE_RESET_EXP}" == true ]] && flags+=(--reset-exp)
    [[ "${ENABLE_NESTEROV}" == true ]] && flags+=(--nesterov)
    [[ "${ENABLE_AMSGRAD}" == true ]] && flags+=(--amsgrad)
    
    echo "${flags[*]}"
}

###################
# Main Function   #
###################

main() {
    # Get feature flags
    local feature_flags
    feature_flags=$(build_feature_flags)

    # Build command array
    local cmd=(
        python tools/train.py
        
        # Experiment Configuration
        --exp-name "${EXP_NAME}"
        --base-dir "${BASE_DIR}"
        
        # Model Architecture
        --n-classes "${N_CLASSES}"
        --class-names "${CLASS_NAMES[@]}"
        --width-list "${WIDTH_LIST[@]}"
        --depth-list "${DEPTH_LIST[@]}"
        --head-width "${HEAD_WIDTH}"
        --head-depth "${HEAD_DEPTH}"
        
        # Training Parameters
        --epochs "${EPOCHS}"
        --batch-size "${BATCH_SIZE}"
        --val-batch-size "${VAL_BATCH_SIZE}"
        --num-workers "${NUM_WORKERS}"
        
        # Model Components
        --norm "${NORM_TYPE}"
        --act "${ACT_TYPE}"
        
        # Optimizer Configuration
        --optimizer "${OPTIMIZER}"
        --lr "${LEARNING_RATE}"
        --momentum "${MOMENTUM}"
        --weight-decay "${WEIGHT_DECAY}"
        --beta1 "${BETA1}"
        --beta2 "${BETA2}"
        
        # Learning Rate Scheduler
        --lr-scheduler "${LR_SCHEDULER}"
        --min-lr "${MIN_LR}"
        --warmup-steps "${WARMUP_STEPS}"
        --update-frequency "${UPDATE_FREQ}"
        --poly-power "${POLY_POWER}"
        
        # Checkpointing
        --save-freq "${SAVE_FREQ}"
        --keep-last-n "${KEEP_LAST_N}"
        
        # Miscellaneous
        --log-interval "${LOG_INTERVAL}"
        --device "${DEVICE}"
        --seed "${SEED}"
        --top_k_percent "${TOP_K_PERCENT}"
    )
    
    # Add feature flags if any are enabled
    if [[ -n "${feature_flags}" ]]; then
        cmd+=($feature_flags)
    fi

    # Execute the command
    "${cmd[@]}"
}

# Error handling function
handle_error() {
    echo "Error occurred in line $1"
    exit 1
}

# Set error handling
trap 'handle_error $LINENO' ERR

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail  # Enable strict mode
    main "$@"
fi