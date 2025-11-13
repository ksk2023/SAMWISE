#!/bin/bash

# Example training script for SAMWISE with Dual Memory Bank
# This script demonstrates how to train the model with long-term and short-term memory

# Basic settings
DATASET="ytvos"
OUTPUT_DIR="output/samwise_dual_memory"
NAME_EXP="dual_memory_experiment"

# Model settings
SAM2_VERSION="base"
BATCH_SIZE=2
NUM_FRAMES=8
EPOCHS=6

# Dual Memory Bank settings
USE_DUAL_MEMORY="--use_dual_memory"
SHORT_TERM_CAPACITY=10
LONG_TERM_CAPACITY=20
OBJECT_SCORE_THRESHOLD=0.0
IOU_THRESHOLD=0.7
MASK_CONSISTENCY_THRESHOLD=0.8
SIMILARITY_THRESHOLD=0.85
FRAME_SAMPLING_INTERVAL=3

# CME settings (optional, can be used together)
USE_CME=""  # Add "--use_cme_head" to enable
CME_DECISION_WINDOW=4

# Data paths (modify according to your setup)
YTVOS_PATH="data/ref-youtube-vos"
COCO_PATH="data/coco"

echo "========================================"
echo "Training SAMWISE with Dual Memory Bank"
echo "========================================"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "Short-term capacity: $SHORT_TERM_CAPACITY"
echo "Long-term capacity: $LONG_TERM_CAPACITY"
echo "Frame sampling interval: $FRAME_SAMPLING_INTERVAL"
echo "========================================"

python main.py \
    --dataset_file $DATASET \
    --ytvos_path $YTVOS_PATH \
    --coco_path $COCO_PATH \
    --output_dir $OUTPUT_DIR \
    --name_exp $NAME_EXP \
    --sam2_version $SAM2_VERSION \
    --batch_size $BATCH_SIZE \
    --num_frames $NUM_FRAMES \
    --epochs $EPOCHS \
    $USE_DUAL_MEMORY \
    --short_term_capacity $SHORT_TERM_CAPACITY \
    --long_term_capacity $LONG_TERM_CAPACITY \
    --object_score_threshold $OBJECT_SCORE_THRESHOLD \
    --iou_threshold $IOU_THRESHOLD \
    --mask_consistency_threshold $MASK_CONSISTENCY_THRESHOLD \
    --similarity_threshold $SIMILARITY_THRESHOLD \
    --frame_sampling_interval $FRAME_SAMPLING_INTERVAL \
    $USE_CME \
    --cme_decision_window $CME_DECISION_WINDOW \
    --fusion_stages 1 2 3 \
    --HSA \
    --lr 1e-5 \
    --weight_decay 0.05 \
    --num_workers 4

echo "========================================"
echo "Training completed!"
echo "Checkpoints saved in: $OUTPUT_DIR"
echo "========================================"
