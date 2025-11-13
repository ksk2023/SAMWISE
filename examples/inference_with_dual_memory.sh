#!/bin/bash

# Example inference script for SAMWISE with Dual Memory Bank
# This script demonstrates how to run inference with long-term and short-term memory

# Dataset selection: "ytvos" or "davis" or "mevis"
DATASET="ytvos"

# Model checkpoint
CHECKPOINT="checkpoints/samwise_dual_memory_best.pth"

# Dual Memory Bank settings
USE_DUAL_MEMORY="--use_dual_memory"
SHORT_TERM_CAPACITY=10
LONG_TERM_CAPACITY=20
IOU_THRESHOLD=0.7
MASK_CONSISTENCY_THRESHOLD=0.8
SIMILARITY_THRESHOLD=0.85
FRAME_SAMPLING_INTERVAL=3

# Data paths (modify according to your setup)
YTVOS_PATH="data/ref-youtube-vos"
DAVIS_PATH="data/ref-davis"
MEVIS_PATH="data/MeViS_release"

# Output settings
RESULTS_PATH="results/${DATASET}_dual_memory"
VISUALIZE=""  # Add "--visualize" to enable visualization

echo "========================================"
echo "Running Inference with Dual Memory Bank"
echo "========================================"
echo "Dataset: $DATASET"
echo "Checkpoint: $CHECKPOINT"
echo "Results path: $RESULTS_PATH"
echo "Short-term capacity: $SHORT_TERM_CAPACITY"
echo "Long-term capacity: $LONG_TERM_CAPACITY"
echo "========================================"

# Run inference based on dataset
if [ "$DATASET" == "ytvos" ]; then
    echo "Running on Ref-YouTube-VOS..."
    python inference_ytvos.py \
        --resume $CHECKPOINT \
        --ytvos_path $YTVOS_PATH \
        --split valid \
        --results_path $RESULTS_PATH \
        $USE_DUAL_MEMORY \
        --short_term_capacity $SHORT_TERM_CAPACITY \
        --long_term_capacity $LONG_TERM_CAPACITY \
        --iou_threshold $IOU_THRESHOLD \
        --mask_consistency_threshold $MASK_CONSISTENCY_THRESHOLD \
        --similarity_threshold $SIMILARITY_THRESHOLD \
        --frame_sampling_interval $FRAME_SAMPLING_INTERVAL \
        $VISUALIZE

elif [ "$DATASET" == "davis" ]; then
    echo "Running on Ref-DAVIS..."
    python inference_davis.py \
        --resume $CHECKPOINT \
        --davis_path $DAVIS_PATH \
        --split val \
        --results_path $RESULTS_PATH \
        $USE_DUAL_MEMORY \
        --short_term_capacity $SHORT_TERM_CAPACITY \
        --long_term_capacity $LONG_TERM_CAPACITY \
        --iou_threshold $IOU_THRESHOLD \
        --mask_consistency_threshold $MASK_CONSISTENCY_THRESHOLD \
        --similarity_threshold $SIMILARITY_THRESHOLD \
        --frame_sampling_interval $FRAME_SAMPLING_INTERVAL \
        $VISUALIZE

elif [ "$DATASET" == "mevis" ]; then
    echo "Running on MeViS..."
    python inference_mevis.py \
        --resume $CHECKPOINT \
        --mevis_path $MEVIS_PATH \
        --split valid \
        --results_path $RESULTS_PATH \
        $USE_DUAL_MEMORY \
        --short_term_capacity $SHORT_TERM_CAPACITY \
        --long_term_capacity $LONG_TERM_CAPACITY \
        --iou_threshold $IOU_THRESHOLD \
        --mask_consistency_threshold $MASK_CONSISTENCY_THRESHOLD \
        --similarity_threshold $SIMILARITY_THRESHOLD \
        --frame_sampling_interval $FRAME_SAMPLING_INTERVAL \
        $VISUALIZE

else
    echo "Error: Unknown dataset '$DATASET'"
    echo "Please choose from: ytvos, davis, mevis"
    exit 1
fi

echo "========================================"
echo "Inference completed!"
echo "Results saved in: $RESULTS_PATH"
echo "========================================"

# Optional: Evaluate results if ground truth is available
if [ "$DATASET" == "davis" ]; then
    echo "Evaluating Ref-DAVIS results..."
    python -c "
from davis2017.evaluation import DAVISEvaluation
dataset = DAVISEvaluation(davis_root='$DAVIS_PATH', task='semi-supervised', gt_set='val')
metrics_res = dataset.evaluate('$RESULTS_PATH')
print(metrics_res)
"
fi
