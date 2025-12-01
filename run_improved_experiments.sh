#!/bin/bash

# Run improved experiments with larger model and more epochs for ALL configurations
# Common Configuration: Hidden=768, Depth=16, Epochs=100

EPOCHS=100
HIDDEN_SIZE=768
DEPTH=16
BATCH_SIZE=32
NOISE_SCALE=1.0

# Function to run experiment
run_experiment() {
    local exp_name=$1
    local dataset_mode=$2
    local pred_mode=$3
    
    echo "=================================================="
    echo "Starting Experiment: $exp_name"
    echo "Dataset: $dataset_mode | Prediction: $pred_mode"
    echo "=================================================="
    
    # Train
    python3 src/train.py \
        --experiment_name "$exp_name" \
        --dataset_mode "$dataset_mode" \
        --loss_type "$pred_mode" \
        --epochs "$EPOCHS" \
        --hidden_size "$HIDDEN_SIZE" \
        --depth "$DEPTH" \
        --batch_size "$BATCH_SIZE" \
        --noise_scale "$NOISE_SCALE"
        
    # Evaluate
    echo "Evaluating $exp_name..."
    python3 src/evaluate.py \
        --checkpoint "checkpoints/model_${exp_name}_ep${EPOCHS}.pth" \
        --dataset_mode "$dataset_mode" \
        --num_samples 16 \
        --output_dir "results/${exp_name}" \
        --pred_mode "$pred_mode" \
        --noise_scale "$NOISE_SCALE" \
        --patch_size 512 \
        --hidden_size "$HIDDEN_SIZE" \
        --depth "$DEPTH"
        
    # Visualize
    echo "Visualizing $exp_name..."
    mkdir -p "spectrum_comparisons/${exp_name}"
    python3 visualize_spectrum.py \
        --use_predictions "results/${exp_name}" \
        --data_root ./data \
        --dataset_mode "$dataset_mode" \
        --num_samples 16 \
        --num_display 8 \
        --output_dir "spectrum_comparisons/${exp_name}" \
        --subset testing
        
    echo "Finished $exp_name"
}

# 1. Control: Spectrogram + Epsilon
run_experiment "improved_spec_eps" "spectrogram" "epsilon"

# 2. Test: Spectrogram + X
run_experiment "improved_spec_x" "spectrogram" "x"

# 3. Baseline: Raw + Epsilon
run_experiment "improved_raw_eps" "raw" "epsilon"

# 4. Main: Raw + X
run_experiment "improved_raw_x" "raw" "x"

echo "=================================================="
echo "All improved experiments completed!"
echo "=================================================="
