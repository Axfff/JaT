#!/bin/bash

# Run improved experiments with larger model and more epochs for ALL configurations
# Common Configuration: Hidden=768, Depth=16, Epochs=100

EPOCHS=100
HIDDEN_SIZE=64
DEPTH=12
BATCH_SIZE=32
NOISE_SCALE=1.0

# Function to run experiment
run_experiment() {
    local exp_name=$1
    local dataset_mode=$2
    local pred_mode=$3
    local patch_size=$4
    local device=$5
    
    echo "=================================================="
    echo "Starting Experiment: $exp_name on $device"
    echo "Dataset: $dataset_mode | Prediction: $pred_mode"
    echo "=================================================="
    
    # Train
    python3 src/train.py \
        --experiment_name "$exp_name" \
        --dataset_mode "$dataset_mode" \
        --patch_size "$patch_size" \
        --loss_type "$pred_mode" \
        --epochs "$EPOCHS" \
        --hidden_size "$HIDDEN_SIZE" \
        --depth "$DEPTH" \
        --batch_size "$BATCH_SIZE" \
        --noise_scale "$NOISE_SCALE" \
        --device "$device"
        
    # Evaluate
    echo "Evaluating $exp_name..."
    python3 src/evaluate.py \
        --checkpoint "checkpoints/model_${exp_name}_ep${EPOCHS}.pth" \
        --dataset_mode "$dataset_mode" \
        --num_samples 16 \
        --output_dir "results/${exp_name}" \
        --pred_mode "$pred_mode" \
        --noise_scale "$NOISE_SCALE" \
        --patch_size "$patch_size" \
        --hidden_size "$HIDDEN_SIZE" \
        --depth "$DEPTH" \
        --bottleneck_dim "$HIDDEN_SIZE" \
        --in_context_len 0 \
        --device "$device"
        
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
        --subset testing \
        --device "$device"
        
    echo "Finished $exp_name"
}

# GPU 0 Experiments
(
    # 0. Control: Spectrogram + Epsilon_eps
    run_experiment "improved_spec_eps" "spectrogram" "epsilon_epsilon_loss" 8 "cuda:0"

    # 2. Test: Spectrogram + X_v
    run_experiment "improved_spec_x" "spectrogram" "x_v_loss" 8 "cuda:0"

    # 4. Main: Raw + X_v
    run_experiment "improved_raw_x" "raw" "x_v_loss" 64 "cuda:0"
    
    # 6. Extra: Raw + X_v (Large Patch)
    run_experiment "improved_raw_x_256" "raw" "x_v_loss" 256 "cuda:0"
) &

# GPU 1 Experiments
(
    # 1. Control: Spectrogram + Velocity_v (Renamed from improved_spec_eps)
    run_experiment "improved_spec_v" "spectrogram" "v_v_loss" 8 "cuda:1"

    # 3. Baseline: Raw + Velocity_v
    run_experiment "improved_raw_eps" "raw" "v_v_loss" 64 "cuda:1"
    
    # 5. Extra: Raw + Velocity_v (Large Patch)
    run_experiment "improved_raw_eps_256" "raw" "v_v_loss" 256 "cuda:1"
) &

wait

echo "=================================================="
echo "All improved experiments completed!"
echo "=================================================="
