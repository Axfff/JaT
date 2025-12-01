#!/bin/bash

# Run all 4 experiments from scratch with JiT-aligned training

# 1. Control (Spectrogram | Epsilon)
echo "Training Experiment 1: Control (Spectrogram | Epsilon)"
python3 src/train.py --experiment_name control_spec_eps --dataset_mode spectrogram --loss_type epsilon --epochs 50 --batch_size 32 --P_mean -0.8 --P_std 0.8 --noise_scale 1.0

# 2. Test 1 (Spectrogram | X)
echo "Training Experiment 2: Test 1 (Spectrogram | X)"
python3 src/train.py --experiment_name test_spec_x --dataset_mode spectrogram --loss_type x --epochs 50 --batch_size 32 --P_mean -0.8 --P_std 0.8 --noise_scale 1.0

# 3. Baseline (Raw | Epsilon)
echo "Training Experiment 3: Baseline (Raw | Epsilon)"
python3 src/train.py --experiment_name baseline_raw_eps --dataset_mode raw --loss_type epsilon --epochs 50 --batch_size 32 --P_mean -0.8 --P_std 0.8 --noise_scale 1.0

# 4. Main Event (Raw | X)
echo "Training Experiment 4: Main Event (Raw | X)"
python3 src/train.py --experiment_name main_raw_x --dataset_mode raw --loss_type x --epochs 50 --batch_size 32 --P_mean -0.8 --P_std 0.8 --noise_scale 1.0

# Evaluation
echo "Evaluating Experiment 1: Control"
python3 src/evaluate.py --checkpoint checkpoints/model_control_spec_eps_ep50.pth --dataset_mode spectrogram --num_samples 16 --output_dir results/control_spec_eps --pred_mode epsilon --noise_scale 1.0

echo "Evaluating Experiment 2: Test 1"
python3 src/evaluate.py --checkpoint checkpoints/model_test_spec_x_ep50.pth --dataset_mode spectrogram --num_samples 16 --output_dir results/test_spec_x --pred_mode x --noise_scale 1.0

echo "Evaluating Experiment 3: Baseline"
python3 src/evaluate.py --checkpoint checkpoints/model_baseline_raw_eps_ep50.pth --dataset_mode raw --num_samples 16 --output_dir results/baseline_raw_eps --pred_mode epsilon --noise_scale 1.0

echo "Evaluating Experiment 4: Main Event"
python3 src/evaluate.py --checkpoint checkpoints/model_main_raw_x_ep50.pth --dataset_mode raw --num_samples 16 --output_dir results/main_raw_x --pred_mode x --noise_scale 1.0

