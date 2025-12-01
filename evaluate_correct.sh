#!/bin/bash

# Evaluation with correct prediction modes and JiT parameters

echo "Evaluating Experiment 1: Control (Spectrogram | Epsilon)"
python3 src/evaluate.py --checkpoint checkpoints/model_control_spec_eps_ep50.pth --dataset_mode spectrogram --num_samples 16 --output_dir results/control_spec_eps --pred_mode epsilon --noise_scale 1.0

echo "Evaluating Experiment 2: Test 1 (Spectrogram | X)"
python3 src/evaluate.py --checkpoint checkpoints/model_test_spec_x_ep50.pth --dataset_mode spectrogram --num_samples 16 --output_dir results/test_spec_x --pred_mode x --noise_scale 1.0

echo "Evaluating Experiment 3: Baseline (Raw | Epsilon)"
python3 src/evaluate.py --checkpoint checkpoints/model_baseline_raw_eps_ep50.pth --dataset_mode raw --num_samples 16 --output_dir results/baseline_raw_eps --pred_mode epsilon --noise_scale 1.0

echo "Evaluating Experiment 4: Main Event (Raw | X)"
python3 src/evaluate.py --checkpoint checkpoints/model_main_raw_x_ep50.pth --dataset_mode raw --num_samples 16 --output_dir results/main_raw_x --pred_mode x --noise_scale 1.0

