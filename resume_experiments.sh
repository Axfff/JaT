#!/bin/bash

# Resume Experiment 4: Main Event (Raw | X)
echo "Resuming Experiment 4: Main Event (Raw | X)"
python3 src/train.py --experiment_name main_raw_x --dataset_mode raw --loss_type x --epochs 50 --batch_size 32 --resume_checkpoint checkpoints/model_main_raw_x_ep20.pth

# Evaluation
echo "Evaluating Experiment 1: Control"
python3 src/evaluate.py --checkpoint checkpoints/model_control_spec_eps_ep50.pth --dataset_mode spectrogram --num_samples 16 --output_dir results/control_spec_eps

echo "Evaluating Experiment 2: Test 1"
python3 src/evaluate.py --checkpoint checkpoints/model_test_spec_x_ep50.pth --dataset_mode spectrogram --num_samples 16 --output_dir results/test_spec_x --pred_mode x

echo "Evaluating Experiment 3: Baseline"
python3 src/evaluate.py --checkpoint checkpoints/model_baseline_raw_eps_ep50.pth --dataset_mode raw --num_samples 16 --output_dir results/baseline_raw_eps

echo "Evaluating Experiment 4: Main Event"
python3 src/evaluate.py --checkpoint checkpoints/model_main_raw_x_ep50.pth --dataset_mode raw --num_samples 16 --output_dir results/main_raw_x --pred_mode x
