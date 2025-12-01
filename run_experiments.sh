#!/bin/bash

# Experiment 1: Control (Spectrogram | Epsilon)
echo "Running Experiment 1: Control (Spectrogram | Epsilon)"
python3 src/train.py --experiment_name control_spec_eps --dataset_mode spectrogram --loss_type epsilon --epochs 50 --batch_size 32

# Experiment 2: Test 1 (Spectrogram | X)
echo "Running Experiment 2: Test 1 (Spectrogram | X)"
python3 src/train.py --experiment_name test_spec_x --dataset_mode spectrogram --loss_type x --epochs 50 --batch_size 32

# Experiment 3: Baseline (Raw | Epsilon)
echo "Running Experiment 3: Baseline (Raw | Epsilon)"
python3 src/train.py --experiment_name baseline_raw_eps --dataset_mode raw --loss_type epsilon --epochs 50 --batch_size 32

# Experiment 4: Main Event (Raw | X)
echo "Running Experiment 4: Main Event (Raw | X)"
python3 src/train.py --experiment_name main_raw_x --dataset_mode raw --loss_type x --epochs 50 --batch_size 32
