#!/bin/bash
set -e

# Experiment 1: epsilon-prediction
echo "Starting Experiment 1: epsilon-prediction"
python3 src/train.py \
    --experiment_name jit_8x8_hidden64_eps \
    --dataset_mode spectrogram \
    --patch_size 8 \
    --hidden_size 64 \
    --loss_type epsilon_epsilon_loss \
    --epochs 1 \
    --batch_size 32

# Experiment 2: v-prediction
echo "Starting Experiment 2: v-prediction"
python3 src/train.py \
    --experiment_name jit_8x8_hidden64_v \
    --dataset_mode spectrogram \
    --patch_size 8 \
    --hidden_size 64 \
    --loss_type v_v_loss \
    --epochs 1 \
    --batch_size 32

# Experiment 3: x-prediction with v-loss
echo "Starting Experiment 3: x-prediction with v-loss"
python3 src/train.py \
    --experiment_name jit_8x8_hidden64_x_v \
    --dataset_mode spectrogram \
    --patch_size 8 \
    --hidden_size 64 \
    --loss_type x_v_loss \
    --epochs 1 \
    --batch_size 32
