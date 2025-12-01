# Spectrum Visualization Tools

This directory contains tools for comparing predicted spectrograms against groundtruth data.

## Overview

The visualization tools support two types of models:
- **Spectrogram Models**: Models that directly predict spectrograms (e.g., `control_spec_eps`)
- **Waveform Models**: Models that predict raw audio waveforms (e.g., `main_raw_x`)

For waveform models, the predictions are automatically converted to spectrograms for comparison.

## Files

- `visualize_spectrum.py`: Main visualization script
- `compare_all_experiments.sh`: Batch processing script for all experiments

## Quick Start

### Option 1: Compare All Experiments (Recommended)

Use pre-generated results from the `results/` directory:

```bash
./compare_all_experiments.sh
```

This will:
1. Process all experiment result directories in `results/`
2. Load pre-generated audio samples
3. Compare with groundtruth spectrograms
4. Generate visualizations and metrics for each experiment
5. Create a summary report at `spectrum_comparisons/SUMMARY.md`

### Option 2: Generate New Predictions and Compare

Edit `compare_all_experiments.sh` and set `use_pregenerated=false`, then run:

```bash
./compare_all_experiments.sh
```

### Option 3: Compare a Single Experiment

#### Using pre-generated predictions:

```bash
python visualize_spectrum.py \
    --use_predictions results/main_raw_x \
    --data_root ./data \
    --dataset_mode raw \
    --num_samples 16 \
    --num_display 8 \
    --output_dir spectrum_comparisons/main_raw_x
```

#### Generating predictions from checkpoint:

```bash
python visualize_spectrum.py \
    --checkpoint checkpoints/model_main_raw_x_ep50.pth \
    --data_root ./data \
    --dataset_mode raw \
    --pred_mode x \
    --num_samples 16 \
    --num_display 8 \
    --output_dir spectrum_comparisons/main_raw_x
```

## Command Line Arguments

### visualize_spectrum.py

Required (one of):
- `--checkpoint PATH`: Path to model checkpoint (generates new predictions)
- `--use_predictions PATH`: Path to directory with pre-generated predictions

Optional:
- `--data_root PATH`: Root directory for dataset (default: `./data`)
- `--dataset_mode {raw,spectrogram}`: Dataset mode (default: `raw`)
- `--pred_mode {epsilon,x,v}`: Prediction mode for model (default: `x`)
- `--num_samples N`: Number of samples to generate/compare (default: 16)
- `--num_display N`: Number of samples to display in visualization (default: 8)
- `--output_dir PATH`: Directory to save visualizations (default: `./spectrum_comparison`)
- `--subset {training,validation,testing}`: Dataset subset for groundtruth (default: `testing`)
- `--device {cuda,cpu}`: Device to run on (default: auto-detect)

## Output

For each experiment, the following files are generated:

1. **`spectrum_comparison.png`**: Visual comparison showing:
   - Column 1: Groundtruth spectrograms
   - Column 2: Predicted spectrograms
   - Column 3: Absolute difference

2. **`metrics.txt`**: Quantitative metrics including:
   - **MSE**: Mean Squared Error
   - **MAE**: Mean Absolute Error
   - **PSNR**: Peak Signal-to-Noise Ratio
   - **Spectral_Convergence**: Spectral convergence metric

3. **`SUMMARY.md`**: Combined summary report (when using batch script)

## Examples

### Compare spectrogram model

```bash
# Using pre-generated predictions
python visualize_spectrum.py \
    --use_predictions results/control_spec_eps \
    --dataset_mode spectrogram \
    --output_dir spectrum_comparisons/control_spec_eps

# From checkpoint
python visualize_spectrum.py \
    --checkpoint checkpoints/model_control_spec_eps_ep50.pth \
    --dataset_mode spectrogram \
    --pred_mode epsilon \
    --output_dir spectrum_comparisons/control_spec_eps
```

### Compare waveform model

```bash
# Using pre-generated predictions
python visualize_spectrum.py \
    --use_predictions results/main_raw_x \
    --dataset_mode raw \
    --output_dir spectrum_comparisons/main_raw_x

# From checkpoint
python visualize_spectrum.py \
    --checkpoint checkpoints/model_main_raw_x_ep50.pth \
    --dataset_mode raw \
    --pred_mode x \
    --output_dir spectrum_comparisons/main_raw_x
```

## Understanding the Metrics

- **MSE (Mean Squared Error)**: Lower is better. Measures average squared difference.
- **MAE (Mean Absolute Error)**: Lower is better. More interpretable than MSE.
- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better. Common in image/audio quality assessment.
- **Spectral Convergence**: Lower is better. Standard metric for spectrogram reconstruction.

## Notes

- The script automatically handles conversion:
  - **Waveform → Spectrogram**: Uses MelSpectrogram with same parameters as training
  - **Spectrogram Model Predictions**: Already in spectrogram format
  
- Groundtruth data is loaded from the SpeechCommands dataset (testing subset by default)

- Spectrograms use the same preprocessing as the dataset:
  - n_fft: 1024
  - hop_length: 256
  - n_mels: 64
  - Size: 64x64
  - Log scaling applied

## Troubleshooting

**Issue**: "Dataset not found"
- **Solution**: Make sure the SpeechCommands dataset is downloaded in the `data/` directory

**Issue**: "Checkpoint not found"
- **Solution**: Verify the checkpoint path. Check `checkpoints/` directory for available models.

**Issue**: "CUDA out of memory"
- **Solution**: Reduce `--num_samples` or use `--device cpu`
