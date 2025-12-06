import os
import argparse
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import soundfile as sf
from tqdm import tqdm

import sys
sys.path.append('src')
from src.model import JiT
from src.model import JiT
from src.dataset import get_dataloader, SpeechCommandsDataset, normalize_spectrogram


def waveform_to_spectrogram(waveform, sample_rate=16000):
    """
    Convert waveform to mel spectrogram matching the dataset preprocessing.
    
    Args:
        waveform: (1, T) or (T,) tensor
        sample_rate: sample rate
        
    Returns:
        spec: (64, 64) mel spectrogram
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=64
    )
    spec = mel_transform(waveform)
    
    # Resize to exactly 64x64
    spec = torch.nn.functional.interpolate(
        spec.unsqueeze(0), 
        size=(64, 64), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    # Log scaling
    spec = torch.log(spec + 1e-9)
    
    # Normalize using shared function
    spec = normalize_spectrogram(spec)
    
    return spec.squeeze(0)  # (64, 64)


def load_model_predictions(checkpoint_path, dataset_mode, pred_mode, num_samples, patch_size=512, device='cuda'):
    """
    Load model and generate predictions.
    
    Args:
        checkpoint_path: path to model checkpoint
        dataset_mode: 'raw' or 'spectrogram'
        pred_mode: 'epsilon', 'x', or 'v'
        num_samples: number of samples to generate
        device: device to run on
        
    Returns:
        predictions: (N, 1, L) for raw or (N, 1, 4096) for spectrogram
    """
    # Import sampling function from evaluate
    from src.evaluate import sample
    
    # Load Model
    if dataset_mode == 'raw':
        input_size = 16384
        in_channels = 1
        is_1d = True
        is_spectrum = False
    elif dataset_mode == 'spectrum_1d':
        input_size = 64
        in_channels = 1
        is_1d = False
        is_spectrum = True
    elif dataset_mode == 'spectrogram':
        input_size = 64
        in_channels = 1
        is_1d = False
        is_spectrum = False
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")
        
    model = JiT(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=512,
        depth=12,
        num_heads=8,
        num_classes=35,
        bottleneck_dim=512, # Default matching hidden_size
        in_context_len=0,   # Default 0
        is_1d=is_1d,
        is_spectrum=is_spectrum,
        freq_bins=64,
        time_frames=64
    ).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    print(f"Generating {num_samples} predictions...")
    predictions = sample(
        model, 
        num_samples, 
        dataset_mode=dataset_mode, 
        pred_mode=pred_mode, 
        device=device
    )
    
    return predictions


def load_groundtruth(data_root, num_samples, dataset_mode='raw', subset='testing'):
    """
    Load groundtruth samples from dataset.
    
    Args:
        data_root: root directory for dataset
        num_samples: number of samples to load
        dataset_mode: 'raw' or 'spectrogram'
        subset: 'training', 'validation', or 'testing'
        
    Returns:
        gt_data: list of groundtruth samples
        gt_labels: list of labels
    """
    dataset = SpeechCommandsDataset(
        data_root, 
        mode=dataset_mode, 
        subset=subset, 
        download=False
    )
    
    gt_data = []
    gt_labels = []
    
    print(f"Loading {num_samples} groundtruth samples...")
    for i in tqdm(range(min(num_samples, len(dataset)))):
        data, label = dataset[i]
        gt_data.append(data)
        gt_labels.append(label)
    
    return gt_data, gt_labels


def visualize_comparison(predictions, groundtruth, save_path, pred_is_waveform=True, gt_is_waveform=True, num_display=8):
    """
    Visualize predicted vs groundtruth spectrograms.
    
    Args:
        predictions: predicted data (waveforms or spectrograms)
        groundtruth: groundtruth data (waveforms or spectrograms)
        save_path: path to save the visualization
        pred_is_waveform: whether predictions are waveforms (True) or spectrograms (False)
        gt_is_waveform: whether groundtruth are waveforms (True) or spectrograms (False)
        num_display: number of samples to display
    """
    num_display = min(num_display, len(predictions), len(groundtruth))
    
    # Convert to spectrograms if needed
    pred_specs = []
    gt_specs = []
    
    for i in range(num_display):
        # Process predictions
        if pred_is_waveform:
            # predictions are waveforms, convert to spectrogram
            pred_spec = waveform_to_spectrogram(predictions[i].cpu())
        else:
            # predictions are already spectrograms [1, 64, 64] or [64, 64]
            pred_tensor = predictions[i].cpu()
            if pred_tensor.dim() == 3:
                pred_spec = pred_tensor.squeeze(0)  # [1, 64, 64] -> [64, 64]
            else:
                pred_spec = pred_tensor.view(64, 64)
        pred_specs.append(pred_spec)
        
        # Process groundtruth
        if gt_is_waveform:
            # groundtruth are waveforms, convert to spectrogram
            gt_spec = waveform_to_spectrogram(groundtruth[i])
        else:
            # groundtruth are already spectrograms
            gt_spec = groundtruth[i].view(64, 64)
        gt_specs.append(gt_spec)
    
    # Create figure
    fig = plt.figure(figsize=(16, 2 * num_display))
    gs = GridSpec(num_display, 3, figure=fig, wspace=0.3, hspace=0.4)
    
    for i in range(num_display):
        # Groundtruth spectrogram
        ax_gt = fig.add_subplot(gs[i, 0])
        im_gt = ax_gt.imshow(
            gt_specs[i].numpy(), 
            aspect='auto', 
            origin='lower', 
            cmap='viridis',
            interpolation='nearest'
        )
        ax_gt.set_title(f'Groundtruth {i}')
        ax_gt.set_ylabel('Mel Frequency')
        ax_gt.set_xlabel('Time')
        plt.colorbar(im_gt, ax=ax_gt)
        
        # Predicted spectrogram
        ax_pred = fig.add_subplot(gs[i, 1])
        im_pred = ax_pred.imshow(
            pred_specs[i].numpy(), 
            aspect='auto', 
            origin='lower', 
            cmap='viridis',
            interpolation='nearest'
        )
        ax_pred.set_title(f'Predicted {i}')
        ax_pred.set_ylabel('Mel Frequency')
        ax_pred.set_xlabel('Time')
        plt.colorbar(im_pred, ax=ax_pred)
        
        # Difference
        ax_diff = fig.add_subplot(gs[i, 2])
        diff = (pred_specs[i] - gt_specs[i]).abs()
        im_diff = ax_diff.imshow(
            diff.numpy(), 
            aspect='auto', 
            origin='lower', 
            cmap='hot',
            interpolation='nearest'
        )
        ax_diff.set_title(f'Absolute Difference {i}')
        ax_diff.set_ylabel('Mel Frequency')
        ax_diff.set_xlabel('Time')
        plt.colorbar(im_diff, ax=ax_diff)
    
    pred_src = "Waveform" if pred_is_waveform else "Spectrogram"
    gt_src = "Waveform" if gt_is_waveform else "Spectrogram"
    plt.suptitle(
        f'Spectrum Comparison (Pred: {pred_src} | GT: {gt_src})', 
        fontsize=16, 
        y=0.995
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


def compute_metrics(predictions, groundtruth, pred_is_waveform=True, gt_is_waveform=True):
    """
    Compute quantitative metrics comparing predictions and groundtruth.
    
    Args:
        predictions: predicted data
        groundtruth: groundtruth data
        pred_is_waveform: whether predictions are waveforms (True) or spectrograms (False)
        gt_is_waveform: whether groundtruth are waveforms (True) or spectrograms (False)
        
    Returns:
        metrics: dictionary of metrics
    """
    metrics = {}
    
    # Convert to spectrograms for comparison
    pred_specs = []
    gt_specs = []
    
    num_samples = min(len(predictions), len(groundtruth))
    
    for i in range(num_samples):
        if pred_is_waveform:
            pred_spec = waveform_to_spectrogram(predictions[i].cpu())
        else:
            pred_tensor = predictions[i].cpu()
            if pred_tensor.dim() == 3:
                pred_spec = pred_tensor.squeeze(0)
            else:
                pred_spec = pred_tensor.view(64, 64)
            
        if gt_is_waveform:
            gt_spec = waveform_to_spectrogram(groundtruth[i])
        else:
            gt_spec = groundtruth[i].view(64, 64)
        
        pred_specs.append(pred_spec)
        gt_specs.append(gt_spec)
    
    # Check if we have any samples to compare
    if len(pred_specs) == 0 or len(gt_specs) == 0:
        print("ERROR: No samples to compare. Predictions or groundtruth list is empty.")
        print(f"  - Predictions: {len(predictions)} samples")
        print(f"  - Groundtruth: {len(groundtruth)} samples")
        return {}
    
    # Stack into tensors
    pred_specs = torch.stack(pred_specs)  # (N, 64, 64)
    gt_specs = torch.stack(gt_specs)      # (N, 64, 64)
    
    # Mean Squared Error
    mse = torch.mean((pred_specs - gt_specs) ** 2).item()
    metrics['MSE'] = mse
    
    # Mean Absolute Error
    mae = torch.mean(torch.abs(pred_specs - gt_specs)).item()
    metrics['MAE'] = mae
    
    # Peak Signal-to-Noise Ratio
    # PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    # Assuming normalized spectrograms, MAX = max value in gt_specs
    max_val = gt_specs.max().item()
    if mse > 0:
        psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
        metrics['PSNR'] = psnr
    else:
        metrics['PSNR'] = float('inf')
    
    # Spectral Convergence (common in audio)
    spec_conv = torch.norm(pred_specs - gt_specs, p='fro') / torch.norm(gt_specs, p='fro')
    metrics['Spectral_Convergence'] = spec_conv.item()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Visualize predicted spectrum vs groundtruth')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (required if not using --use_predictions)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset')
    parser.add_argument('--dataset_mode', type=str, default='raw', 
                        choices=['raw', 'spectrogram', 'spectrum_1d'],
                        help='Dataset mode: raw waveform, spectrogram, or spectrum_1d')
    parser.add_argument('--pred_mode', type=str, default='x', 
                        choices=['epsilon', 'x', 'v'],
                        help='Prediction mode for the model')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to generate and compare')
    parser.add_argument('--num_display', type=int, default=8,
                        help='Number of samples to display in visualization')
    parser.add_argument('--output_dir', type=str, default='./spectrum_comparison',
                        help='Directory to save visualizations')
    parser.add_argument('--subset', type=str, default='testing', 
                        choices=['training', 'validation', 'testing'],
                        help='Dataset subset for groundtruth')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    parser.add_argument('--use_predictions', type=str, default=None,
                        help='Optional: path to pre-generated predictions directory (skip model inference)')
    parser.add_argument('--patch_size', type=int, default=512,
                        help='Patch size for the model (default: 512)')
    
    args = parser.parse_args()
    
    # Validation: need either checkpoint or use_predictions
    if not args.checkpoint and not args.use_predictions:
        parser.error("Either --checkpoint or --use_predictions must be provided")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or generate predictions
    pred_is_waveform = True  # Default assumption
    if args.use_predictions:
        print(f"Loading predictions from {args.use_predictions}")
        predictions = []
        for i in range(args.num_samples):
            wav_path = os.path.join(args.use_predictions, f'sample_{i}.wav')
            if not os.path.exists(wav_path):
                print(f"Warning: {wav_path} not found, stopping at {i} samples")
                break
            wav_np, sr = sf.read(wav_path)
            waveform = torch.from_numpy(wav_np).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            predictions.append(waveform)
        # When loading from audio files, predictions are always waveforms
        pred_is_waveform = True
    else:
        # Generate predictions using model
        predictions = load_model_predictions(
            args.checkpoint,
            args.dataset_mode,
            args.pred_mode,
            args.num_samples,
            args.patch_size,
            args.device
        )
        predictions = [predictions[i] for i in range(len(predictions))]
        # Model predictions match the dataset_mode
        pred_is_waveform = (args.dataset_mode == 'raw')
    
    # Load groundtruth
    groundtruth, labels = load_groundtruth(
        args.data_root,
        args.num_samples,
        dataset_mode=args.dataset_mode,
        subset=args.subset
    )
    # Groundtruth format matches the dataset_mode
    gt_is_waveform = (args.dataset_mode == 'raw')
    
    # Ensure same number of samples
    num_samples = min(len(predictions), len(groundtruth))
    predictions = predictions[:num_samples]
    groundtruth = groundtruth[:num_samples]
    
    print(f"\nComparing {num_samples} samples...")
    print(f"Predictions are: {'waveforms' if pred_is_waveform else 'spectrograms'}")
    print(f"Groundtruth are: {'waveforms' if gt_is_waveform else 'spectrograms'}")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, groundtruth, pred_is_waveform, gt_is_waveform)
    
    if not metrics:
        print("\nERROR: Could not compute metrics (likely due to empty predictions).")
        print("Exiting without creating visualization.")
        return
    
    print("\n" + "="*50)
    print("METRICS:")
    print("="*50)
    for metric_name, value in metrics.items():
        print(f"{metric_name:25s}: {value:.6f}")
    print("="*50)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Spectrum Comparison Metrics\n")
        f.write("="*50 + "\n")
        f.write(f"Checkpoint: {args.checkpoint if args.checkpoint else 'N/A (using pre-generated)'}\n")
        f.write(f"Predictions: {args.use_predictions if args.use_predictions else 'Generated from model'}\n")
        f.write(f"Dataset Mode: {args.dataset_mode}\n")
        f.write(f"Prediction Mode: {args.pred_mode if not args.use_predictions else 'N/A'}\n")
        f.write(f"Number of Samples: {num_samples}\n")
        f.write(f"Prediction Format: {'waveform' if pred_is_waveform else 'spectrogram'}\n")
        f.write(f"Groundtruth Format: {'waveform' if gt_is_waveform else 'spectrogram'}\n")
        f.write("="*50 + "\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name:25s}: {value:.6f}\n")
    print(f"\nMetrics saved to {metrics_path}")
    
    # Visualize
    print("\nCreating visualization...")
    viz_path = os.path.join(args.output_dir, 'spectrum_comparison.png')
    visualize_comparison(
        predictions, 
        groundtruth, 
        viz_path, 
        pred_is_waveform,
        gt_is_waveform,
        args.num_display
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
