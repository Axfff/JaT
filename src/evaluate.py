import os
import argparse
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from frechet_audio_distance import FrechetAudioDistance
from model import JustAudioTransformer
from dataset import get_dataloader
import soundfile as sf



def sample(model, num_samples, steps=50, device='cuda', dataset_mode='raw', pred_mode='epsilon', noise_scale=1.0):
    model.eval()
    
    if dataset_mode == 'raw':
        shape = (num_samples, 1, 16384)
    else:
        shape = (num_samples, 1, 4096)
        
    z = torch.randn(shape, device=device) * noise_scale
    ts = torch.linspace(0, 1, steps, device=device)
    dt = ts[1] - ts[0]
    
    y = torch.randint(0, 35, (num_samples,), device=device)
    
    with torch.no_grad():
        for i in range(steps - 1):
            t = ts[i]
            t_batch = torch.ones(num_samples, device=device) * t
            
            # Model input t in [0, 1] (raw, not scaled)
            model_out = model(z, t_batch, y)
            
            # Calculate v
            # z_t = t * x + (1-t) * eps
            # v = x - eps
            
            if pred_mode == 'epsilon':
                eps = model_out
                # x = (z - (1-t)*eps) / t
                # This is unstable at t=0.
                # But we only need v.
                # v = (z - eps) / t - eps ? No.
                # v = x - eps.
                # z = t(x) + (1-t)eps = t(v+eps) + (1-t)eps = tv + eps.
                # => v = (z - eps) / t.
                # Still unstable at t=0.
                
                # Alternative:
                # z_{t+dt} = z_t + v * dt
                # If we use standard DDIM/DDPM sampling, it's different.
                # But for Flow Matching:
                # We need v.
                # If we predict eps, we are essentially predicting the "noise" component.
                # At t=0, z=eps. So model(z, 0) should predict z.
                # v = (z - eps) / t is correct for Flow Matching.
                # We can clip t to avoid division by zero: max(t, 1e-5).
                
                # Use a larger epsilon for stability or conditional check
                # At t=0, v is undefined if we just divide. 
                # But we can assume v is large or just clamp t.
                v = (z - eps) / torch.maximum(t_batch.view(-1, 1, 1), torch.tensor(1e-3, device=device))
                
            elif pred_mode == 'x':
                x = model_out
                # v = (x - z_t) / (1 - t)
                # Note: t is in [0, 1].
                # At t=1, 1-t=0. We need to clip.
                v = (x - z) / torch.maximum(1 - t_batch.view(-1, 1, 1), torch.tensor(1e-5, device=device))
                
            elif pred_mode == 'v':
                # WARNING: This assumes model outputs v directly.
                # If trained with loss_type='v' in train.py, the model actually predicts x!
                # Use pred_mode='x' for models trained with loss_type='v'.
                v = model_out
            
            z = z + v * dt
            
    return z

def save_audio(batch, dataset_mode, output_dir, sample_rate=16000):
    os.makedirs(output_dir, exist_ok=True)
    for i, item in enumerate(batch):
        if dataset_mode == 'spectrogram':
            # item: [1, 4096] -> [1, 64, 64]
            spec = item.view(1, 64, 64)
            # Inverse Mel? Griffin-Lim.
            # We need to match the MelSpectrogram params from dataset.py
            # n_fft=1024, hop_length=256, n_mels=64
            # But we don't have the phase.
            # Griffin-Lim requires linear spectrogram, not Mel.
            # So we need InverseMelScale first.
            
            # This is complicated. 
            # For "Control", maybe we just visualize the spectrogram?
            # Or we try to invert.
            # Let's try to invert for FAD.
            
            inv_mel = torchaudio.transforms.InverseMelScale(n_stft=1024 // 2 + 1, n_mels=64, sample_rate=sample_rate)
            griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=256)
            
            # spec is log(mel + 1e-9). Inverse log.
            spec = torch.exp(spec) - 1e-9
            spec = spec.cpu()
            
            try:
                linear_spec = inv_mel(spec)
                waveform = griffin_lim(linear_spec)
            except Exception as e:
                print(f"Griffin-Lim failed: {e}")
                waveform = torch.zeros(1, 16000)
                
        else:
            waveform = item.cpu()
            
        # Normalize
        waveform = waveform / torch.max(torch.abs(waveform) + 1e-9)
        
        # torchaudio.save(os.path.join(output_dir, f"sample_{i}.wav"), waveform, sample_rate, backend="soundfile")
        # Use soundfile directly
        # waveform is [1, T] or [T]
        wav_np = waveform.squeeze().numpy()
        sf.write(os.path.join(output_dir, f"sample_{i}.wav"), wav_np, sample_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--dataset_mode', type=str, default='raw')
    parser.add_argument('--pred_mode', type=str, default='x', choices=['epsilon', 'x', 'v'])
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--noise_scale', type=float, default=1.0, help='Noise scale factor')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--hidden_size', type=int, default=512, help='Model hidden size')
    parser.add_argument('--depth', type=int, default=12, help='Model depth')
    
    args = parser.parse_args()
    
    # Load Model
    if args.dataset_mode == 'raw':
        input_size = 16384
        in_channels = 1
    else:
        input_size = 4096
        in_channels = 1
        
    model = JustAudioTransformer(
        input_size=input_size,
        patch_size=args.patch_size,
        in_channels=in_channels,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=8,
        num_classes=35
    ).to(args.device)
    
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state_dict)
    
    print("Generating samples...")
    samples = sample(model, args.num_samples, dataset_mode=args.dataset_mode, 
                    pred_mode=args.pred_mode, noise_scale=args.noise_scale, device=args.device)
    
    save_audio(samples, args.dataset_mode, args.output_dir)
    print(f"Saved to {args.output_dir}")
