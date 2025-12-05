import os
import argparse
import torch
from torch.cuda.amp import autocast
import torchaudio
import numpy as np
from tqdm import tqdm
from frechet_audio_distance import FrechetAudioDistance
from frechet_audio_distance import FrechetAudioDistance
from model import JiT
from dataset import get_dataloader, denormalize_spectrogram
import soundfile as sf
import torchaudio

# Monkeypatch torchaudio for speechbrain compatibility
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

from speechbrain.inference.vocoders import HIFIGAN

# Global vocoder instance
VOCODER = None

def load_vocoder(device='cuda'):
    global VOCODER
    if VOCODER is None:
        print("Loading HiFi-GAN vocoder...")
        try:
            # Use a local directory for the model to avoid re-downloading
            savedir = os.path.join(os.path.expanduser("~"), ".gemini", "speechbrain_hifigan")
            os.makedirs(savedir, exist_ok=True)
            VOCODER = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir=savedir, run_opts={"device": device})
            print("HiFi-GAN loaded.")
        except Exception as e:
            print(f"Failed to load HiFi-GAN: {e}")
            VOCODER = None
    return VOCODER



def sample(model, num_samples, steps=50, device='cuda', dataset_mode='raw', pred_mode='epsilon', noise_scale=1.0):
    model.eval()
    
    if dataset_mode == 'raw':
        shape = (num_samples, 1, 16384)
    elif dataset_mode in ['spectrogram', 'spectrum_1d']:
        # Both spectrogram and spectrum_1d use 64x64 mel spectrograms
        shape = (num_samples, 1, 64, 64)
    else:
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")
        
    z = torch.randn(shape, device=device) * noise_scale
    ts = torch.linspace(0, 1, steps, device=device)
    dt = ts[1] - ts[0]
    
    y = torch.randint(0, 35, (num_samples,), device=device)
    
    with torch.no_grad():
        for i in range(steps - 1):
            t = ts[i]
            t_batch = torch.ones(num_samples, device=device) * t
            
            # Model input t in [0, 1] (raw, not scaled)
            with autocast(enabled=(device!='cpu')):
                model_out = model(z, t_batch, y)
            
            # Calculate v
            # z_t = t * x + (1-t) * eps
            # v = x - eps
            
            # Calculate v
            # z_t = t * x + (1-t) * eps
            # v = x - eps
            
            if pred_mode == 'epsilon' or pred_mode == 'epsilon_epsilon_loss':
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
                # Create proper view shape for broadcasting: [B] -> [B, 1, ...] matching z's dimensions
                view_shape = [t_batch.shape[0]] + [1] * (z.dim() - 1)
                t_view = t_batch.view(*view_shape)
                v = (z - eps) / torch.maximum(t_view, torch.tensor(1e-3, device=device))
                
            elif pred_mode == 'x' or pred_mode == 'x_v_loss':
                x = model_out
                # v = (x - z_t) / (1 - t)
                # Note: t is in [0, 1].
                # At t=1, 1-t=0. We need to clip.
                # Create proper view shape for broadcasting: [B] -> [B, 1, ...] matching z's dimensions
                view_shape = [t_batch.shape[0]] + [1] * (z.dim() - 1)
                t_view = t_batch.view(*view_shape)
                v = (x - z) / torch.maximum(1 - t_view, torch.tensor(1e-5, device=device))
                
            elif pred_mode == 'v' or pred_mode == 'v_v_loss':
                # WARNING: This assumes model outputs v directly.
                # If trained with loss_type='v' in train.py, the model actually predicts x!
                # Use pred_mode='x' for models trained with loss_type='v'.
                v = model_out
            
            z = z + v * dt
            
    return z

def save_audio(batch, dataset_mode, output_dir, sample_rate=16000, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)
    
    vocoder = None
    if dataset_mode == 'spectrogram':
        vocoder = load_vocoder(device=device)

    for i, item in enumerate(batch):
        if dataset_mode == 'spectrogram':
            # item: [1, 64, 64] (or similar)
            # We need to ensure it's on the correct device for the vocoder
            spec = item.to(device)
            
            if spec.dim() == 2:
                spec = spec.unsqueeze(0) # [1, 64, 64]
                
            # Denormalize
            spec = denormalize_spectrogram(spec)
            
            # Log-Mel to Linear Mel (HiFi-GAN expects log-mel, but let's check range)
            # SpeechBrain HiFi-GAN expects log-mel spectrograms.
            # Our denormalize returns log-mel.
            
            # Resize from 64 mels to 80 mels
            # Input: [1, 64, T]
            # We treat it as an image [1, 1, 64, T] for interpolation
            spec_img = spec.unsqueeze(1) 
            
            # Interpolate
            # Note: We want to preserve the time dimension T, and change 64 -> 80
            # Target size: (80, T)
            T = spec.shape[-1]
            spec_80_img = torch.nn.functional.interpolate(spec_img, size=(80, T), mode='bilinear', align_corners=False)
            spec_80 = spec_80_img.squeeze(1) # [1, 80, T]
            
            # HiFi-GAN expects [B, T, Mel] or [B, Mel, T]?
            # In our test, [B, Mel, T] worked and produced [B, 1, T_out]
            
            waveform = None
            if vocoder is not None:
                try:
                    # decode_batch expects [B, Mel, T]
                    wav = vocoder.decode_batch(spec_80)
                    waveform = wav.squeeze(1) # [1, T_out]
                except Exception as e:
                    print(f"HiFi-GAN inference failed: {e}")
            
            if waveform is None:
                # Fallback to Griffin-Lim (Legacy)
                print("Falling back to Griffin-Lim...")
                inv_mel = torchaudio.transforms.InverseMelScale(n_stft=1024 // 2 + 1, n_mels=64, sample_rate=sample_rate).to(device)
                griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=256, n_iter=64, momentum=0.99).to(device)
                
                # Inverse log
                spec_linear = torch.exp(spec) - 1e-9
                try:
                    linear_spec = inv_mel(spec_linear)
                    waveform = griffin_lim(linear_spec)
                except Exception as e:
                    print(f"Griffin-Lim failed: {e}")
                    waveform = torch.zeros(1, 16000, device=device)

        else:
            waveform = item.to(device)
            
        # Normalize audio volume
        waveform = waveform.cpu()
        waveform = waveform / torch.max(torch.abs(waveform) + 1e-9)
        
        # Save
        wav_np = waveform.squeeze().numpy()
        sf.write(os.path.join(output_dir, f"sample_{i}.wav"), wav_np, sample_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--dataset_mode', type=str, default='raw', choices=['raw', 'spectrogram', 'spectrum_1d'])
    parser.add_argument('--pred_mode', type=str, default='x', choices=['epsilon', 'x', 'v', 'epsilon_epsilon_loss', 'v_v_loss', 'x_v_loss'])
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--noise_scale', type=float, default=1.0, help='Noise scale factor')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--hidden_size', type=int, default=512, help='Model hidden size')
    parser.add_argument('--depth', type=int, default=12, help='Model depth')
    parser.add_argument('--bottleneck_dim', type=int, default=None, help='Bottleneck dimension (default: hidden_size)')
    parser.add_argument('--in_context_len', type=int, default=0, help='In-context length')
    
    # Audio-specific model parameters (must match training configuration)
    parser.add_argument('--use_snake', action='store_true', default=False,
                        help='Use Snake activation (must match training config)')
    parser.add_argument('--freq_bins', type=int, default=64, help='Number of frequency bins (for spectrum_1d mode)')
    parser.add_argument('--time_frames', type=int, default=64, help='Number of time frames (for spectrum_1d mode)')
    
    args = parser.parse_args()
    
    # Load Model - determine configuration based on dataset_mode
    if args.dataset_mode == 'raw':
        input_size = 16384
        in_channels = 1
        is_1d = True
        is_spectrum = False
    elif args.dataset_mode == 'spectrogram':
        input_size = 64
        in_channels = 1
        is_1d = False
        is_spectrum = False
    elif args.dataset_mode == 'spectrum_1d':
        input_size = 64
        in_channels = 1
        is_1d = False
        is_spectrum = True
    else:
        raise ValueError(f"Unknown dataset_mode: {args.dataset_mode}")
        
    bottleneck_dim = args.bottleneck_dim if args.bottleneck_dim is not None else args.hidden_size

    model = JiT(
        input_size=input_size,
        patch_size=args.patch_size,
        in_channels=in_channels,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=8,
        num_classes=35,
        bottleneck_dim=bottleneck_dim,
        in_context_len=args.in_context_len,
        is_1d=is_1d,
        is_spectrum=is_spectrum,
        freq_bins=args.freq_bins,
        time_frames=args.time_frames,
        use_snake=args.use_snake
    ).to(args.device)
    
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state_dict)
    
    print("Generating samples...")
    samples = sample(model, args.num_samples, dataset_mode=args.dataset_mode, 
                    pred_mode=args.pred_mode, noise_scale=args.noise_scale, device=args.device)
    
    save_audio(samples, args.dataset_mode, args.output_dir, device=args.device)
    print(f"Saved to {args.output_dir}")
