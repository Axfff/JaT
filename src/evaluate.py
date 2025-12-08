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



def sample(model, num_samples, steps=50, device='cuda', dataset_mode='raw', pred_mode='epsilon', noise_scale=1.0, cfg_scale=1.0, use_fp16=True, sampler='euler'):
    """
    Generate samples using Flow Matching with optional Classifier-Free Guidance.
    
    Args:
        model: The JiT model
        num_samples: Number of samples to generate
        steps: Number of sampling steps
        device: Device to use
        dataset_mode: 'raw', 'spectrogram', or 'spectrum_1d'
        pred_mode: Prediction mode ('epsilon', 'x', 'v', etc.)
        noise_scale: Scale for initial noise
        cfg_scale: CFG guidance scale (1.0 = no guidance, >1.0 = stronger guidance)
        use_fp16: Whether to use FP16 mixed precision (default: True)
        sampler: Sampling method ('euler' or 'heun')
    """
    model.eval()
    
    if dataset_mode == 'raw':
        shape = (num_samples, 1, 16384)
    elif dataset_mode in ['spectrogram', 'spectrum_1d']:
        # Both spectrogram and spectrum_1d use freq_bins x time_frames mel spectrograms
        shape = (num_samples, 1, model.freq_bins, model.time_frames)
    else:
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")
        
    z = torch.randn(shape, device=device) * noise_scale
    ts = torch.linspace(0, 1, steps, device=device)
    dt = ts[1] - ts[0]
    
    y = torch.randint(0, 35, (num_samples,), device=device)
    
    def compute_velocity(z_in, t_val):
        """Compute velocity at given state and time."""
        t_batch = torch.ones(num_samples, device=device) * t_val
        
        with autocast(enabled=(use_fp16 and device != 'cpu')):
            if cfg_scale > 1.0:
                model_out = model.forward_with_cfg(z_in, t_batch, y, cfg_scale=cfg_scale)
            else:
                model_out = model(z_in, t_batch, y)
        
        # Calculate v based on prediction mode
        view_shape = [t_batch.shape[0]] + [1] * (z_in.dim() - 1)
        t_view = t_batch.view(*view_shape)
        
        if pred_mode == 'epsilon' or pred_mode == 'epsilon_epsilon_loss':
            eps = model_out
            v = (z_in - eps) / torch.maximum(t_view, torch.tensor(1e-3, device=device))
        elif pred_mode == 'x' or pred_mode == 'x_v_loss':
            x = model_out
            v = (x - z_in) / torch.maximum(1 - t_view, torch.tensor(1e-5, device=device))
        elif pred_mode == 'v' or pred_mode == 'v_v_loss':
            v = model_out
        else:
            raise ValueError(f"Unknown pred_mode: {pred_mode}")
        
        return v
    
    with torch.no_grad():
        for i in range(steps - 1):
            t = ts[i]
            t_next = ts[i + 1]
            
            if sampler == 'euler':
                # Euler method (1st order)
                v = compute_velocity(z, t)
                z = z + v * dt
                
            elif sampler == 'heun':
                # Heun's method (2nd order Runge-Kutta)
                # Step 1: Euler prediction
                v1 = compute_velocity(z, t)
                z_pred = z + v1 * dt
                
                # Step 2: Corrector using velocity at predicted point
                v2 = compute_velocity(z_pred, t_next)
                
                # Step 3: Average the two velocities
                z = z + (v1 + v2) * 0.5 * dt
            else:
                raise ValueError(f"Unknown sampler: {sampler}. Choose 'euler' or 'heun'.")
            
    return z

def save_audio(batch, dataset_mode, output_dir, sample_rate=16000, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)
    
    vocoder = None
    if dataset_mode in ['spectrogram', 'spectrum_1d']:
        vocoder = load_vocoder(device=device)

    for i, item in enumerate(batch):
        if dataset_mode in ['spectrogram', 'spectrum_1d']:
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
            
            # Resize from freq_bins mels to 80 mels
            # Input: [1, freq_bins, T]
            # We treat it as an image [1, 1, freq_bins, T] for interpolation
            spec_img = spec.unsqueeze(1) 
            
            # Interpolate
            # Note: We want to preserve the time dimension T, and change F -> 80
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
                n_mels = spec.shape[-2] # Should be freq_bins
                # n_stft assuming standard relation typically n_fft/2 + 1 >= n_mels. 
                # If we used n_fft=1024 for 64 mels, for other mels we might need adjustment.
                # But InverseMelScale needs to match what was used. 
                # We can try to infer n_stft or just use a large enough one.
                # Standard speech commands is 16k.
                
                inv_mel = torchaudio.transforms.InverseMelScale(n_stft=1024 // 2 + 1, n_mels=n_mels, sample_rate=sample_rate).to(device)
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
    parser.add_argument('--hop_size', type=int, default=None,
                        help='Hop size for overlapping patches (raw mode only). Must match training config.')
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
    
    # Classifier-Free Guidance
    parser.add_argument('--cfg_scale', type=float, default=1.0,
                        help='CFG guidance scale (1.0 = no guidance, recommended: 3.0-5.0 for sharper outputs)')
    
    # Sampler
    parser.add_argument('--sampler', type=str, default='euler', choices=['euler', 'heun'],
                        help='Sampling method: euler (1st order, faster) or heun (2nd order, better quality)')
    
    # Precision
    parser.add_argument('--no-fp16', dest='fp16', action='store_false',
                        help='Disable mixed precision inference (use full FP32, must match training)')
    parser.set_defaults(fp16=True)
    
    # EMA
    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='Use EMA weights from checkpoint (if available)')
    
    args = parser.parse_args()
    
    # Load Model - determine configuration based on dataset_mode
    if args.dataset_mode == 'raw':
        input_size = 16384
        in_channels = 1
        is_1d = True
        is_spectrum = False
    elif args.dataset_mode == 'spectrogram':
        input_size = args.freq_bins
        in_channels = 1
        is_1d = False
        is_spectrum = False
    elif args.dataset_mode == 'spectrum_1d':
        input_size = args.freq_bins
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
        use_snake=args.use_snake,
        hop_size=args.hop_size if args.dataset_mode == 'raw' else None  # Overlap for raw audio only
    ).to(args.device)
    
    # Load checkpoint - support both new format (dict) and legacy format (state_dict)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        if args.use_ema and 'ema' in checkpoint:
            print("Loading EMA weights for inference...")
            ema_params = checkpoint['ema']['params']
            
            # Check if EMA params are stored as dict (new format) or list (legacy)
            if isinstance(ema_params, dict):
                # New format: match by parameter name (safe)
                model_state = model.state_dict()
                missing_keys = []
                for name, model_p in model.named_parameters():
                    if name in ema_params:
                        model_p.data.copy_(ema_params[name].to(args.device))
                    else:
                        missing_keys.append(name)
                if missing_keys:
                    print(f"Warning: {len(missing_keys)} parameters not found in EMA state")
            else:
                # Legacy list format: positional matching (UNSAFE - may cause issues!)
                print("WARNING: Checkpoint uses legacy list-based EMA format.")
                print("         This may cause incorrect weight assignment!")
                print("         Re-train with the updated EMA class for reliable results.")
                for p, ema_p in zip(model.parameters(), ema_params):
                    p.data.copy_(ema_p.to(args.device))
        else:
            model.load_state_dict(checkpoint['model'])
            if args.use_ema:
                print("Warning: --use_ema specified but checkpoint has no EMA state")
    else:
        # Legacy format: checkpoint is just state_dict
        model.load_state_dict(checkpoint)
        if args.use_ema:
            print("Warning: --use_ema specified but checkpoint is legacy format (no EMA)")
    
    if args.cfg_scale > 1.0:
        print(f"Generating samples with CFG scale={args.cfg_scale}...")
    else:
        print("Generating samples...")
    
    if not args.fp16:
        print("Using FP32 inference (--no-fp16)")
        
    print(f"Using sampler: {args.sampler}")
        
    samples = sample(model, args.num_samples, dataset_mode=args.dataset_mode, 
                    pred_mode=args.pred_mode, noise_scale=args.noise_scale, 
                    device=args.device, cfg_scale=args.cfg_scale, use_fp16=args.fp16,
                    sampler=args.sampler)
    
    save_audio(samples, args.dataset_mode, args.output_dir, device=args.device)
    print(f"Saved to {args.output_dir}")

