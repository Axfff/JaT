import os
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from dataset import get_dataloader
from model import JiT, JiT_models


# --- Cosine Schedule for t sampling ---
def cosine_schedule(t, s=0.008):
    """
    Cosine schedule that transforms uniform t to concentrate samples near t=0.
    
    This focuses more training on the "high-fidelity" regime (near t=0),
    forcing the model to learn how to clean up the final layer of noise.
    
    Args:
        t: Uniformly sampled timesteps in [0, 1]
        s: Small offset to prevent singularity at t=0 (default: 0.008)
        
    Returns:
        Rescaled t values concentrated near 0
    """
    # Cosine schedule: more samples near t=0 (clean data)
    return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2


# --- Multi-Scale Spectral Loss for Audio ---
class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-scale spectral loss to improve audio fidelity.
    
    Computes L1 loss between STFT magnitudes at multiple window sizes.
    This guides the model to preserve high-frequency content that MSE 
    on waveforms tends to blur out.
    
    Args:
        fft_sizes: List of FFT window sizes to use (default: [512, 1024, 2048])
        hop_ratio: Hop length as ratio of FFT size (default: 0.25)
        weight: Weight for the spectral loss term (default: 0.1)
        use_log: If True, compute loss on log-magnitude (reduces penalty on loud signals)
    """
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_ratio=0.25, weight=0.1, use_log=False):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_ratio = hop_ratio
        self.weight = weight
        self.use_log = use_log
        self.eps = 1e-7  # Small constant for log stability
        
    def forward(self, pred_wave, target_wave):
        """
        Compute multi-scale spectral loss.
        
        Args:
            pred_wave: Predicted waveform [B, C, L] or [B, L]
            target_wave: Target waveform [B, C, L] or [B, L]
            
        Returns:
            Spectral loss value (already weighted)
        """
        # Flatten to [B, L] if needed
        if pred_wave.dim() == 3:
            pred_wave = pred_wave.squeeze(1)
            target_wave = target_wave.squeeze(1)
        
        loss = 0.0
        for fft_size in self.fft_sizes:
            hop_length = int(fft_size * self.hop_ratio)
            
            # Compute STFT magnitude
            pred_spec = torch.stft(
                pred_wave, 
                n_fft=fft_size, 
                hop_length=hop_length,
                window=torch.hann_window(fft_size, device=pred_wave.device),
                return_complex=True
            ).abs()
            
            target_spec = torch.stft(
                target_wave, 
                n_fft=fft_size, 
                hop_length=hop_length,
                window=torch.hann_window(fft_size, device=target_wave.device),
                return_complex=True
            ).abs()
            
            if self.use_log:
                # Log-scale spectral loss: reduces penalty on loud signals,
                # balances training across frequencies
                pred_spec = torch.log(pred_spec + self.eps)
                target_spec = torch.log(target_spec + self.eps)
            
            # L1 loss on magnitude (ignores phase misalignment)
            loss += F.l1_loss(pred_spec, target_spec)
        
        # Average over number of scales and apply weight
        return self.weight * (loss / len(self.fft_sizes))


# --- Noise Scheduler ---
def get_beta_schedule(num_timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, num_timesteps)

class Diffusion:
    def __init__(self, num_timesteps=1000, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = get_beta_schedule(num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise, noise

# --- Loss Calculation ---
def calculate_loss(model, x_0, t, y, diffusion, loss_type='epsilon'):
    noise = torch.randn_like(x_0)
    x_t, _ = diffusion.q_sample(x_0, t, noise)
    
    model_output = model(x_t, t, y)
    
    if loss_type == 'epsilon':
        target = noise
        loss = nn.functional.mse_loss(model_output, target)
    elif loss_type == 'x':
        target = x_0
        loss = nn.functional.mse_loss(model_output, target)
    elif loss_type == 'v':
        # v = alpha_t * noise - sigma_t * x_0 (Wait, check paper definition)
        # Usually v = alpha_t * epsilon - sigma_t * x_0
        # Or v = (x_t - x_0) / ...
        # Paper "Progressive Distillation for Fast Sampling of Diffusion Models" defined v-prediction.
        # v_t = alpha_t * epsilon - sigma_t * x_0
        
        # Let's use the formula:
        # alpha_t = sqrt_alphas_cumprod
        # sigma_t = sqrt_one_minus_alphas_cumprod
        
        alpha_t = diffusion.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sigma_t = diffusion.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        v_target = alpha_t * noise - sigma_t * x_0
        
        # If model predicts v directly:
        loss = nn.functional.mse_loss(model_output, v_target)
        
        # Note: The plan says:
        # If predicting x: derive v = (x_pred - z_t) / (1 - t) ... Wait, that formula in plan looks like flow matching or something else.
        # Plan says: "If predicting x: derive v = (x_pred - z_t) / (1 - t)"
        # And "The paper's final algorithm (Eq. 6) uses x-prediction but optimizes the v-loss (velocity)."
        # "Please implement a toggle loss_type='mse_x' vs loss_type='mse_v'."
        
        # Let's stick to the plan's instruction for 'mse_v' if possible, but the formula in the plan:
        # v_pred = (model_output - z_t) / (1 - t)
        # This looks like Rectified Flow or a specific formulation.
        # Standard DDPM v-prediction is different.
        # Given "Back to Basics" paper, they might be using Flow Matching or standard Diffusion.
        # The plan mentions "z_t = t * x_0 + (1-t) * epsilon". This is Rectified Flow / Flow Matching interpolation!
        # NOT standard DDPM (sqrt_alpha ...).
        
        # CRITICAL CHECK:
        # Plan Phase 3:
        # Noise Scheduler: Linear schedule (t in [0, 1])
        # Forward Process: z_t = t * x_0 + (1-t) * epsilon
        # This is Flow Matching (specifically Conditional Flow Matching with optimal transport path, or just linear interpolation).
        # So my Diffusion class above (DDPM) is WRONG for this specific plan.
        # I must follow the plan's "Forward Process".
        
        pass 
        
    return loss

# --- Revised Training Logic for Flow Matching / Plan ---
class FlowMatching:
    def __init__(self, device='cuda'):
        self.device = device

    def q_sample(self, x_0, t, noise=None):
        # z_t = t * x_0 + (1-t) * epsilon
        # t is [B], broadcast to shape
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Reshape t to [B, 1, 1, ...] matching x_0
        # t is [B]
        view_shape = [t.shape[0]] + [1] * (x_0.dim() - 1)
        t_view = t.view(*view_shape)
        # Flow Matching: z_t = t * x_1 + (1 - t) * x_0
        # Here x_1 = data (x_0 in our notation), x_0 = noise.
        # So z_t = t * data + (1 - t) * noise.
        # This matches the paper's "Optimal Transport" path.
        z_t = t_view * x_0 + (1 - t_view) * noise
        return z_t, noise

def train(args):
    device = torch.device(args.device)
    
    # Dataset
    print(f"Loading dataset: {args.dataset_mode}")
    dataloader = get_dataloader(
        args.data_root, 
        mode=args.dataset_mode, 
        batch_size=args.batch_size,
        subset='training',
        mock=args.mock
    )
    
    # Validation dataloader (SpeechCommands validation_list.txt is speaker-disjoint)
    val_dataloader = None
    if not args.mock:
        val_dataloader = get_dataloader(
            args.data_root,
            mode=args.dataset_mode,
            batch_size=args.batch_size,
            subset='validation',
            mock=False
        )
        print(f"Loaded {len(val_dataloader.dataset)} validation samples (speaker-disjoint)")
    
    # Model
    print(f"Creating model: Patch={args.patch_size}, Loss={args.loss_type}, Mode={args.dataset_mode}")
    # Determine input channels and size based on mode
    if args.dataset_mode == 'raw':
        # Raw audio input size is 16384 (1 second at 16kHz usually, or defined by dataset)
        input_size = 16384 
        in_channels = 1
        patch_size = args.patch_size
        is_1d = True
        is_spectrum = False
        freq_bins = None
        time_frames = None
    elif args.dataset_mode == 'spectrogram':
        # Standard 2D spectrogram patchification (like image patches)
        # JiT expects square image input.
        # Our spectrogram is 64x64.
        input_size = 64
        in_channels = 1
        patch_size = args.patch_size  # e.g. 16 or 8
        is_1d = False
        is_spectrum = False
        freq_bins = None
        time_frames = None
    elif args.dataset_mode == 'spectrum_1d':
        # New mode: Patch only along time axis, preserve full frequency resolution
        # Each patch covers [freq_bins, patch_size] -> better for audio
        input_size = 64  # Not used for spectrum mode, but kept for compatibility
        in_channels = 1
        patch_size = args.patch_size
        is_1d = False
        is_spectrum = True
        freq_bins = 64  # Number of mel bins
        time_frames = 64  # Number of time frames
    else:
        raise ValueError(f"Unknown dataset_mode: {args.dataset_mode}")

    # Instantiate JiT
    # We can use the helper functions if args match, or direct instantiation
    # Let's use direct instantiation to be safe with args
    model = JiT(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=8,  # Fixed or arg?
        num_classes=35,
        mlp_ratio=4.0,
        bottleneck_dim=args.hidden_size,  # Default
        in_context_len=0,  # Disable in-context for now unless specified
        is_1d=is_1d,
        is_spectrum=is_spectrum,
        freq_bins=freq_bins if freq_bins else 64,
        time_frames=time_frames if time_frames else 64,
        use_snake=args.use_snake  # Snake activation for audio periodicity
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # FP16 Scaler
    scaler = GradScaler(enabled=args.fp16)
    
    # Scheduler / Flow
    # Plan: z_t = t * x_0 + (1-t) * epsilon
    # This implies t goes from 0 (noise) to 1 (data)?
    # Wait.
    # t=0: z_0 = epsilon.
    # t=1: z_1 = x_0.
    # Yes.
    
    flow = FlowMatching(device)
    
    # Multi-Scale Spectral Loss for raw audio waveforms
    spectral_loss_fn = None
    if args.use_spectral_loss and args.dataset_mode == 'raw':
        spectral_loss_fn = MultiScaleSpectralLoss(
            fft_sizes=[512, 1024, 2048],
            weight=args.spectral_loss_weight,
            use_log=args.spectral_log_scale
        )
        log_msg = "log-scale" if args.spectral_log_scale else "linear-scale"
        print(f"Using Multi-Scale Spectral Loss ({log_msg}) with weight={args.spectral_loss_weight}")
    
    model.train()
    
    start_epoch = 0
    if args.resume_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        state_dict = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        # Assuming checkpoint filename format: model_{name}_ep{epoch}.pth
        # We can parse epoch from filename or just pass it.
        # Let's try to parse.
        try:
            start_epoch = int(args.resume_checkpoint.split('_ep')[-1].split('.pth')[0])
            print(f"Resumed from epoch {start_epoch}")
        except:
            print("Could not parse epoch from checkpoint name, starting from next epoch if possible or 0")
            
    # Logging - with separate loss components
    log_file = os.path.join('checkpoints', f'training_log_{args.experiment_name}.csv')
    os.makedirs('checkpoints', exist_ok=True)
    with open(log_file, 'w') as f:
        # Header includes separate loss components for when spectral loss is enabled
        f.write('epoch,split,step,loss_time,loss_spec,loss_total\n')
            
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss_time = 0.0
        epoch_loss_spec = 0.0
        epoch_loss_total = 0.0
        steps = 0
        for i, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            
            # If spectrogram or spectrum_1d, ensure [B, 1, 64, 64]
            if args.dataset_mode in ['spectrogram', 'spectrum_1d']:
                if x.dim() == 3:  # [B, 64, 64] -> [B, 1, 64, 64]
                    x = x.unsqueeze(1)
            
            # Sample t from logit-normal distribution (JiT-style)
            # z ~ N(P_mean, P_std), then t = sigmoid(z)
            # This concentrates samples near 0 and 1
            z = torch.randn(x.shape[0], device=device) * args.P_std + args.P_mean
            t = torch.sigmoid(z)
            
            # Apply schedule transformation if using cosine schedule
            if args.schedule_type == 'cosine':
                t = cosine_schedule(t)
            
            # CFG: Label dropout (randomly drop labels with probability label_dropout)
            drop_labels = None
            if args.label_dropout > 0:
                drop_labels = torch.rand(x.shape[0], device=device) < args.label_dropout
            
            # Forward process
            noise = torch.randn_like(x) * args.noise_scale
            z_t, _ = flow.q_sample(x, t, noise)
            
            # Model prediction (pass raw t in [0, 1], not scaled)
            with autocast(enabled=args.fp16):
                model_output = model(z_t, t, y, drop_labels=drop_labels)
            
                # Validate output shape matches input shape
                if model_output.shape != x.shape:
                    raise RuntimeError(
                        f"Model output shape {model_output.shape} doesn't match input shape {x.shape}. "
                        f"This usually indicates a patch size issue. "
                        f"Patch size {args.patch_size} must divide evenly into input size "
                        f"({'16384 for raw audio' if args.dataset_mode == 'raw' else '64 for spectrogram'})."
                    )
                
                # Loss
                if args.loss_type == 'epsilon_epsilon_loss':
                    target = noise
                    loss = F.mse_loss(model_output, target)
                elif args.loss_type == 'v_v_loss':
                    # Direct v-prediction: Model predicts v directly
                    v_pred = model_output
                    
                    # Calculate Ground Truth Velocity
                    # v = x - epsilon
                    v_target = x - noise
                    
                    # Compute Loss
                    loss = F.mse_loss(v_pred, v_target)
                elif args.loss_type == 'x_v_loss':
                    # 1. JI-T Philosophy: The Network predicts X (Clean Audio)
                    x_pred = model_output 
        
                    # 2. Reshape t for broadcasting
                    t_view = t.view(*([t.shape[0]] + [1] * (x.dim() - 1)))
                    
                    # 3. CRITICAL: The paper clips the denominator to avoid explosion at t=1
                    # See paper: "clip its denominator (by default, 0.05)"
                    # FP16 FIX: Use a larger min value and compute in float32 for stability
                    denominator = (1 - t_view).float()
                    denominator = torch.clamp(denominator, min=0.1)  # Larger min for FP16 stability
                    
                    # 4. Calculate Velocity Prediction (Formula from Eq. 6)
                    # v_pred = (x_pred - z_t) / (1 - t)
                    # FP16 FIX: Compute in float32 then convert back, and clamp to prevent overflow
                    v_pred = ((x_pred.float() - z_t.float()) / denominator)
                    v_pred = torch.clamp(v_pred, min=-100.0, max=100.0)  # Prevent extreme values
                    v_pred = v_pred.to(x_pred.dtype)
                    
                    # 5. Calculate Ground Truth Velocity
                    # It is cleaner to use (x - noise) than deriving it from z_t
                    # v = x - epsilon [cite: 120]
                    v_target = x - noise
                    
                    # 6. Compute Loss (in float32 for numerical stability)
                    loss_time = F.mse_loss(v_pred.float(), v_target.float())
                    
                    # 7. Add Multi-Scale Spectral Loss for raw audio (if enabled)
                    # This guides the model to preserve high-frequency content
                    loss_spec = torch.tensor(0.0, device=device)
                    if spectral_loss_fn is not None:
                        # Spectral loss on predicted clean audio vs target clean audio
                        loss_spec = spectral_loss_fn(x_pred.float(), x.float())
                        loss = loss_time + loss_spec
                    else:
                        loss = loss_time
                else:
                    raise ValueError(f"Unknown loss type: {args.loss_type}")
            
            # FP16 Backward Pass
            optimizer.zero_grad()
            
            # Check for NaN/Inf in loss and skip batch if found
            if not torch.isfinite(loss):
                print(f"WARNING: Non-finite loss detected ({loss.item()}), skipping batch")
                # Don't call scaler.update() here - no gradient ops were done
                continue
            
            # Scale loss and backward
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            
            # Gradient clipping to prevent exploding gradients (critical for FP16)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step optimizer (will skip if gradients contain inf/nan)
            scaler.step(optimizer)
            scaler.update()
            
            # Track separate losses
            loss_time_val = loss_time.item() if isinstance(loss_time, torch.Tensor) else loss_time
            loss_spec_val = loss_spec.item() if isinstance(loss_spec, torch.Tensor) else loss_spec
            loss_total_val = loss.item()
            
            pbar.set_postfix({
                'loss': loss_total_val,
                'time': loss_time_val,
                'spec': loss_spec_val if spectral_loss_fn else 0.0
            })
            epoch_loss_time += loss_time_val
            epoch_loss_spec += loss_spec_val
            epoch_loss_total += loss_total_val
            steps += 1
            
            # Log step loss (every 100 steps to reduce I/O)
            if i % 100 == 0:
                with open(log_file, 'a') as f:
                    f.write(f'{epoch},train,{i},{loss_time_val:.6f},{loss_spec_val:.6f},{loss_total_val:.6f}\n')
        
        # End of epoch - log average training loss
        if steps > 0:
            avg_train_time = epoch_loss_time / steps
            avg_train_spec = epoch_loss_spec / steps
            avg_train_total = epoch_loss_total / steps
            with open(log_file, 'a') as f:
                f.write(f'{epoch},train_avg,-1,{avg_train_time:.6f},{avg_train_spec:.6f},{avg_train_total:.6f}\n')
            print(f"\nEpoch {epoch+1} Train - Total: {avg_train_total:.4f}, Time: {avg_train_time:.4f}, Spec: {avg_train_spec:.4f}")
        
        # Validation loop
        if val_dataloader is not None:
            model.eval()
            val_loss_time = 0.0
            val_loss_spec = 0.0
            val_loss_total = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for x, y in tqdm(val_dataloader, desc=f"Validation", leave=False):
                    x = x.to(device)
                    y = y.to(device)
                    
                    # If spectrogram or spectrum_1d, ensure [B, 1, 64, 64]
                    if args.dataset_mode in ['spectrogram', 'spectrum_1d']:
                        if x.dim() == 3:
                            x = x.unsqueeze(1)
                    
                    # Sample t (use same distribution as training for fair comparison)
                    z_val = torch.randn(x.shape[0], device=device) * args.P_std + args.P_mean
                    t = torch.sigmoid(z_val)
                    if args.schedule_type == 'cosine':
                        t = cosine_schedule(t)
                    
                    # Forward process
                    noise = torch.randn_like(x) * args.noise_scale
                    z_t, _ = flow.q_sample(x, t, noise)
                    
                    # Model prediction (no label dropout for validation)
                    with autocast(enabled=args.fp16):
                        model_output = model(z_t, t, y)
                    
                    # Compute validation loss (x_v_loss style for x-prediction models)
                    if args.loss_type == 'x_v_loss':
                        x_pred = model_output
                        t_view = t.view(*([t.shape[0]] + [1] * (x.dim() - 1)))
                        denominator = torch.clamp((1 - t_view).float(), min=0.1)
                        v_pred = torch.clamp((x_pred.float() - z_t.float()) / denominator, -100.0, 100.0)
                        v_target = x - noise
                        batch_loss_time = F.mse_loss(v_pred.float(), v_target.float())
                        
                        batch_loss_spec = torch.tensor(0.0, device=device)
                        if spectral_loss_fn is not None:
                            batch_loss_spec = spectral_loss_fn(x_pred.float(), x.float())
                        batch_loss_total = batch_loss_time + batch_loss_spec
                    else:
                        # For other loss types, use simple MSE
                        batch_loss_time = F.mse_loss(model_output, x if args.loss_type != 'epsilon_epsilon_loss' else noise)
                        batch_loss_spec = torch.tensor(0.0, device=device)
                        batch_loss_total = batch_loss_time
                    
                    val_loss_time += batch_loss_time.item()
                    val_loss_spec += batch_loss_spec.item()
                    val_loss_total += batch_loss_total.item()
                    val_steps += 1
            
            if val_steps > 0:
                avg_val_time = val_loss_time / val_steps
                avg_val_spec = val_loss_spec / val_steps
                avg_val_total = val_loss_total / val_steps
                with open(log_file, 'a') as f:
                    f.write(f'{epoch},val,-1,{avg_val_time:.6f},{avg_val_spec:.6f},{avg_val_total:.6f}\n')
                print(f"Epoch {epoch+1} Val   - Total: {avg_val_total:.4f}, Time: {avg_val_time:.4f}, Spec: {avg_val_spec:.4f}")
            
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == 0:
            torch.save(model.state_dict(), f'checkpoints/model_{args.experiment_name}_ep{epoch+1}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--dataset_mode', type=str, default='raw', choices=['raw', 'spectrogram', 'spectrum_1d'])
    parser.add_argument('--loss_type', type=str, default='x', choices=['epsilon_epsilon_loss', 'v_v_loss', 'x_v_loss'])
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mock', action='store_true', help='Use mock dataset')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Model Capacity
    parser.add_argument('--hidden_size', type=int, default=512, help='Model hidden size')
    parser.add_argument('--depth', type=int, default=12, help='Model depth (number of layers)')
    
    # JiT-style parameters
    parser.add_argument('--P_mean', type=float, default=-0.8, help='Logit-normal mean for time sampling')
    parser.add_argument('--P_std', type=float, default=0.8, help='Logit-normal std for time sampling')
    parser.add_argument('--noise_scale', type=float, default=1.0, help='Noise scale factor')
    parser.add_argument('--t_eps', type=float, default=5e-2, help='Minimum t value to avoid division by zero')
    parser.add_argument('--fp16', action='store_true', default=True, help='Enable mixed precision training')
    
    # Schedule type
    parser.add_argument('--schedule_type', type=str, default='linear', choices=['linear', 'cosine'],
                        help='Noise schedule type: linear (default) or cosine (focuses on high-fidelity regime)')
    
    # Classifier-Free Guidance (CFG)
    parser.add_argument('--label_dropout', type=float, default=0.0,
                        help='Label dropout probability for CFG training (default: 0.0, recommended: 0.1)')
    
    # Audio-specific improvements
    parser.add_argument('--use_snake', action='store_true', default=False, 
                        help='Use Snake activation for better audio periodicity (recommended for raw audio)')
    parser.add_argument('--use_spectral_loss', action='store_true', default=False,
                        help='Add multi-scale spectral loss for raw audio (guides high-frequency preservation)')
    parser.add_argument('--spectral_loss_weight', type=float, default=0.1,
                        help='Weight for spectral loss term (default: 0.1)')
    parser.add_argument('--spectral_log_scale', action='store_true', default=False,
                        help='Use log-scale spectral loss (balances training across frequencies)')
    
    args = parser.parse_args()
    train(args)

