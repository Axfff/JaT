import os
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from dataset import get_dataloader
from model import JiT, JiT_models


# --- Exponential Moving Average ---
class EMA:
    """
    Exponential Moving Average for model parameters.
    
    Maintains a shadow copy of model parameters that is updated with a decay
    factor after each optimizer step. EMA weights typically produce smoother
    and more stable results for inference.
    
    IMPORTANT: Parameters are stored as a dict keyed by name to ensure correct
    matching during checkpoint load/save, avoiding misalignment bugs.
    
    Args:
        model: The model to track
        decay: EMA decay rate (default: 0.9999, higher = slower update)
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        # Store parameters as dict keyed by name for exact matching on load
        self.params = {}
        for name, p in model.named_parameters():
            self.params[name] = p.clone().detach()
            self.params[name].requires_grad = False
    
    @torch.no_grad()
    def update(self, model):
        """Update EMA parameters with current model parameters."""
        for name, model_p in model.named_parameters():
            if name in self.params:
                self.params[name].mul_(self.decay).add_(model_p, alpha=1 - self.decay)
    
    def state_dict(self):
        """Return state dict for checkpointing."""
        return {'params': self.params, 'decay': self.decay}
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.params = state_dict['params']
        self.decay = state_dict['decay']
    
    def apply_to(self, model):
        """Copy EMA parameters to model (for inference)."""
        for name, model_p in model.named_parameters():
            if name in self.params:
                model_p.data.copy_(self.params[name])


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


# --- Min-SNR Weighting ---
def min_snr_weight(t, gamma=5.0, eps=1e-6):
    """
    Compute min-SNR weighting for Flow Matching loss.
    
    From "Efficient Diffusion Training via Min-SNR Weighting Strategy":
    - Reweights loss at each timestep based on signal-to-noise ratio
    - Stabilizes training and speeds up convergence
    - Particularly helpful for v-prediction
    
    For Flow Matching with z_t = t * x_0 + (1-t) * noise:
    - Signal coefficient = t (contribution from clean data)
    - Noise coefficient = (1-t) (contribution from noise)
    - SNR(t) = t² / (1-t)²
    
    Weight = min(SNR, gamma) / SNR
    - When SNR < gamma: weight = 1 (no change)
    - When SNR > gamma: weight = gamma / SNR (reduces weight)
    
    This downweights the "easy" timesteps (high SNR, near t=1) and
    focuses training on the difficult ones (low SNR, near t=0).
    
    Args:
        t: Timesteps in [0, 1], shape [B]
        gamma: SNR clipping threshold (default: 5.0, paper recommends 5)
        eps: Small value to prevent division by zero
        
    Returns:
        Weights of shape [B], ready for broadcasting
    """
    # Clamp t away from boundaries to prevent extreme SNR values
    t_clamped = torch.clamp(t, min=eps, max=1 - eps)
    
    # SNR = (signal coefficient)² / (noise coefficient)² = t² / (1-t)²
    snr = (t_clamped ** 2) / ((1 - t_clamped) ** 2)
    
    # min-SNR weight: min(SNR, gamma) / SNR
    # = 1 when SNR <= gamma (no change for hard timesteps)
    # = gamma / SNR when SNR > gamma (downweight easy timesteps)
    weight = torch.clamp(snr, max=gamma) / snr
    
    return weight


# --- SNR-Adaptive Loss Weighting ---
def snr_adaptive_weights(t, spec_weight_base=1.0, time_weight_base=1.0, 
                         t_spec_peak=0.3, t_time_peak=0.8, slope=10.0, eps=1e-6):
    """
    Compute SNR-adaptive weights for spectral and time-domain losses.
    
    Intuition:
    - Low SNR (t near 0, very noisy): Waveform MSE is mostly noise; spectral 
      envelope & energy are meaningful → spectral loss should dominate.
    - Mid SNR: Both are useful for learning correct envelope and waveform.
    - High SNR (t near 1, mostly clean): Precise phase, micro-structure, and
      killing residual noise matter → time-domain MSE should dominate.
    
    Implementation:
    - Spectral weight peaks at low t (noisy), decreases towards t=1
    - Time weight peaks at high t (clean), decreases towards t=0
    - Uses smooth sigmoid transitions for stable gradients
    
    For Flow Matching: z_t = t * x_0 + (1-t) * noise
    - t=0: pure noise (low SNR) → spec dominates
    - t=1: pure signal (high SNR) → time dominates
    
    Args:
        t: Timesteps in [0, 1], shape [B]
        spec_weight_base: Base weight for spectral loss (default: 1.0)
        time_weight_base: Base weight for time loss (default: 1.0)
        t_spec_peak: t value where spectral weight is at 50% (default: 0.3)
        t_time_peak: t value where time weight is at 50% (default: 0.8)
        slope: Controls sharpness of transition (default: 10.0, higher = sharper)
        eps: Small value for numerical stability
        
    Returns:
        Tuple of (time_weight, spec_weight), each of shape [B]
    """
    # Clamp t for numerical stability
    t_clamped = torch.clamp(t, min=eps, max=1 - eps)
    
    # Spectral weight: high at low t (noisy), low at high t (clean)
    # sigmoid((t_spec_peak - t) * slope) → 1 when t << t_spec_peak, 0 when t >> t_spec_peak
    spec_weight = spec_weight_base * torch.sigmoid((t_spec_peak - t_clamped) * slope)
    
    # Time weight: low at low t (noisy), high at high t (clean)
    # sigmoid((t - t_time_peak) * slope) → 0 when t << t_time_peak, 1 when t >> t_time_peak
    # But we want time loss to still contribute at mid-t, so use different formulation:
    # Time weight ramps up as t increases
    time_weight = time_weight_base * torch.sigmoid((t_clamped - t_time_peak) * slope)
    
    # Ensure minimum weights so neither loss ever completely vanishes
    # This keeps gradients flowing from both terms throughout training
    min_weight = 0.1
    spec_weight = spec_weight + min_weight * spec_weight_base
    time_weight = time_weight + min_weight * time_weight_base
    
    return time_weight, spec_weight


class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-scale spectral loss to improve audio fidelity.
    
    Computes L1 loss between STFT magnitudes at multiple window sizes.
    This guides the model to preserve high-frequency content that MSE 
    on waveforms tends to blur out.
    
    Noise-floor-aware mode: Adds an extra term that penalizes predicted magnitude
    in regions where ground truth is near silent. This explicitly kills hiss in
    gaps between syllables and removes "electronic" buzz in nominally silent bins.
    
    Args:
        fft_sizes: List of FFT window sizes to use (default: [512, 1024, 2048])
        hop_ratio: Hop length as ratio of FFT size (default: 0.25)
        weight: Weight for the spectral loss term (default: 0.1)
        use_log: If True, compute loss on log-magnitude (reduces penalty on loud signals)
        use_noise_floor: If True, add noise-floor-aware penalty (default: False)
        noise_floor_weight: Weight for the noise-floor term (default: 0.5)
        noise_floor_thresh_ratio: Threshold as fraction of mean magnitude (default: 0.1)
        noise_floor_sharpness: Sharpness of sigmoid mask transition (default: 10.0)
    """
    def __init__(self, fft_sizes=[64, 128, 256, 512, 1024, 2048], hop_ratio=0.25, weight=0.1, use_log=False,
                 use_noise_floor=False, noise_floor_weight=0.5, noise_floor_thresh_ratio=0.1, noise_floor_sharpness=10.0):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_ratio = hop_ratio
        self.weight = weight
        self.use_log = use_log
        
        # Noise-floor-aware parameters
        self.use_noise_floor = use_noise_floor
        self.noise_floor_weight = noise_floor_weight
        self.noise_floor_thresh_ratio = noise_floor_thresh_ratio
        self.noise_floor_sharpness = noise_floor_sharpness
        
        # Adaptive epsilon: smaller FFT sizes need larger eps to prevent gradient explosion
        # This is because smaller FFTs have less energy per bin (more spread out)
        # eps is chosen so max gradient (1/eps) stays reasonable:
        # - FFT 64-128: eps=1e-2 -> max grad 100
        # - FFT 256: eps=5e-3 -> max grad 200  
        # - FFT 512+: eps=1e-3 -> max grad 1000
        self.eps_map = {}
        for fft_size in fft_sizes:
            if fft_size <= 128:
                self.eps_map[fft_size] = 1e-2
            elif fft_size <= 256:
                self.eps_map[fft_size] = 5e-3
            else:
                self.eps_map[fft_size] = 1e-3
        
        # Pre-register windows as buffers to avoid recreation every forward pass
        # This also ensures they move to the correct device with the module
        for fft_size in fft_sizes:
            self.register_buffer(f'window_{fft_size}', torch.hann_window(fft_size))
        
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
        
        # CRITICAL: Force float32 for STFT computation
        # STFT is numerically unstable in FP16 and can produce NaN/Inf
        pred_wave = pred_wave.float()
        target_wave = target_wave.float()
        
        loss_base = 0.0
        loss_noise_floor = 0.0
        
        for fft_size in self.fft_sizes:
            hop_length = int(fft_size * self.hop_ratio)
            window = getattr(self, f'window_{fft_size}')
            eps = self.eps_map[fft_size]
            
            # Compute STFT magnitude (in float32)
            pred_spec = torch.stft(
                pred_wave, 
                n_fft=fft_size, 
                hop_length=hop_length,
                window=window,
                return_complex=True
            ).abs()
            
            target_spec = torch.stft(
                target_wave, 
                n_fft=fft_size, 
                hop_length=hop_length,
                window=window,
                return_complex=True
            ).abs()
            
            # Normalize by sqrt(fft_size) to make magnitudes comparable across scales
            # This compensates for the fact that smaller FFTs distribute energy differently
            norm_factor = math.sqrt(fft_size / 2048.0)  # Normalize relative to largest FFT
            pred_spec = pred_spec / norm_factor
            target_spec = target_spec / norm_factor
            
            # Store original (linear) target_spec for noise-floor computation
            target_spec_linear = target_spec
            pred_spec_linear = pred_spec
            
            if self.use_log:
                # Log-scale spectral loss: reduces penalty on loud signals,
                # balances training across frequencies
                # 
                # SAFETY MEASURES:
                # 1. Use adaptive eps per FFT size (smaller FFT = larger eps)
                # 2. Clamp output to reasonable range
                pred_spec = torch.log(torch.clamp(pred_spec, min=eps))
                target_spec = torch.log(torch.clamp(target_spec, min=eps))
                
                # Clamp log output to prevent extreme values that could overflow
                pred_spec = torch.clamp(pred_spec, min=-10.0, max=10.0)
                target_spec = torch.clamp(target_spec, min=-10.0, max=10.0)
            
            # L1 loss on magnitude (ignores phase misalignment)
            loss_base += F.l1_loss(pred_spec, target_spec)
            
            # Noise-floor-aware penalty: penalize predicted magnitude where GT is quiet
            # This kills hiss in silent gaps and electronic buzz in nominally silent bins
            if self.use_noise_floor:
                # Compute threshold based on per-sample mean magnitude
                # thresh = noise_floor_thresh_ratio * mean(target_spec_linear)
                # Shape: [B, 1, 1] for broadcasting
                thresh = self.noise_floor_thresh_ratio * target_spec_linear.mean(dim=(1, 2), keepdim=True)
                
                # Smooth sigmoid mask: 1 where GT is quiet, 0 where GT is loud
                # silence_mask = sigmoid((thresh - spec_gt) * k)
                # When spec_gt << thresh: positive input -> sigmoid ~1 (quiet region)
                # When spec_gt >> thresh: negative input -> sigmoid ~0 (loud region)
                silence_mask = torch.sigmoid((thresh - target_spec_linear) * self.noise_floor_sharpness)
                
                # Penalize predicted magnitude only in quiet regions
                # Higher pred_spec_linear in silent regions = higher penalty
                loss_noise_floor += (silence_mask * pred_spec_linear).mean()
        
        # Average over number of scales
        loss_base = loss_base / len(self.fft_sizes)
        
        # Combine losses
        if self.use_noise_floor:
            loss_noise_floor = loss_noise_floor / len(self.fft_sizes)
            total_loss = loss_base + self.noise_floor_weight * loss_noise_floor
        else:
            total_loss = loss_base
        
        # Apply weight and return
        return self.weight * total_loss


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
        mock=args.mock,
        freq_bins=args.freq_bins,
        time_frames=args.time_frames
    )
    
    # Validation dataloader (SpeechCommands validation_list.txt is speaker-disjoint)
    val_dataloader = None
    if not args.mock:
        val_dataloader = get_dataloader(
            args.data_root,
            mode=args.dataset_mode,
            batch_size=args.batch_size,
            subset='validation',
            mock=False,
            freq_bins=args.freq_bins,
            time_frames=args.time_frames
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
        input_size = args.freq_bins  # Use freq_bins as input_size for patch calculations
        in_channels = 1
        patch_size = args.patch_size  # e.g. 16 or 8
        is_1d = False
        is_spectrum = False
        freq_bins = args.freq_bins
        time_frames = args.time_frames
    elif args.dataset_mode == 'spectrum_1d':
        # New mode: Patch only along time axis, preserve full frequency resolution
        # Each patch covers [freq_bins, patch_size] -> better for audio
        input_size = args.freq_bins  # Not used for spectrum mode, but kept for compatibility
        in_channels = 1
        patch_size = args.patch_size
        is_1d = False
        is_spectrum = True
        freq_bins = args.freq_bins
        time_frames = args.time_frames
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
        freq_bins=freq_bins if freq_bins else args.freq_bins,
        time_frames=time_frames if time_frames else args.time_frames,
        use_snake=args.use_snake,  # Snake activation for audio periodicity
        hop_size=args.hop_size if args.dataset_mode == 'raw' else None  # Overlap for raw audio only
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Learning Rate Scheduler
    scheduler = None
    if args.lr_schedule == 'cosine':
        # Calculate total training steps/epochs
        total_epochs = args.epochs
        
        if args.warmup_epochs > 0:
            # Linear warmup + Cosine annealing
            warmup_scheduler = LinearLR(
                optimizer, 
                start_factor=0.01,  # Start at 1% of lr
                end_factor=1.0,
                total_iters=args.warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - args.warmup_epochs,
                eta_min=args.lr_min
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[args.warmup_epochs]
            )
            print(f"Using Cosine LR schedule with {args.warmup_epochs} warmup epochs, min_lr={args.lr_min}")
        else:
            # Pure cosine annealing
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=args.lr_min
            )
            print(f"Using Cosine LR schedule (no warmup), min_lr={args.lr_min}")
    
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
            fft_sizes=[64, 128, 256, 512, 1024, 2048],
            weight=args.spectral_loss_weight,
            use_log=args.spectral_log_scale,
            use_noise_floor=args.use_noise_floor,
            noise_floor_weight=args.noise_floor_weight,
            noise_floor_thresh_ratio=args.noise_floor_thresh,
            noise_floor_sharpness=args.noise_floor_sharpness
        ).to(device)
        log_msg = "log-scale" if args.spectral_log_scale else "linear-scale"
        nf_msg = f", noise-floor weight={args.noise_floor_weight}" if args.use_noise_floor else ""
        print(f"Using Multi-Scale Spectral Loss ({log_msg}) with weight={args.spectral_loss_weight}{nf_msg}")
        if args.use_snr_adaptive:
            print(f"  SNR-adaptive weighting: spec peaks at t<{args.snr_adaptive_t_spec}, time peaks at t>{args.snr_adaptive_t_time}")
    
    # EMA (Exponential Moving Average)
    ema = None
    if args.ema_decay > 0:
        ema = EMA(model, decay=args.ema_decay)
        print(f"Using EMA with decay={args.ema_decay}")
    
    if args.min_snr_gamma > 0:
        print(f"Using Min-SNR weighting with gamma={args.min_snr_gamma}")
    
    model.train()
    
    start_epoch = 0
    if args.resume_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        # Support both new format (dict with 'model' and 'ema') and legacy format (just state_dict)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            if 'ema' in checkpoint and ema is not None:
                ema.load_state_dict(checkpoint['ema'])
                print("Restored EMA state from checkpoint")
        else:
            # Legacy format: checkpoint is just state_dict
            model.load_state_dict(checkpoint)
        # Parse epoch from filename
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
        # Clear GPU cache to prevent fragmentation (especially after validation)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss_time = 0.0
        epoch_loss_spec = 0.0
        epoch_loss_total = 0.0
        steps = 0
        for i, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            
            # If spectrogram or spectrum_1d, ensure [B, 1, freq_bins, time_frames]
            if args.dataset_mode in ['spectrogram', 'spectrum_1d']:
                if x.dim() == 3:  # [B, F, T] -> [B, 1, F, T]
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
                        f"({'16384 for raw audio' if args.dataset_mode == 'raw' else f'{args.freq_bins}x{args.time_frames} for spectrogram'})."
                    )
                
                # Compute min-SNR weight if enabled (gamma > 0)
                snr_weight = None
                if args.min_snr_gamma > 0:
                    snr_weight = min_snr_weight(t, gamma=args.min_snr_gamma)
                    # Reshape for broadcasting: [B] -> [B, 1, 1, ...]
                    snr_weight = snr_weight.view(*([t.shape[0]] + [1] * (x.dim() - 1)))
                
                # Loss
                if args.loss_type == 'epsilon_epsilon_loss':
                    target = noise
                    if snr_weight is not None:
                        # Weighted MSE: mean((pred - target)^2 * weight)
                        loss_time = (F.mse_loss(model_output, target, reduction='none') * snr_weight).mean()
                    else:
                        loss_time = F.mse_loss(model_output, target)
                    loss_spec = torch.tensor(0.0, device=device)
                    loss = loss_time
                elif args.loss_type == 'v_v_loss':
                    # Direct v-prediction: Model predicts v directly
                    v_pred = model_output
                    
                    # Calculate Ground Truth Velocity
                    # v = x - epsilon
                    v_target = x - noise
                    
                    # Compute Loss (with optional min-SNR weighting)
                    if snr_weight is not None:
                        loss_time = (F.mse_loss(v_pred, v_target, reduction='none') * snr_weight).mean()
                    else:
                        loss_time = F.mse_loss(v_pred, v_target)
                    loss_spec = torch.tensor(0.0, device=device)
                    loss = loss_time
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
                    
                    # 6. Compute Loss (in float32 for numerical stability, with optional min-SNR)
                    if snr_weight is not None:
                        loss_time = (F.mse_loss(v_pred.float(), v_target.float(), reduction='none') * snr_weight).mean()
                    else:
                        loss_time = F.mse_loss(v_pred.float(), v_target.float())
                    
                    # 7. Add Multi-Scale Spectral Loss for raw audio (if enabled)
                    # This guides the model to preserve high-frequency content
                    loss_spec = torch.tensor(0.0, device=device)
                    if spectral_loss_fn is not None:
                        # Spectral loss on predicted clean audio vs target clean audio
                        loss_spec = spectral_loss_fn(x_pred.float(), x.float())
                        
                        # SNR-adaptive loss weighting: 
                        # - At low t (noisy): spectral loss dominates (coarse envelope matters)
                        # - At high t (clean): time-domain loss dominates (precise phase matters)
                        if args.use_snr_adaptive:
                            time_weight, spec_weight = snr_adaptive_weights(
                                t,
                                spec_weight_base=args.snr_adaptive_spec_base,
                                time_weight_base=args.snr_adaptive_time_base,
                                t_spec_peak=args.snr_adaptive_t_spec,
                                t_time_peak=args.snr_adaptive_t_time,
                                slope=args.snr_adaptive_slope
                            )
                            # Reshape for broadcasting: [B] -> [B, 1, ...]
                            time_weight = time_weight.view(*([t.shape[0]] + [1] * (x.dim() - 1)))
                            spec_weight = spec_weight.mean()  # Average across batch for consistent spectral weight
                            
                            loss = time_weight.mean() * loss_time + spec_weight * loss_spec
                        else:
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
            
            # Update EMA
            if ema is not None:
                ema.update(model)
            
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
            
            # Log step loss (every 10 steps to reduce I/O)
            if i % 10 == 0:
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
        
        # Step learning rate scheduler (per epoch)
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.2e}")
        
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
                    
                    # If spectrogram or spectrum_1d, ensure [B, 1, freq_bins, time_frames]
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
                    # Apply min-SNR weighting to match training (for comparable metrics)
                    snr_weight = None
                    if args.min_snr_gamma > 0:
                        snr_weight = min_snr_weight(t, gamma=args.min_snr_gamma)
                        snr_weight = snr_weight.view(*([t.shape[0]] + [1] * (x.dim() - 1)))
                    
                    if args.loss_type == 'x_v_loss':
                        x_pred = model_output
                        t_view = t.view(*([t.shape[0]] + [1] * (x.dim() - 1)))
                        denominator = torch.clamp((1 - t_view).float(), min=0.1)
                        v_pred = torch.clamp((x_pred.float() - z_t.float()) / denominator, -100.0, 100.0)
                        v_target = x - noise
                        
                        if snr_weight is not None:
                            batch_loss_time = (F.mse_loss(v_pred.float(), v_target.float(), reduction='none') * snr_weight).mean()
                        else:
                            batch_loss_time = F.mse_loss(v_pred.float(), v_target.float())
                        
                        batch_loss_spec = torch.tensor(0.0, device=device)
                        if spectral_loss_fn is not None:
                            batch_loss_spec = spectral_loss_fn(x_pred.float(), x.float())
                            
                            # Apply SNR-adaptive weighting to match training
                            if args.use_snr_adaptive:
                                time_weight, spec_weight = snr_adaptive_weights(
                                    t,
                                    spec_weight_base=args.snr_adaptive_spec_base,
                                    time_weight_base=args.snr_adaptive_time_base,
                                    t_spec_peak=args.snr_adaptive_t_spec,
                                    t_time_peak=args.snr_adaptive_t_time,
                                    slope=args.snr_adaptive_slope
                                )
                                time_weight = time_weight.view(*([t.shape[0]] + [1] * (x.dim() - 1)))
                                spec_weight = spec_weight.mean()
                                
                                batch_loss_total = time_weight.mean() * batch_loss_time + spec_weight * batch_loss_spec
                            else:
                                batch_loss_total = batch_loss_time + batch_loss_spec
                        else:
                            batch_loss_total = batch_loss_time
                    elif args.loss_type == 'v_v_loss':
                        v_pred = model_output
                        v_target = x - noise
                        if snr_weight is not None:
                            batch_loss_time = (F.mse_loss(v_pred, v_target, reduction='none') * snr_weight).mean()
                        else:
                            batch_loss_time = F.mse_loss(v_pred, v_target)
                        batch_loss_spec = torch.tensor(0.0, device=device)
                        batch_loss_total = batch_loss_time
                    else:
                        # epsilon_epsilon_loss
                        target = noise
                        if snr_weight is not None:
                            batch_loss_time = (F.mse_loss(model_output, target, reduction='none') * snr_weight).mean()
                        else:
                            batch_loss_time = F.mse_loss(model_output, target)
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
            
        # Save checkpoint (new format with EMA support)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            checkpoint = {'model': model.state_dict()}
            if ema is not None:
                checkpoint['ema'] = ema.state_dict()
            torch.save(checkpoint, f'checkpoints/model_{args.experiment_name}_ep{epoch+1}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--dataset_mode', type=str, default='raw', choices=['raw', 'spectrogram', 'spectrum_1d'])
    parser.add_argument('--loss_type', type=str, default='x', choices=['epsilon_epsilon_loss', 'v_v_loss', 'x_v_loss'])
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--hop_size', type=int, default=None,
                        help='Hop size for overlapping patches (raw mode only). Default: same as patch_size (no overlap)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # Learning Rate Schedule
    parser.add_argument('--lr_schedule', type=str, default='constant', choices=['constant', 'cosine'],
                        help='Learning rate schedule: constant (default) or cosine')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate for cosine schedule (default: 1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of warmup epochs for cosine schedule (default: 0)')
    
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mock', action='store_true', help='Use mock dataset')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Model Capacity
    parser.add_argument('--hidden_size', type=int, default=512, help='Model hidden size')
    parser.add_argument('--depth', type=int, default=12, help='Model depth (number of layers)')
    
    # Spectrum resolution (for spectrogram and spectrum_1d modes)
    parser.add_argument('--freq_bins', type=int, default=64,
                        help='Number of frequency bins for spectrogram/spectrum_1d (default: 64)')
    parser.add_argument('--time_frames', type=int, default=64,
                        help='Number of time frames for spectrogram/spectrum_1d (default: 64)')
    
    # JiT-style parameters
    parser.add_argument('--P_mean', type=float, default=-0.8, help='Logit-normal mean for time sampling')
    parser.add_argument('--P_std', type=float, default=0.8, help='Logit-normal std for time sampling')
    parser.add_argument('--noise_scale', type=float, default=1.0, help='Noise scale factor')
    parser.add_argument('--t_eps', type=float, default=5e-2, help='Minimum t value to avoid division by zero')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false', help='Disable mixed precision training (use full FP32)')
    parser.set_defaults(fp16=True)
    
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
    
    # Noise-floor-aware spectral loss (kills hiss in silent gaps)
    parser.add_argument('--use_noise_floor', action='store_true', default=False,
                        help='Add noise-floor penalty to spectral loss (penalizes energy in silent regions)')
    parser.add_argument('--noise_floor_weight', type=float, default=0.5,
                        help='Weight for noise-floor penalty (default: 0.5)')
    parser.add_argument('--noise_floor_thresh', type=float, default=0.1,
                        help='Threshold as fraction of mean magnitude (default: 0.1 = 10%% of mean)')
    parser.add_argument('--noise_floor_sharpness', type=float, default=10.0,
                        help='Sharpness of sigmoid mask transition (default: 10.0, higher = sharper)')
    
    # SNR-adaptive loss weighting (spec dominates at low SNR, time at high SNR)
    parser.add_argument('--use_snr_adaptive', action='store_true', default=False,
                        help='Use SNR-adaptive weighting for spectral vs time-domain loss')
    parser.add_argument('--snr_adaptive_spec_base', type=float, default=1.0,
                        help='Base weight for spectral loss in SNR-adaptive mode (default: 1.0)')
    parser.add_argument('--snr_adaptive_time_base', type=float, default=1.0,
                        help='Base weight for time loss in SNR-adaptive mode (default: 1.0)')
    parser.add_argument('--snr_adaptive_t_spec', type=float, default=0.3,
                        help='t value where spectral weight is at 50%% (default: 0.3, lower = spec active longer)')
    parser.add_argument('--snr_adaptive_t_time', type=float, default=0.8,
                        help='t value where time weight is at 50%% (default: 0.8, higher = time active later)')
    parser.add_argument('--snr_adaptive_slope', type=float, default=10.0,
                        help='Sharpness of weight transitions (default: 10.0, higher = sharper)')

    # EMA (Exponential Moving Average)
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay rate (0 to disable, default: 0.9999)')
    
    # Min-SNR Weighting
    parser.add_argument('--min_snr_gamma', type=float, default=0.0,
                        help='Min-SNR gamma for loss weighting (0 to disable, paper recommends 5.0)')
    
    args = parser.parse_args()
    train(args)

