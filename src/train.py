import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import get_dataloader
from model import JiT, JiT_models

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
    
    # Model
    print(f"Creating model: Patch={args.patch_size}, Loss={args.loss_type}")
    # Determine input channels and size based on mode
    if args.dataset_mode == 'raw':
        # Raw audio not fully supported yet with JiT (needs 2D reshaping)
        # For now, let's assume we might use it but it will likely fail or need reshaping
        input_size = 128 # Dummy
        in_channels = 1
        patch_size = args.patch_size
    else: # spectrogram
        # JiT expects square image input.
        # Our spectrogram is 64x64.
        input_size = 64
        in_channels = 1
        patch_size = args.patch_size # e.g. 16 or 8

    # Instantiate JiT
    # We can use the helper functions if args match, or direct instantiation
    # Let's use direct instantiation to be safe with args
    model = JiT(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=8, # Fixed or arg?
        num_classes=35,
        mlp_ratio=4.0,
        bottleneck_dim=args.hidden_size, # Default
        in_context_len=0, # Disable in-context for now unless specified
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Scheduler / Flow
    # Plan: z_t = t * x_0 + (1-t) * epsilon
    # This implies t goes from 0 (noise) to 1 (data)?
    # Wait.
    # t=0: z_0 = epsilon.
    # t=1: z_1 = x_0.
    # Yes.
    
    flow = FlowMatching(device)
    
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
            
    # Logging
    log_file = os.path.join('checkpoints', f'training_log_{args.experiment_name}.csv')
    with open(log_file, 'w') as f:
        f.write('epoch,step,loss\n')
            
    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        steps = 0
        for i, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            
            # If spectrogram, flatten
            # If spectrogram, ensure [B, 1, 64, 64]
            if args.dataset_mode == 'spectrogram':
                if x.dim() == 3: # [B, 64, 64] -> [B, 1, 64, 64]
                    x = x.unsqueeze(1)
                # If flattened, reshape? Dataset returns [64, 64] usually (after squeeze)
                # Dataset code: returns spec [64, 64] (if squeezed) or [1, 64, 64]
                # Let's check dataset.py again. It returns `spec` which is [64, 64] after squeeze.
                # So x is [B, 64, 64].
                # We need [B, 1, 64, 64].
                if x.dim() == 3:
                    x = x.unsqueeze(1)
            
            # Sample t from logit-normal distribution (JiT-style)
            # z ~ N(P_mean, P_std), then t = sigmoid(z)
            # This concentrates samples near 0 and 1
            z = torch.randn(x.shape[0], device=device) * args.P_std + args.P_mean
            t = torch.sigmoid(z)
            
            # Forward process
            noise = torch.randn_like(x) * args.noise_scale
            z_t, _ = flow.q_sample(x, t, noise)
            
            # Model prediction (pass raw t in [0, 1], not scaled)
            model_output = model(z_t, t, y)
            
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
                denominator = 1 - t_view
                denominator = torch.clamp(denominator, min=0.05) 
                
                # 4. Calculate Velocity Prediction (Formula from Eq. 6)
                # v_pred = (x_pred - z_t) / (1 - t)
                v_pred = (x_pred - z_t) / denominator
                
                # 5. Calculate Ground Truth Velocity
                # It is cleaner to use (x - noise) than deriving it from z_t
                # v = x - epsilon [cite: 120]
                v_target = x - noise
                
                # 6. Compute Loss
                loss = F.mse_loss(v_pred, v_target)
            else:
                raise ValueError(f"Unknown loss type: {args.loss_type}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': loss.item()})
            epoch_loss += loss.item()
            steps += 1
            
            # Log step loss
            with open(log_file, 'a') as f:
                f.write(f'{epoch},{i},{loss.item()}\n')
            
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/model_{args.experiment_name}_ep{epoch+1}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--dataset_mode', type=str, default='raw', choices=['raw', 'spectrogram'])
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
    
    args = parser.parse_args()
    train(args)
