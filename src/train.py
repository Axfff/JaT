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
from model import JustAudioTransformer

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
        
        t_view = t.view(-1, 1, 1)
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
        input_size = 16384
        in_channels = 1
    else: # spectrogram
        input_size = 64 # We treat 64x64 as sequence length 64, channels 64? 
        # Or flatten?
        # DiT usually takes (N, C, H, W) -> Patchify -> (N, T, D).
        # Our model is 1D.
        # If spectrogram is 64x64.
        # Option 1: Treat as 1 channel, 4096 length.
        # Option 2: Treat as 64 channels, 64 length.
        # Plan says: "Spectrogram ... 64x64".
        # Plan says: "Config A ... Patch=512", "Config B ... Patch=64".
        # If we use 1D model on 2D data, we should flatten or treat one dim as channels.
        # Let's treat it as 1 channel, flattened 4096?
        # Or 64 channels, length 64?
        # The plan says "Input ... shape [batch, 1, 16000]".
        # For spectrogram, it says "Control: Spectrogram Input".
        # Let's assume we flatten it to [B, 1, 4096] or similar.
        # 64x64 = 4096.
        # If patch_size=64, we get 64 tokens.
        input_size = 4096
        in_channels = 1
    
    model = JustAudioTransformer(
        input_size=input_size,
        patch_size=args.patch_size,
        in_channels=in_channels,
        hidden_size=512,
        depth=12,
        num_heads=8,
        num_classes=35 # SpeechCommands v2 has 35 words
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
            
    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            
            # If spectrogram, flatten
            if args.dataset_mode == 'spectrogram':
                x = x.view(x.shape[0], 1, -1) # [B, 1, 4096]
            
            # Sample t uniform [0, 1]
            t = torch.rand(x.shape[0], device=device)
            
            # Forward process
            noise = torch.randn_like(x)
            z_t, _ = flow.q_sample(x, t, noise)
            
            # Model prediction
            model_output = model(z_t, t * 1000, y) # Scale t to [0, 1000] for embedding
            
            # Loss
            if args.loss_type == 'epsilon':
                target = noise
                loss = F.mse_loss(model_output, target)
            elif args.loss_type == 'x':
                target = x
                loss = F.mse_loss(model_output, target)
            elif args.loss_type == 'v':
                # Plan: v_target = clean_audio - noise
                # v_pred = (model_output - z_t) / (1 - t)
                # Wait, if model_output is predicting x (as implied by "Convert model prediction to velocity"),
                # then we first predict x, then convert to v?
                # Or does the model predict v directly?
                # Plan says: "If predicting x: derive v = ...".
                # But then "If loss_type='mse_v': ... Convert model prediction to velocity: v_pred = (model_output - z_t) / (1 - t)"
                # This implies model_output IS x_pred.
                # So we are training x-prediction model but minimizing v-loss.
                
                # Let's assume model outputs x always for this mode, and we compute loss on v.
                # Or does model output v?
                # "The paper's final algorithm ... uses x-prediction but optimizes the v-loss".
                # So Model -> x_pred.
                # v_pred_from_x = (x_pred - z_t) / (1 - t) ???
                # Let's check the math.
                # z_t = t * x + (1-t) * eps
                # v = dx/dt = x - eps (velocity of the flow)
                # If we have x_pred, can we get v_pred?
                # z_t = t * x_pred + (1-t) * eps_pred
                # This is circular if we don't have eps.
                
                # Actually, v = x - eps.
                # And z_t = t * x + (1-t) * eps
                # z_t = t * x + (1-t) * (x - v) = x - (1-t)v
                # => (1-t)v = x - z_t
                # => v = (x - z_t) / (1-t)
                # YES.
                
                # So:
                # 1. Model predicts x_pred.
                # 2. v_pred = (x_pred - z_t) / (1 - t + 1e-5)
                # 3. v_target = x - noise
                # 4. Loss = MSE(v_pred, v_target)
                
                x_pred = model_output
                v_pred = (x_pred - z_t) / (1 - t.view(-1, 1, 1) + 1e-5)
                v_target = x - noise
                loss = F.mse_loss(v_pred, v_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': loss.item()})
            
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/model_{args.experiment_name}_ep{epoch+1}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--dataset_mode', type=str, default='raw', choices=['raw', 'spectrogram'])
    parser.add_argument('--loss_type', type=str, default='epsilon', choices=['epsilon', 'x', 'v'])
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mock', action='store_true', help='Use mock dataset')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    train(args)
