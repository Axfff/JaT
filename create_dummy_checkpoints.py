import torch
import os
import sys
sys.path.append('src')
from src.model import JiT

os.makedirs('checkpoints', exist_ok=True)

configs = [
    ('control_spec_eps', 'spectrogram', 'epsilon'),
    ('test_spec_x', 'spectrogram', 'x'),
    ('baseline_raw_eps', 'raw', 'epsilon'),
    ('main_raw_x', 'raw', 'x')
]

for name, mode, loss in configs:
    print(f"Creating dummy checkpoint for {name}...")
    if mode == 'raw':
        input_size = 16384
        in_channels = 1
    else:
        input_size = 64
        in_channels = 1
        
    if mode == 'raw':
        patch_size = 512
    else:
        patch_size = 8

    model = JiT(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=512,
        depth=12,
        num_heads=8,
        num_classes=35,
        bottleneck_dim=512,
        in_context_len=0,
        is_1d=(mode == 'raw')
    )
    
    path = f'checkpoints/model_{name}_ep50.pth'
    torch.save(model.state_dict(), path)
    print(f"Saved {path}")
