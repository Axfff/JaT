import torch
import os
from src.model import JustAudioTransformer

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
        input_size = 4096
        in_channels = 1
        
    model = JustAudioTransformer(
        input_size=input_size,
        patch_size=512,
        in_channels=in_channels,
        hidden_size=512,
        depth=12,
        num_heads=8,
        num_classes=35
    )
    
    path = f'checkpoints/model_{name}_ep50.pth'
    torch.save(model.state_dict(), path)
    print(f"Saved {path}")
