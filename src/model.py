# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid, RMSNorm


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True, is_1d=False):
        super().__init__()
        self.is_1d = is_1d
        if is_1d:
            self.img_size = img_size # int
            self.patch_size = patch_size # int
            self.num_patches = img_size // patch_size
            self.proj1 = nn.Conv1d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
            self.proj2 = nn.Conv1d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)
        else:
            img_size = (img_size, img_size)
            patch_size = (patch_size, patch_size)
            num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
            self.img_size = img_size
            self.patch_size = patch_size
            self.num_patches = num_patches

            self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
            self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        if self.is_1d:
            B, C, L = x.shape
            assert L == self.img_size, f"Input size {L} doesn't match model {self.img_size}."
            x = self.proj2(self.proj1(x)).transpose(1, 2)
        else:
            B, C, H, W = x.shape
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class OverlappingPatchEmbed1D(nn.Module):
    """
    1D Patch Embedding with configurable overlap (hop_size).
    
    For raw audio of shape [B, C, L]:
    - Extracts overlapping windows of size `patch_size` with stride `hop_size`
    - When hop_size < patch_size, windows overlap
    - Number of patches = (L - patch_size) // hop_size + 1
    
    Args:
        input_size: Total input length (e.g., 16384 for 1s at 16kHz)
        patch_size: Size of each patch/window
        hop_size: Stride between patches (hop_size < patch_size = overlap)
        in_chans: Input channels (1 for mono audio)
        pca_dim: Bottleneck dimension
        embed_dim: Output embedding dimension
    """
    def __init__(self, input_size=16384, patch_size=512, hop_size=256, 
                 in_chans=1, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.hop_size = hop_size
        self.in_chans = in_chans
        
        # Calculate number of patches
        self.num_patches = (input_size - patch_size) // hop_size + 1
        
        # Two-stage projection (bottleneck architecture like BottleneckPatchEmbed)
        # Input per patch: [in_chans * patch_size]
        patch_dim = in_chans * patch_size
        self.proj1 = nn.Linear(patch_dim, pca_dim, bias=False)
        self.proj2 = nn.Linear(pca_dim, embed_dim, bias=bias)
        
        # Register Hann window as buffer for overlap-add reconstruction
        # This will be used by unpatchify
        hann = torch.hann_window(patch_size)
        self.register_buffer('hann_window', hann)
        
    def forward(self, x):
        """
        x: [B, C, L] - raw audio
        Returns: [B, num_patches, embed_dim]
        """
        B, C, L = x.shape
        assert L == self.input_size, f"Input size {L} doesn't match model {self.input_size}."
        
        # Extract overlapping patches using unfold
        # unfold(dimension, size, step) -> [B, C, num_patches, patch_size]
        patches = x.unfold(2, self.patch_size, self.hop_size)  # [B, C, num_patches, patch_size]
        
        # Reshape to [B, num_patches, C * patch_size]
        patches = patches.permute(0, 2, 1, 3)  # [B, num_patches, C, patch_size]
        patches = patches.reshape(B, self.num_patches, C * self.patch_size)
        
        # Apply projections
        x = self.proj2(self.proj1(patches))  # [B, num_patches, embed_dim]
        
        return x



class SpectrumPatchEmbed(nn.Module):
    """
    Spectrogram to Patch Embedding - patches ONLY along time axis.
    
    For a spectrogram of shape [B, C, freq_bins, time_frames]:
    - Divides along time axis with stride=patch_size
    - Each patch has shape [freq_bins * patch_size * C] 
    - Number of patches = time_frames // patch_size
    
    This preserves the full frequency resolution in each patch, which is 
    beneficial for audio as frequency bins are semantically coupled.
    """
    def __init__(self, freq_bins=64, time_frames=64, patch_size=8, in_chans=1, 
                 pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        self.freq_bins = freq_bins
        self.time_frames = time_frames
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_patches = time_frames // patch_size
        
        # Each patch covers [freq_bins, patch_size] -> flatten to [freq_bins * patch_size * in_chans]
        patch_dim = freq_bins * patch_size * in_chans
        
        # Two-stage projection with bottleneck (like BottleneckPatchEmbed)
        # Use Conv1d treating time axis as the sequence dimension
        # Input: [B, C * freq_bins, time_frames] -> kernel covers patch_size time steps
        self.proj1 = nn.Conv1d(in_chans * freq_bins, pca_dim, kernel_size=patch_size, 
                               stride=patch_size, bias=False)
        self.proj2 = nn.Conv1d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)
        
    def forward(self, x):
        """
        x: [B, C, freq_bins, time_frames] - standard spectrogram format
        Returns: [B, num_patches, embed_dim]
        """
        B, C, F, T = x.shape
        assert F == self.freq_bins and T == self.time_frames, \
            f"Input shape ({F}, {T}) doesn't match model ({self.freq_bins}, {self.time_frames})."
        
        # Reshape to [B, C * freq_bins, time_frames] for Conv1d along time axis
        x = x.reshape(B, C * F, T)
        
        # Apply projections: [B, C*F, T] -> [B, pca_dim, num_patches] -> [B, embed_dim, num_patches]
        x = self.proj2(self.proj1(x))
        
        # Transpose to [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. 
    Supports label dropout for classifier-free guidance (CFG).
    
    The embedding table has num_classes + 1 entries, where the last entry
    (index = num_classes) is the null/unconditional token used for CFG.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.null_token_idx = num_classes  # Index for unconditional embedding

    def forward(self, labels, drop_labels=None):
        """
        Args:
            labels: Class labels [B]
            drop_labels: Optional boolean mask [B] indicating which labels to drop.
                        If None, no labels are dropped.
        Returns:
            embeddings: Label embeddings [B, hidden_size]
        """
        if drop_labels is not None:
            # Replace dropped labels with null token index
            labels = labels.clone()
            labels[drop_labels] = self.null_token_idx
        embeddings = self.embedding_table(labels)
        return embeddings
    
    def get_null_embedding(self, batch_size, device):
        """Get null embeddings for CFG unconditional generation."""
        null_labels = torch.full((batch_size,), self.null_token_idx, dtype=torch.long, device=device)
        return self.embedding_table(null_labels)


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    """
    Scaled dot-product attention with FP16-safe computation.
    Attention weights are computed in float32 to prevent overflow in softmax.
    """
    original_dtype = query.dtype
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))

    # Compute attention in float32 for numerical stability with FP16
    query_f32 = query.float()
    key_f32 = key.float()
    value_f32 = value.float()
    
    attn_weight = query_f32 @ key_f32.transpose(-2, -1) * scale_factor
    
    # Clamp attention logits to prevent overflow in softmax
    attn_weight = torch.clamp(attn_weight, min=-65504.0, max=65504.0)
    
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    
    output = attn_weight @ value_f32
    
    # Return in original dtype
    return output.to(original_dtype)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class Snake(nn.Module):
    """
    Snake activation function: f(x) = x + sin²(x)
    
    Provides inductive bias for periodicity, helping networks generate 
    sharp harmonics instead of muffled noise. Particularly useful for 
    audio waveform generation.
    
    Reference: Neural Networks Fail to Learn Periodic Functions (Ziyin et al.)
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        # alpha controls the frequency of the periodic component
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
    def forward(self, x):
        return x + torch.sin(self.alpha * x) ** 2


class SnakeGLUFFN(nn.Module):
    """
    GLU-style FFN with Snake activation instead of SiLU.
    Better suited for audio waveform generation due to periodicity bias.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)
        self.snake = Snake(alpha=1.0)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        # Use Snake activation instead of SiLU for periodicity bias
        hidden = self.snake(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, is_1d=False):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        out_dim = patch_size * out_channels if is_1d else patch_size * patch_size * out_channels
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SpectrumFinalLayer(nn.Module):
    """
    Final layer for spectrum mode - outputs [freq_bins * patch_size * out_channels] per patch.
    """
    def __init__(self, hidden_size, freq_bins, patch_size, out_channels):
        super().__init__()
        self.freq_bins = freq_bins
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        self.norm_final = RMSNorm(hidden_size)
        # Output dimension: freq_bins * patch_size * channels
        out_dim = freq_bins * patch_size * out_channels
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, use_snake=False):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        # Use Snake activation for audio waveform generation
        if use_snake:
            self.mlp = SnakeGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        else:
            self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
            
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JiT(nn.Module):
    """
    Just image Transformer.
    
    Modes:
    - is_1d=False, is_spectrum=False: Standard 2D image patchification (default)  
    - is_1d=True: 1D signal patchification (for raw audio)
    - is_spectrum=True: Spectrum patchification (patches along time axis only, preserves full frequency)
    
    Args:
        use_snake: If True, use Snake activation in FFN blocks for better audio periodicity
        hop_size: For 1D mode, stride between patches. If hop_size < patch_size, patches overlap.
                 Use None or 0 for non-overlapping (defaults to patch_size).
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        is_1d=False,
        is_spectrum=False,
        freq_bins=64,  # For spectrum mode: number of frequency bins
        time_frames=64,  # For spectrum mode: number of time frames
        use_snake=False,  # Use Snake activation for audio waveform generation
        hop_size=None  # For 1D mode: stride between patches (None = patch_size = no overlap)
    ):
        super().__init__()
        self.is_1d = is_1d
        self.is_spectrum = is_spectrum
        self.freq_bins = freq_bins
        self.time_frames = time_frames
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        self.use_snake = use_snake
        
        # For 1D overlapping mode
        self.hop_size = hop_size if hop_size is not None and hop_size > 0 else patch_size
        self.use_overlap = is_1d and (self.hop_size < patch_size)

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed - choose based on mode
        if is_spectrum:
            self.x_embedder = SpectrumPatchEmbed(
                freq_bins=freq_bins, 
                time_frames=time_frames,
                patch_size=patch_size, 
                in_chans=in_channels, 
                pca_dim=bottleneck_dim, 
                embed_dim=hidden_size, 
                bias=True
            )
        elif is_1d and self.use_overlap:
            # 1D with overlapping patches
            self.x_embedder = OverlappingPatchEmbed1D(
                input_size=input_size,
                patch_size=patch_size,
                hop_size=self.hop_size,
                in_chans=in_channels,
                pca_dim=bottleneck_dim,
                embed_dim=hidden_size,
                bias=True
            )
        else:
            self.x_embedder = BottleneckPatchEmbed(
                input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True, is_1d=is_1d
            )

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope - for spectrum mode, use 1D rope since we only have sequence along time
        half_head_dim = hidden_size // num_heads // 2
        if is_1d or is_spectrum:
            hw_seq_len = num_patches
            rope_is_1d = True
        else:
            hw_seq_len = input_size // patch_size
            rope_is_1d = False
            
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0,
            is_1d=rope_is_1d
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len,
            is_1d=rope_is_1d
        )

        # transformer blocks with optional Snake activation
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     use_snake=use_snake)
            for i in range(depth)
        ])

        # linear predict - choose based on mode
        if is_spectrum:
            self.final_layer = SpectrumFinalLayer(hidden_size, freq_bins, patch_size, self.out_channels)
        else:
            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, is_1d=is_1d)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.is_1d or self.is_spectrum:
            # 1D positional embedding for raw audio or spectrum mode
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.x_embedder.num_patches, dtype=np.float32))
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """
        Reconstruct the original spatial structure from patch sequence.
        
        Args:
            x: Patch sequence with output dimensions
               - 2D mode: (N, T, patch_size**2 * C) where T = (H/p) * (W/p)
               - 1D mode: (N, T, patch_size * C) where T = L/p
               - Spectrum mode: (N, T, freq_bins * patch_size * C) where T = time_frames/p
            p: patch_size
            
        Returns:
            - 2D mode: (N, C, H, W)
            - 1D mode: (N, C, L)
            - Spectrum mode: (N, C, freq_bins, time_frames)
        """
        c = self.out_channels
        
        if self.is_spectrum:
            # x: [N, T, freq_bins * patch_size * C]
            # output: [N, C, freq_bins, time_frames]
            f = self.freq_bins
            num_time_patches = x.shape[1]  # T = time_frames / patch_size
            
            # Reshape: [N, T, f*p*c] -> [N, T, f, p, c]
            x = x.reshape(x.shape[0], num_time_patches, f, p, c)
            
            # Rearrange: [N, T, f, p, c] -> [N, c, f, T, p] -> [N, c, f, T*p]
            x = x.permute(0, 4, 2, 1, 3)  # [N, c, f, T, p]
            x = x.reshape(x.shape[0], c, f, num_time_patches * p)  # [N, c, f, time_frames]
            
            return x
        elif self.is_1d:
            if self.use_overlap:
                # Overlapping patches: use Hann window overlap-add reconstruction
                return self._unpatchify_overlap(x, p)
            else:
                # Non-overlapping: simple reshape
                # x: [N, T, p*c] -> [N, c, L]
                x = x.reshape(shape=(x.shape[0], x.shape[1], p, c))
                x = torch.einsum('ntpc->nctp', x)
                imgs = x.reshape(shape=(x.shape[0], c, x.shape[2] * p))
                return imgs
        else:
            h = w = int(x.shape[1] ** 0.5)
            assert h * w == x.shape[1]

            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            x = torch.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
            return imgs

    def _unpatchify_overlap(self, x, p):
        """
        Overlap-add reconstruction with Hann windowing for smooth crossfade.
        
        Args:
            x: [N, num_patches, p * c] patch sequence
            p: patch_size
            
        Returns:
            [N, c, L] reconstructed signal
        """
        N, num_patches, _ = x.shape
        c = self.out_channels
        hop = self.hop_size
        L = self.input_size  # Original input length
        
        # Reshape to [N, num_patches, p, c]
        x = x.reshape(N, num_patches, p, c)
        
        # Permute to [N, c, num_patches, p]
        x = x.permute(0, 3, 1, 2)  # [N, c, num_patches, p]
        
        # Get Hann window from embedder
        hann = self.x_embedder.hann_window  # [p]
        
        # Prepare output and normalization buffers
        output = torch.zeros(N, c, L, device=x.device, dtype=x.dtype)
        norm = torch.zeros(L, device=x.device, dtype=x.dtype)
        
        # Overlap-add with Hann windowing
        for i in range(num_patches):
            start = i * hop
            end = start + p
            
            # Apply Hann window to each patch
            windowed = x[:, :, i, :] * hann  # [N, c, p] * [p] -> [N, c, p]
            
            output[:, :, start:end] += windowed
            norm[start:end] += hann
        
        # Normalize by window overlap
        # Avoid division by zero
        norm = torch.clamp(norm, min=1e-8)
        output = output / norm.unsqueeze(0).unsqueeze(0)  # [N, c, L] / [1, 1, L]
        
        return output

    def forward(self, x, t, y, drop_labels=None):
        """
        Forward pass with optional label dropout for CFG training.
        
        Args:
            x: (N, C, H, W) or (N, C, L) depending on mode
            t: (N,) timesteps
            y: (N,) class labels
            drop_labels: Optional (N,) boolean mask for CFG training
        """
        # class and time embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, drop_labels=drop_labels)
        c = t_emb + y_emb

        # forward JiT
        x = self.x_embedder(x)
        x += self.pos_embed

        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)

        x = x[:, self.in_context_len:]

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        return output
    
    def forward_with_cfg(self, x, t, y, cfg_scale=1.0):
        """
        Forward pass with Classifier-Free Guidance for inference.
        
        CFG formula: output = uncond + cfg_scale * (cond - uncond)
        
        Args:
            x: Input tensor
            t: Timesteps
            y: Class labels
            cfg_scale: Guidance scale (1.0 = no guidance)
            
        Returns:
            CFG-guided output
        """
        if cfg_scale == 1.0:
            return self.forward(x, t, y)
        
        # Conditional prediction
        out_cond = self.forward(x, t, y)
        
        # Unconditional prediction (using null token)
        batch_size = x.shape[0]
        drop_all = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        out_uncond = self.forward(x, t, y, drop_labels=drop_all)
        
        # CFG combination
        return out_uncond + cfg_scale * (out_cond - out_uncond)


def JiT_B_16(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)

def JiT_B_32(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32, **kwargs)

def JiT_L_16(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)

def JiT_L_32(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)

def JiT_H_16(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16, **kwargs)

def JiT_H_32(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32, **kwargs)


JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
}
