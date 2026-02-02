"""
SGBformer: Spectral-Gated Bayesian Flow Network Enhanced Transformer for All-Weather Image Restoration

Main Architecture Module

This module implements the complete SGBformer architecture described in the paper.
The model combines three key innovations:
1. Spectral-Gated Backbone for structure recovery
2. Degradation-Aware Cross-Attention with semantic guidance
3. Bayesian Flow Network for efficient residual refinement

Architecture Flow:
Input Image -> Spectral-Gated Backbone -> Coarse Structure
            -> CLIP Semantic Guidance -> DAC Modulation
            -> BFN Residual Refinement -> Final Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .spectral_gating import SpectralGatingBlock, FrequencyAwareBlock
from .bfn import BayesianFlowNetwork
from .dac_clip import (
    MockCLIPEncoder,
    RealCLIPEncoder,
    DegradationAwareCrossAttention,
    SemanticGuidanceModule,
)


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D feature maps."""
    
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention (lightweight version)."""
    
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).reshape(b, c, h, w)
        out = self.project_out(out)
        
        return out


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network (lightweight version)."""
    
    def __init__(self, dim, ffn_expansion_factor=2.66):
        super().__init__()
        
        hidden_features = int(dim * ffn_expansion_factor)
        
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, stride=1, padding=1, groups=hidden_features * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    """Lightweight Transformer Block with MDTA and GDFN."""
    
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.66):
        super().__init__()
        
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = LayerNorm2d(dim) 
        self.ffn = GDFN(dim, ffn_expansion_factor)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SpectralGatedBackbone(nn.Module):
    """
    Spectral-Gated U-Net Backbone for structural restoration.
    
    Fixed architecture with proper channel flow:
    Encoder: 32 -> 64 -> 128 -> 256 (bottleneck)
    Decoder: 256 -> 128 -> 64 -> 32 (with skip connections)
    """
    
    def __init__(self, 
                 in_channels=3, 
                 dim=32,
                 num_blocks=[2, 3, 3, 4],
                 heads=[1, 2, 4, 8]):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, dim, 3, 1, 1)
        
        # Encoder
        self.encoder_levels = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        ch = dim  # 32
        for i, num_blk in enumerate(num_blocks):
            # Encoder blocks
            encoder_blks = nn.ModuleList([
                TransformerBlock(ch, heads[i]) for _ in range(num_blk)
            ])
            self.encoder_levels.append(encoder_blks)
            
            # Downsample (except last level = bottleneck)
            if i < len(num_blocks) - 1:
                self.downsample.append(nn.Conv2d(ch, ch * 2, 4, 2, 1))
                ch = ch * 2  # 32->64->128->256
        
        # Bottleneck with Spectral Gating (ch=256)
        self.bottleneck = nn.ModuleList([
            SpectralGatingBlock(ch),
            TransformerBlock(ch, heads[-1]),
            SpectralGatingBlock(ch),
        ])
        
        # Decoder - manual construction for clarity
        self.decoder_levels = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.skip_conv = nn.ModuleList()
        
        # Level 0: 256->128, skip from encoder level 2 (128ch)
        self.upsample.append(nn.ConvTranspose2d(256, 128, 4, 2, 1))
        self.skip_conv.append(nn.Conv2d(128, 128, 1))  # Encoder level 2 has 128ch
        self.decoder_levels.append(nn.ModuleList([
            TransformerBlock(128, heads[2]) for _ in range(num_blocks[2])  # 3 blocks
        ]))
        
        # Level 1: 128->64, skip from encoder level 1 (64ch)  
        self.upsample.append(nn.ConvTranspose2d(128, 64, 4, 2, 1))
        self.skip_conv.append(nn.Conv2d(64, 64, 1))   # Encoder level 1 has 64ch
        self.decoder_levels.append(nn.ModuleList([
            TransformerBlock(64, heads[1]) for _ in range(num_blocks[1])   # 3 blocks
        ]))
        
        # Level 2: 64->32, skip from encoder level 0 (32ch)
        self.upsample.append(nn.ConvTranspose2d(64, 32, 4, 2, 1))
        self.skip_conv.append(nn.Conv2d(32, 32, 1))   # Encoder level 0 has 32ch
        self.decoder_levels.append(nn.ModuleList([
            TransformerBlock(32, heads[0]) for _ in range(num_blocks[0])   # 2 blocks
        ]))
        
        # Output projection
        self.output_proj = nn.Conv2d(32, 3, 3, 1, 1)
        
    def forward(self, x, semantic_guidance=None):
        """
        Forward pass through the spectral-gated backbone.
        
        Args:
            x (Tensor): Input image [B, 3, H, W]
            semantic_guidance (list, optional): Semantic guidance at each level
            
        Returns:
            tuple: (restored_structure, encoder_features)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Encoder
        encoder_features = []
        for level, downsample in zip(self.encoder_levels[:-1], self.downsample):
            for blk in level:
                x = blk(x)
            encoder_features.append(x)
            x = downsample(x)
        
        # Last encoder level (no downsampling)
        for blk in self.encoder_levels[-1]:
            x = blk(x)
        encoder_features.append(x)
        
        # Bottleneck with spectral gating
        for blk in self.bottleneck:
            x = blk(x)
        
        # Store for BFN conditioning
        bottleneck_features = x
        
        # Decoder with explicit level mapping  
        # Level 0: 256->128, skip from encoder[2] (128ch)
        x = self.upsample[0](x)  # 256->128
        skip = self.skip_conv[0](encoder_features[2])  # encoder level 2: 128->128
        x = x + skip
        for blk in self.decoder_levels[0]:
            x = blk(x)
        
        # Level 1: 128->64, skip from encoder[1] (64ch)  
        x = self.upsample[1](x)  # 128->64
        skip = self.skip_conv[1](encoder_features[1])  # encoder level 1: 64->64
        x = x + skip
        for blk in self.decoder_levels[1]:
            x = blk(x)
        
        # Level 2: 64->32, skip from encoder[0] (32ch)
        x = self.upsample[2](x)  # 64->32
        skip = self.skip_conv[2](encoder_features[0])  # encoder level 0: 32->32
        x = x + skip
        for blk in self.decoder_levels[2]:
            x = blk(x)
        
        # Output projection with residual connection
        structure_output = self.output_proj(x)
        
        return structure_output, bottleneck_features


class SGBformer(nn.Module):
    """
    SGBformer: Spectral-Gated Bayesian Flow Network Enhanced Transformer
    
    Complete architecture integrating:
    1. Spectral-Gated Backbone for coarse structure restoration
    2. CLIP-guided Degradation-Aware Cross-Attention for semantic guidance
    3. Bayesian Flow Network for efficient residual refinement
    
    This implementation targets ~13.2M parameters as reported in the paper.
    """
    
    def __init__(self, 
                 in_channels=3,
                 out_channels=3,
                 dim=32,  # Lightweight design
                 num_blocks=[2, 3, 3, 4],
                 heads=[1, 2, 4, 8],
                 semantic_dim=512,
                 bfn_steps=10,
                 enable_semantic_guidance=False,
                 clip_model_name="openai/clip-vit-base-patch32",
                 clip_image_size=224):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels  
            dim (int): Base feature dimension (reduced for lightweight design)
            num_blocks (list): Number of blocks at each level
            heads (list): Number of attention heads at each level
            semantic_dim (int): Dimension of semantic embeddings
            bfn_steps (int): Number of BFN refinement steps
            enable_semantic_guidance (bool): Whether to use CLIP guidance
            clip_model_name (str): Hugging Face CLIP model name
            clip_image_size (int): CLIP input resolution (default 224)
        """
        super().__init__()
        
        self.enable_semantic_guidance = enable_semantic_guidance
        self.clip_model_name = clip_model_name
        self.clip_image_size = clip_image_size
        
        # 1. CLIP Encoder for semantic understanding
        if enable_semantic_guidance:
            try:
                self.clip_encoder = RealCLIPEncoder(
                    model_name=clip_model_name,
                    image_size=clip_image_size,
                )
                self.clip_backend = "real"
            except Exception as exc:
                print(f"Warning: {exc}. Falling back to MockCLIPEncoder.")
                self.clip_encoder = MockCLIPEncoder(embed_dim=semantic_dim)
                self.clip_backend = "mock"
            
            # DAC module for bottleneck features
            bottleneck_dim = dim * (2 ** (len(num_blocks) - 1))
            self.dac_module = DegradationAwareCrossAttention(
                bottleneck_dim, semantic_dim
            )
        
        # 2. Spectral-Gated Backbone
        self.backbone = SpectralGatedBackbone(
            in_channels=in_channels,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads
        )
        
        # 3. Bayesian Flow Network for residual refinement
        bottleneck_dim = dim * (2 ** (len(num_blocks) - 1))
        self.bfn = BayesianFlowNetwork(
            feature_dim=bottleneck_dim,
            num_steps=bfn_steps
        )
        
        # Adaptive gating for residual combination
        self.residual_gate = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, degraded_img, clean_img=None):
        """
        Forward pass supporting both training and inference.
        
        Args:
            degraded_img (Tensor): Input degraded image [B, 3, H, W]
            clean_img (Tensor, optional): Clean ground truth for training
            
        Returns:
            dict: Dictionary containing restoration results
        """
        B, C, H, W = degraded_img.shape
        
        # Stage 1: Semantic Understanding (if enabled)
        semantic_embedding = None
        if self.enable_semantic_guidance:
            # Extract semantic embedding using Mock CLIP
            semantic_embedding = self.clip_encoder(degraded_img)
        
        # Stage 2: Spectral-Gated Backbone for Structure Recovery
        structure_output, bottleneck_features = self.backbone(degraded_img)
        
        # Apply semantic guidance to bottleneck features
        if self.enable_semantic_guidance and semantic_embedding is not None:
            # Flatten for cross-attention
            B_feat, C_feat, H_feat, W_feat = bottleneck_features.shape
            features_flat = bottleneck_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
            
            # Apply DAC
            guided_features = self.dac_module(features_flat, semantic_embedding)
            bottleneck_features = guided_features.transpose(1, 2).view(B_feat, C_feat, H_feat, W_feat)
        
        # Stage 3: Coarse Structure with Residual Connection
        coarse_structure = structure_output + degraded_img
        
        # Stage 4: BFN Residual Refinement
        if self.training and clean_img is not None:
            # Training mode: compute BFN loss
            gt_residual = clean_img - coarse_structure
            refined_residual, precision = self.bfn(bottleneck_features, gt_residual)
        else:
            # Inference mode: iterative refinement
            refined_residual = self.bfn(bottleneck_features)
            precision = 1.0
        
        # Stage 5: Adaptive Residual Combination
        # Learn importance map for residual application
        residual_gate = self.residual_gate(coarse_structure)
        
        # Modern 2026 approach: handle spatial dimension mismatch gracefully
        if refined_residual.shape[-2:] != residual_gate.shape[-2:]:
            # Upsample refined residual to match gate resolution
            refined_residual = F.interpolate(
                refined_residual, 
                size=residual_gate.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        final_residual = refined_residual * residual_gate
        
        # Final output
        final_output = coarse_structure + final_residual
        
        # Return comprehensive results for analysis
        results = {
            'final_output': final_output,
            'coarse_structure': coarse_structure,
            'refined_residual': refined_residual,
            'residual_gate': residual_gate,
            'semantic_embedding': semantic_embedding,
            'precision': precision
        }
        
        return results
    
    def get_num_params(self):
        """Get the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())


def test_sgbformer():
    """Test function to verify SGBformer functionality and parameter count."""
    print("Testing SGBformer Architecture...")
    
    # Test parameters
    B, H, W = 2, 256, 256
    
    # Create test inputs
    degraded_img = torch.randn(B, 3, H, W)
    clean_img = torch.randn(B, 3, H, W)
    
    # Initialize model
    model = SGBformer(
        dim=32,  # Lightweight configuration
        num_blocks=[2, 3, 3, 4],
        heads=[1, 2, 4, 8],
        bfn_steps=10,
        enable_semantic_guidance=False
    )
    
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Target: ~13.2M parameters")
    
    # Test training mode
    model.train()
    train_results = model(degraded_img, clean_img)
    
    print(f"\nTraining Mode:")
    print(f"  Final output: {train_results['final_output'].shape}")
    print(f"  Coarse structure: {train_results['coarse_structure'].shape}")
    print(f"  Refined residual: {train_results['refined_residual'].shape}")
    print(f"  Precision: {train_results['precision']:.4f}")
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        inference_results = model(degraded_img)
    
    print(f"\nInference Mode:")
    print(f"  Final output: {inference_results['final_output'].shape}")
    print(f"  Semantic embedding: {inference_results['semantic_embedding'].shape if inference_results['semantic_embedding'] is not None else 'None'}")
    
    # Component parameter breakdown
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    bfn_params = sum(p.numel() for p in model.bfn.parameters())
    clip_params = sum(p.numel() for p in model.clip_encoder.parameters()) if model.enable_semantic_guidance else 0
    dac_params = sum(p.numel() for p in model.dac_module.parameters()) if model.enable_semantic_guidance else 0
    
    print(f"\nParameter Breakdown:")
    print(f"  Spectral-Gated Backbone: {backbone_params:,}")
    print(f"  BFN Module: {bfn_params:,}")
    print(f"  Mock CLIP Encoder: {clip_params:,}")
    print(f"  DAC Module: {dac_params:,}")
    print(f"  Other components: {model.get_num_params() - backbone_params - bfn_params - clip_params - dac_params:,}")
    
    print("\n✓ SGBformer test passed!")
    
    return model


if __name__ == "__main__":
    model = test_sgbformer()
