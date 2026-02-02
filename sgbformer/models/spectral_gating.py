"""
Spectral-Gated Backbone Module

This module implements the frequency-domain gating mechanism described in the paper.
The core idea is to suppress periodic weather artifacts (rain streaks, snow) by learning
adaptive filters in the frequency domain while preserving structural information.

Key Components:
- FFT-based frequency domain transformation
- Learnable soft gating masks 
- Amplitude modulation with phase preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralGatingBlock(nn.Module):
    """
    Spectral Gating Block with FFT-based frequency filtering.
    
    Architecture:
    1. Apply 2D FFT to input features
    2. Decouple amplitude and phase components
    3. Learn soft gating mask via 1x1 convolutions on real/imaginary parts
    4. Apply gate to modulate frequency components
    5. Reconstruct signal via inverse FFT
    
    This implements the core mechanism described in Eq. 3-4 of the paper.
    """
    
    def __init__(self, dim):
        """
        Args:
            dim (int): Channel dimension of input features
        """
        super().__init__()
        
        # Learnable gating network operating on frequency domain
        # Input: concatenated real and imaginary parts (2*dim channels)
        self.gate_network = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=1),
            nn.Sigmoid()  # Soft gating in [0, 1]
        )
        
        # Optional: learnable frequency-dependent weights
        self.freq_weights = nn.Parameter(torch.ones(1, dim, 1, 1))
        
    def forward(self, x):
        """
        Forward pass with frequency-domain gating.
        
        Args:
            x (Tensor): Input features [B, C, H, W]
            
        Returns:
            Tensor: Filtered features [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Step 1: Transform to frequency domain using 2D Real FFT
        # This corresponds to F(x) in the paper
        x_freq = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, W//2+1]
        
        # Step 2: Extract amplitude and phase components
        amplitude = torch.abs(x_freq)  # |F(x)|
        phase = torch.angle(x_freq)   # arg(F(x))
        
        # Step 3: Prepare input for gating network
        # Concatenate real and imaginary parts for learnable gating
        real_part = x_freq.real
        imag_part = x_freq.imag
        freq_concat = torch.cat([real_part, imag_part], dim=1)  # [B, 2C, H, W//2+1]
        
        # Step 4: Learn adaptive gating mask
        # This implements the learnable filter G(·) from Eq. 4
        gate_mask = self.gate_network(freq_concat)  # [B, C, H, W//2+1]
        
        # Optional: Apply frequency-dependent scaling
        gate_mask = gate_mask * self.freq_weights
        
        # Step 5: Apply gating to amplitude while preserving phase
        # Filtered amplitude: G(F(x)) * |F(x)|
        filtered_amplitude = amplitude * gate_mask
        
        # Step 6: Reconstruct complex spectrum with filtered amplitude
        filtered_freq = filtered_amplitude * torch.exp(1j * phase)
        
        # Step 7: Inverse FFT to return to spatial domain
        # This gives us the filtered output F^(-1)(G(F(x)))
        filtered_x = torch.fft.irfft2(filtered_freq, s=(H, W), norm='ortho')
        
        return filtered_x


class FrequencyAwareBlock(nn.Module):
    """
    Enhanced block combining spatial and frequency processing.
    
    This implements the dual-path design where features flow through both
    spatial attention and frequency-domain filtering in parallel.
    """
    
    def __init__(self, dim, num_heads=4):
        super().__init__()
        
        # Spatial processing path
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # Frequency processing path  
        self.freq_norm = nn.LayerNorm(dim)
        self.spectral_gate = SpectralGatingBlock(dim)
        
        # Feature fusion
        self.fusion_conv = nn.Conv2d(dim * 2, dim, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input features [B, C, H, W]
        """
        B, C, H, W = x.shape
        identity = x
        
        # Spatial attention path
        x_spatial = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_spatial = self.spatial_norm(x_spatial)
        x_spatial, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = x_spatial.transpose(1, 2).reshape(B, C, H, W)
        
        # Frequency filtering path
        x_freq = self.freq_norm(x.flatten(2).transpose(1, 2))
        x_freq = x_freq.transpose(1, 2).reshape(B, C, H, W)
        x_freq = self.spectral_gate(x_freq)
        
        # Fuse spatial and frequency features
        x_fused = torch.cat([x_spatial, x_freq], dim=1)
        x_fused = self.fusion_conv(x_fused)
        
        # Feed-forward network
        x_out = self.ffn(x_fused) + identity
        
        return x_out


def test_spectral_gating():
    """Test function to verify spectral gating functionality."""
    print("Testing Spectral Gating Block...")
    
    # Test parameters
    batch_size = 2
    channels = 64
    height, width = 32, 32
    
    # Create test input
    x = torch.randn(batch_size, channels, height, width)
    
    # Test SpectralGatingBlock
    gating_block = SpectralGatingBlock(channels)
    output = gating_block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in gating_block.parameters()):,}")
    
    assert output.shape == x.shape, "Shape mismatch!"
    print("✓ SpectralGatingBlock test passed!")
    
    # Test FrequencyAwareBlock
    freq_block = FrequencyAwareBlock(channels)
    output2 = freq_block(x)
    
    assert output2.shape == x.shape, "Shape mismatch!"
    print("✓ FrequencyAwareBlock test passed!")


if __name__ == "__main__":
    test_spectral_gating()
