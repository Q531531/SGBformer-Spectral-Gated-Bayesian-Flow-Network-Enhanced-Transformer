

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BayesianFlowNetwork(nn.Module):
    """
    Bayesian Flow Network for residual refinement.
    
    This implements the BFN framework described in Eq 6-10 of the paper.
    The key insight is to model the residual generation as a parameter space
    evolution rather than pixel space denoising, enabling faster convergence.
    """
    
    def __init__(self, feature_dim, num_steps=10, time_embedding_dim=64):
        """
        Args:
            feature_dim (int): Dimension of input features from backbone
            num_steps (int): Number of BFN refinement steps (default: 10)
            time_embedding_dim (int): Dimension of time embeddings
        """
        super().__init__()
        
        self.num_steps = num_steps
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding network for step conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # Receiver network - lightweight denoising architecture
        self.receiver = ReceiverNetwork(
            in_channels=3,  # RGB residual
            feature_dim=feature_dim,
            time_dim=time_embedding_dim
        )
        
        # Learnable precision schedule (can be optimized during training)
        # Corresponds to α(t) in Eq 8 of the paper
        # Modern 2026 approach: use more stable initialization to prevent NaN
        self.precision_schedule = nn.Parameter(
            torch.linspace(0.1, 0.9, num_steps)  # More conservative range to prevent instability
        )
        
    def sender_process(self, residual_gt, step_idx):
        """
        Sender process implementing discretized Bayesian update (Eq 8-9).
        
        This simulates the information accumulation process where precision
        increases over time and the mean estimate approaches ground truth.
        
        Args:
            residual_gt (Tensor): Ground truth residual [B, 3, H, W]
            step_idx (int): Current timestep index
            
        Returns:
            tuple: (noisy_mean, precision) at current timestep
        """
        # Get precision at current step (α_t in the paper)
        alpha_t = self.precision_schedule[step_idx]
        
        # Bayesian update rule from Eq 8
        # μ_t = α_t * r_gt + √(1-α_t) * ε, where ε ~ N(0,I)
        # Modern 2026 approach: add numerical stability and gradient clipping
        noise = torch.randn_like(residual_gt)
        alpha_t_clamped = torch.clamp(alpha_t, min=0.1, max=0.9)  # Prevent extreme values
        
        # Stable computation to prevent NaN
        sqrt_term = torch.sqrt(torch.clamp(1 - alpha_t_clamped, min=1e-8))
        noisy_mean = alpha_t_clamped * residual_gt + sqrt_term * noise * 0.1  # Scale noise down
        
        # Precision increases over time (Eq 9)
        precision = alpha_t
        
        return noisy_mean, precision
        
    def receiver_process(self, noisy_mean, timestep, condition):
        """
        Receiver process for estimating clean residual.
        
        This implements the denoising function Ψ described in Eq 10,
        which takes noisy observations and predicts the clean residual.
        
        Args:
            noisy_mean (Tensor): Current noisy estimate [B, 3, H, W] 
            timestep (Tensor): Current timestep [B]
            condition (Tensor): Conditioning features from backbone [B, C, H, W]
            
        Returns:
            Tensor: Predicted clean residual [B, 3, H, W]
        """
        # Embed timestep
        t_emb = self.time_mlp(timestep.float().unsqueeze(-1))
        
        # Receiver network prediction
        predicted_residual = self.receiver(noisy_mean, t_emb, condition)
        
        return predicted_residual
        
    def forward(self, condition, residual_gt=None, num_inference_steps=None):
        """
        Forward pass supporting both training and inference modes.
        
        Training mode: Single-step denoising loss computation
        Inference mode: Iterative refinement over multiple steps
        
        Args:
            condition (Tensor): Backbone features [B, C, H, W]
            residual_gt (Tensor, optional): Ground truth residual for training
            num_inference_steps (int, optional): Override number of inference steps
            
        Returns:
            Tensor: Refined residual [B, 3, H, W]
        """
        B, C, H, W = condition.shape
        device = condition.device
        
        if self.training and residual_gt is not None:
            # Training mode: random timestep sampling
            step_idx = torch.randint(0, self.num_steps, (1,), device=device).item()
            timestep = torch.full((B,), step_idx / self.num_steps, device=device)
            
            # Apply sender process
            noisy_mean, precision = self.sender_process(residual_gt, step_idx)
            
            # Predict clean residual
            predicted_residual = self.receiver_process(noisy_mean, timestep, condition)
            
            return predicted_residual, precision
            
        else:
            # Inference mode: iterative refinement
            steps = num_inference_steps or self.num_steps
            
            # Initialize from prior: μ₀ = 0 (Eq 7)
            current_mean = torch.zeros(B, 3, H, W, device=device)
            
            for step in range(steps):
                timestep = torch.full((B,), step / steps, device=device)
                
                # Receiver prediction at current step
                predicted_residual = self.receiver_process(
                    current_mean, timestep, condition
                )
                
                # Bayesian update: move mean towards prediction
                # This implements the iterative refinement described after Eq 10
                alpha_t = self.precision_schedule[min(step, self.num_steps - 1)]
                current_mean = current_mean + alpha_t * (predicted_residual - current_mean) / (step + 1)
            
            return current_mean


class ReceiverNetwork(nn.Module):
    """
    Lightweight receiver network for BFN residual prediction.
    
    This implements the denoising function Ψ with conditional inputs:
    - Noisy residual estimate
    - Timestep embedding  
    - Backbone features (condition)
    """
    
    def __init__(self, in_channels, feature_dim, time_dim):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, 64, 3, 1, 1)
        
        # Conditioning projection
        self.cond_proj = nn.Conv2d(feature_dim, 64, 1)
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64)
        )
        
        # Denoising backbone - lightweight ResNet-style
        self.backbone = nn.ModuleList([
            ResidualBlock(64 + 64, 64, time_dim=64),  # noisy + condition
            ResidualBlock(64, 64, time_dim=64),
            ResidualBlock(64, 64, time_dim=64),
            ResidualBlock(64, 32, time_dim=64),
        ])
        
        # Output projection
        self.output_proj = nn.Conv2d(32, in_channels, 3, 1, 1)
        
    def forward(self, noisy_input, time_emb, condition):
        """
        Args:
            noisy_input (Tensor): Noisy residual estimate [B, 3, H, W]
            time_emb (Tensor): Time embedding [B, time_dim] 
            condition (Tensor): Backbone features [B, feature_dim, H_cond, W_cond]
        """
        B, C, H, W = noisy_input.shape
        
        # Project inputs
        x = self.input_proj(noisy_input)
        cond = self.cond_proj(condition)
        
        # Upsample condition to match residual spatial dimensions
        # Modern 2026 approach: use interpolation to align spatial dimensions
        if cond.shape[-2:] != (H, W):
            cond = F.interpolate(cond, size=(H, W), mode='bilinear', align_corners=False)
        
        # Fuse input and condition
        x = torch.cat([x, cond], dim=1)
        
        # Process through backbone with time conditioning
        for block in self.backbone:
            x = block(x, time_emb)
            
        # Output clean residual
        output = self.output_proj(x)
        
        return output


class ResidualBlock(nn.Module):
    """Time-conditioned residual block with GroupNorm."""
    
    def __init__(self, in_channels, out_channels, time_dim, num_groups=8):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        
        # Time conditioning
        self.time_proj = nn.Linear(time_dim, out_channels * 2)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_emb):
        identity = self.skip(x)
        
        # First convolution
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Time conditioning (scale and shift)
        time_params = self.time_proj(time_emb)  # [B, 2*C]
        scale, shift = time_params.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        
        h = h * (1 + scale) + shift
        h = F.relu(h)
        
        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        
        return F.relu(h + identity)


def test_bfn():
    """Test function to verify BFN functionality."""
    print("Testing Bayesian Flow Network...")
    
    # Test parameters
    B, C, H, W = 2, 64, 32, 32
    
    # Create test inputs
    condition = torch.randn(B, C, H, W)
    residual_gt = torch.randn(B, 3, H, W)
    
    # Initialize BFN
    bfn = BayesianFlowNetwork(feature_dim=C, num_steps=10)
    
    # Test training mode
    bfn.train()
    predicted_residual, precision = bfn(condition, residual_gt)
    
    print(f"Training mode:")
    print(f"  Predicted residual shape: {predicted_residual.shape}")
    print(f"  Precision: {precision:.4f}")
    
    # Test inference mode
    bfn.eval()
    with torch.no_grad():
        refined_residual = bfn(condition)
    
    print(f"Inference mode:")
    print(f"  Refined residual shape: {refined_residual.shape}")
    print(f"Parameters: {sum(p.numel() for p in bfn.parameters()):,}")
    
    print("✓ BFN test passed!")


if __name__ == "__main__":
    test_bfn()
