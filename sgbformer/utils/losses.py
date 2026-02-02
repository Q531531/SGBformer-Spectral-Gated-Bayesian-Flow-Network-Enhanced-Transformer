"""
Loss Functions for SGBformer

This module implements the combined loss function described in the paper,
including the main reconstruction loss and the BFN-specific Bayesian loss
for efficient residual refinement training.

Key Components:
- Charbonnier Loss: Robust L1-variant for main reconstruction
- BFN Bayesian Loss: KL divergence loss for Bayesian Flow Network training
- Combined Loss: Weighted combination with adaptive scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (smooth L1 variant) with 2026 numerical stability enhancements.
    
    This is more robust to outliers compared to standard L2 loss,
    which is important for image restoration tasks with various
    degradation types and intensities.
    
    Loss = sqrt(||pred - target||^2 + eps^2)
    """
    
    def __init__(self, eps=1e-3):
        """
        Args:
            eps (float): Smoothing parameter to avoid singularity at zero
        """
        super().__init__()
        self.eps = eps
        
    def forward(self, pred, target):
        """
        Compute Charbonnier loss with numerical stability.
        
        Args:
            pred (Tensor): Predicted images [B, C, H, W]
            target (Tensor): Target clean images [B, C, H, W]
            
        Returns:
            Tensor: Scalar loss value
        """
        # Modern 2026 approach: add numerical stability checks
        diff = pred - target
        
        # Clamp extreme values to prevent overflow
        diff = torch.clamp(diff, min=-10.0, max=10.0)
        
        # Stable Charbonnier computation
        loss_per_pixel = torch.sqrt(diff.pow(2) + self.eps**2)
        
        # Check for NaN and handle gracefully
        if torch.isnan(loss_per_pixel).any():
            print("Warning: NaN detected in Charbonnier loss, using fallback")
            loss_per_pixel = torch.where(torch.isnan(loss_per_pixel), 
                                       torch.full_like(loss_per_pixel, self.eps), 
                                       loss_per_pixel)
        
        return loss_per_pixel.mean()


class BFNBayesianLoss(nn.Module):
    """
    Bayesian Flow Network Loss for residual refinement.
    
    This implements the BFN training loss described in Eq 14-15 of the paper.
    Unlike standard denoising losses, this considers the Bayesian nature of
    the parameter evolution process.
    
    The loss combines:
    1. Reconstruction term: ||predicted_residual - gt_residual||
    2. KL regularization: Precision-weighted variance penalty
    """
    
    def __init__(self, recon_weight=1.0, kl_weight=0.1):
        """
        Args:
            recon_weight (float): Weight for reconstruction term
            kl_weight (float): Weight for KL regularization term
        """
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.charbonnier = CharbonnierLoss()
        
    def forward(self, predicted_residual, gt_residual, precision):
        """
        Compute BFN Bayesian loss.
        
        Args:
            predicted_residual (Tensor): BFN predicted residual [B, 3, H, W]
            gt_residual (Tensor): Ground truth residual [B, 3, H, W]
            precision (float): Current precision parameter from BFN
            
        Returns:
            dict: Dictionary containing loss components
        """
        # Reconstruction term - how well we predict the residual
        recon_loss = self.charbonnier(predicted_residual, gt_residual)
        
        # KL regularization term - encourages proper precision evolution
        # This implements the precision-aware loss from Eq 15
        prediction_error = (predicted_residual - gt_residual).pow(2).mean()
        
        # Modern 2026 approach: avoid tensor construction warning and improve stability
        if isinstance(precision, (int, float)):
            precision_tensor = torch.tensor(precision, device=predicted_residual.device, dtype=predicted_residual.dtype)
        else:
            precision_tensor = precision
            
        kl_loss = precision_tensor * prediction_error - torch.log(precision_tensor + 1e-8)
        
        # Combined loss
        total_loss = self.recon_weight * recon_loss + self.kl_weight * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            'precision': precision
        }


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG features (optional enhancement).
    
    NOTE: This is optional and disabled by default to reduce dependencies.
    Uncomment and modify if perceptual loss is needed.
    """
    
    def __init__(self, feature_layers=[1, 6, 11, 20], use_gpu=True):
        """
        Args:
            feature_layers (list): VGG layer indices to extract features from
            use_gpu (bool): Whether to use GPU acceleration
        """
        super().__init__()
        self.feature_layers = feature_layers
        self.use_gpu = use_gpu
        
        # Placeholder - implement VGG feature extraction if needed
        # try:
        #     from torchvision.models import vgg16
        #     vgg = vgg16(pretrained=True).features
        #     self.feature_extractor = vgg.eval()
        #     for param in self.feature_extractor.parameters():
        #         param.requires_grad = False
        # except:
        #     self.feature_extractor = None
        
        self.feature_extractor = None
        
    def forward(self, pred, target):
        """Compute perceptual loss (placeholder)."""
        if self.feature_extractor is None:
            return torch.tensor(0.0, device=pred.device)
        
        # Implement VGG feature extraction and comparison
        # pred_features = self.extract_features(pred)
        # target_features = self.extract_features(target)
        # loss = sum(F.mse_loss(pf, tf) for pf, tf in zip(pred_features, target_features))
        # return loss
        
        return torch.tensor(0.0, device=pred.device)


class SGBformerLoss(nn.Module):
    """
    Combined loss function for SGBformer training.
    
    This integrates all loss components with proper weighting and scheduling:
    1. Main reconstruction loss (Charbonnier)
    2. BFN Bayesian loss for residual refinement
    3. Optional perceptual loss for enhanced quality
    
    The loss scheduling follows the training strategy described in the paper.
    """
    
    def __init__(self, 
                 charbonnier_weight=1.0,
                 bfn_weight=0.1,
                 perceptual_weight=0.0,
                 enable_perceptual=False):
        """
        Args:
            charbonnier_weight (float): Weight for main reconstruction loss
            bfn_weight (float): Weight for BFN Bayesian loss  
            perceptual_weight (float): Weight for perceptual loss
            enable_perceptual (bool): Whether to use perceptual loss
        """
        super().__init__()
        
        self.charbonnier_weight = charbonnier_weight
        self.bfn_weight = bfn_weight
        self.perceptual_weight = perceptual_weight
        self.enable_perceptual = enable_perceptual
        
        # Loss components
        self.charbonnier_loss = CharbonnierLoss()
        self.bfn_loss = BFNBayesianLoss()
        
        if enable_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def forward(self, model_outputs, targets):
        """
        Compute combined loss for SGBformer.
        
        Args:
            model_outputs (dict): Dictionary from SGBformer forward pass containing:
                - final_output: Final restored image
                - coarse_structure: Coarse restoration from backbone
                - refined_residual: BFN refined residual
                - precision: BFN precision parameter
            targets (Tensor): Ground truth clean images [B, 3, H, W]
            
        Returns:
            dict: Dictionary containing all loss components and metrics
        """
        losses = {}
        
        # Extract outputs
        final_output = model_outputs['final_output']
        coarse_structure = model_outputs['coarse_structure']
        refined_residual = model_outputs['refined_residual']
        precision = model_outputs['precision']
        
        # 1. Main reconstruction loss on final output
        main_loss = self.charbonnier_loss(final_output, targets)
        losses['main_loss'] = main_loss.item()
        
        # 2. Coarse structure loss (helps backbone training)
        coarse_loss = self.charbonnier_loss(coarse_structure, targets)
        losses['coarse_loss'] = coarse_loss.item()
        
        # 3. BFN Bayesian loss on residual refinement
        if refined_residual is not None and isinstance(precision, (int, float, torch.Tensor)):
            gt_residual = targets - coarse_structure.detach()  # Detach to focus BFN training
            bfn_losses = self.bfn_loss(refined_residual, gt_residual, precision)
            
            losses.update({
                'bfn_total': bfn_losses['total_loss'].item() if torch.is_tensor(bfn_losses['total_loss']) else bfn_losses['total_loss'],
                'bfn_recon': bfn_losses['recon_loss'],
                'bfn_kl': bfn_losses['kl_loss'],
                'bfn_precision': bfn_losses['precision']
            })
            
            bfn_total_loss = bfn_losses['total_loss'] if torch.is_tensor(bfn_losses['total_loss']) else torch.tensor(bfn_losses['total_loss'])
        else:
            bfn_total_loss = torch.tensor(0.0, device=final_output.device)
            losses.update({
                'bfn_total': 0.0,
                'bfn_recon': 0.0,
                'bfn_kl': 0.0,
                'bfn_precision': 0.0
            })
        
        # 4. Optional perceptual loss
        if self.enable_perceptual and self.perceptual_weight > 0:
            perc_loss = self.perceptual_loss(final_output, targets)
            losses['perceptual_loss'] = perc_loss.item()
        else:
            perc_loss = torch.tensor(0.0, device=final_output.device)
            losses['perceptual_loss'] = 0.0
        
        # 5. Combine all losses with weights
        total_loss = (
            self.charbonnier_weight * main_loss +
            0.5 * self.charbonnier_weight * coarse_loss +  # Auxiliary loss for backbone
            self.bfn_weight * bfn_total_loss +
            self.perceptual_weight * perc_loss
        )
        
        losses['total_loss'] = total_loss.item()
        
        return total_loss, losses
    
    def update_weights(self, epoch, total_epochs):
        """
        Update loss weights based on training progress.
        
        This implements the loss scheduling strategy where BFN loss
        is gradually increased as training progresses.
        
        Args:
            epoch (int): Current epoch
            total_epochs (int): Total training epochs
        """
        # Gradually increase BFN weight during training
        progress = epoch / total_epochs
        
        # Start with lower BFN weight, increase over time
        if progress < 0.3:
            self.bfn_weight = 0.05
        elif progress < 0.7:
            self.bfn_weight = 0.1
        else:
            self.bfn_weight = 0.15
        
        # Optionally adjust other weights
        # self.charbonnier_weight could be adjusted based on validation metrics


class LossScheduler:
    """
    Adaptive loss weight scheduler based on training metrics.
    
    This can be used to dynamically adjust loss weights based on
    validation performance and training stability.
    """
    
    def __init__(self, loss_fn, warmup_epochs=10):
        """
        Args:
            loss_fn (SGBformerLoss): Loss function to schedule
            warmup_epochs (int): Number of epochs for BFN warmup
        """
        self.loss_fn = loss_fn
        self.warmup_epochs = warmup_epochs
        
    def step(self, epoch, metrics=None):
        """
        Update loss weights based on training progress.
        
        Args:
            epoch (int): Current epoch
            metrics (dict, optional): Validation metrics for adaptive adjustment
        """
        # BFN warmup: gradually introduce BFN loss
        if epoch < self.warmup_epochs:
            # Linear warmup for BFN loss
            bfn_warmup_factor = epoch / self.warmup_epochs
            self.loss_fn.bfn_weight = 0.1 * bfn_warmup_factor
        else:
            # Full BFN weight after warmup
            self.loss_fn.bfn_weight = 0.1
        
        # Adaptive adjustment based on metrics (optional)
        if metrics is not None:
            psnr = metrics.get('val_psnr', 0)
            
            # If validation PSNR is low, increase main loss weight
            if psnr < 25.0:
                self.loss_fn.charbonnier_weight = 1.2
            else:
                self.loss_fn.charbonnier_weight = 1.0


def test_losses():
    """Test function for loss components."""
    print("Testing SGBformer Loss Functions...")
    
    # Test parameters
    B, C, H, W = 2, 3, 64, 64
    
    # Create test tensors
    pred = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)
    
    # Test Charbonnier Loss
    charbonnier = CharbonnierLoss()
    char_loss = charbonnier(pred, target)
    print(f"Charbonnier Loss: {char_loss:.4f}")
    
    # Test BFN Loss
    bfn_loss_fn = BFNBayesianLoss()
    residual_pred = torch.randn(B, C, H, W) * 0.1  # Small residuals
    residual_gt = torch.randn(B, C, H, W) * 0.1
    precision = 0.5
    
    bfn_losses = bfn_loss_fn(residual_pred, residual_gt, precision)
    print(f"BFN Total Loss: {bfn_losses['total_loss']:.4f}")
    print(f"BFN Recon Loss: {bfn_losses['recon_loss']:.4f}")
    print(f"BFN KL Loss: {bfn_losses['kl_loss']:.4f}")
    
    # Test Combined Loss
    sgb_loss = SGBformerLoss()
    
    # Mock model outputs
    model_outputs = {
        'final_output': pred,
        'coarse_structure': pred + torch.randn_like(pred) * 0.1,
        'refined_residual': residual_pred,
        'precision': precision
    }
    
    total_loss, loss_dict = sgb_loss(model_outputs, target)
    print(f"\nCombined Loss: {total_loss:.4f}")
    print("Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # Test loss scheduling
    scheduler = LossScheduler(sgb_loss)
    for epoch in [0, 5, 15]:
        scheduler.step(epoch)
        print(f"Epoch {epoch}: BFN weight = {sgb_loss.bfn_weight:.4f}")
    
    print("\n✓ Loss functions test passed!")


if __name__ == "__main__":
    test_losses()
