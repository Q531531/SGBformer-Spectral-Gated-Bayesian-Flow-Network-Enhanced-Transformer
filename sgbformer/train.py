

import os
import sys
import argparse
import time
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import SGBformer
from utils.dataset import get_dataloader
from utils.losses import SGBformerLoss, LossScheduler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SGBformer Training')
    
    # Dataset arguments
    parser.add_argument('--data_root', type=str, default='dummy', 
                       help='Path to dataset root (use "dummy" for synthetic data)')
    parser.add_argument('--dataset_type', type=str, default='synthetic', 
                       choices=['synthetic', 'real'],
                       help='Dataset type: synthetic or real')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of synthetic samples (if using synthetic dataset)')
    
    # Model arguments
    parser.add_argument('--dim', type=int, default=32,
                       help='Base feature dimension (lightweight: 32)')
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[2, 3, 3, 4],
                       help='Number of blocks at each level')
    parser.add_argument('--heads', type=int, nargs='+', default=[1, 2, 4, 8],
                      help='Number of attention heads at each level')
    parser.add_argument('--bfn_steps', type=int, default=10,
                      help='Number of BFN refinement steps')
    parser.add_argument('--enable_semantic', dest='enable_semantic', action='store_true',
                      help='Enable semantic guidance (default: disabled)')
    parser.add_argument('--disable_semantic', dest='enable_semantic', action='store_false',
                      help='Disable semantic guidance (default: disabled)')
    parser.set_defaults(enable_semantic=False)
    parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-base-patch32',
                      help='Hugging Face CLIP model name')
    parser.add_argument('--clip_image_size', type=int, default=224,
                      help='CLIP input image size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='BFN warmup epochs')
    
    # Loss weights
    parser.add_argument('--charbonnier_weight', type=float, default=1.0,
                       help='Weight for Charbonnier loss')
    parser.add_argument('--bfn_weight', type=float, default=0.1,
                       help='Weight for BFN loss')
    parser.add_argument('--perceptual_weight', type=float, default=0.0,
                       help='Weight for perceptual loss')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval (iterations)')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Checkpoint saving interval (epochs)')
    parser.add_argument('--val_interval', type=int, default=5,
                       help='Validation interval (epochs)')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with reduced dataset and epochs')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup computation device."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return torch.device(device)


def calculate_metrics(pred, target):
    """Calculate image quality metrics."""
    # Denormalize to [0, 1] for metrics
    pred_01 = (pred + 1.0) / 2.0
    target_01 = (target + 1.0) / 2.0
    
    # PSNR
    mse = F.mse_loss(pred_01, target_01)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
    
    # SSIM (simplified version)
    # Note: For full SSIM, consider using external libraries like pytorch-ssim
    mu_pred = F.avg_pool2d(pred_01, 3, 1, 1)
    mu_target = F.avg_pool2d(target_01, 3, 1, 1)
    
    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = F.avg_pool2d(pred_01 * pred_01, 3, 1, 1) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d(target_01 * target_01, 3, 1, 1) - mu_target_sq
    sigma_pred_target = F.avg_pool2d(pred_01 * target_01, 3, 1, 1) - mu_pred_target
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    ssim = ssim_map.mean()
    
    return {
        'psnr': psnr.item(),
        'ssim': ssim.item()
    }


def validate(model, val_loader, device, epoch):
    """Run validation."""
    model.eval()
    val_metrics = defaultdict(list)
    
    with torch.no_grad():
        for batch_idx, (degraded, clean) in enumerate(val_loader):
            if batch_idx >= 20:  # Limit validation samples for speed
                break
                
            degraded = degraded.to(device)
            clean = clean.to(device)
            
            # Forward pass
            outputs = model(degraded)
            final_output = outputs['final_output']
            
            # Calculate metrics
            metrics = calculate_metrics(final_output, clean)
            val_metrics['psnr'].append(metrics['psnr'])
            val_metrics['ssim'].append(metrics['ssim'])
    
    # Average metrics
    if len(val_metrics['psnr']) == 0:
        print("Warning: no validation samples available; skipping metrics.")
        model.train()
        return {'val_psnr': 0.0, 'val_ssim': 0.0}

    avg_metrics = {
        'val_psnr': sum(val_metrics['psnr']) / len(val_metrics['psnr']),
        'val_ssim': sum(val_metrics['ssim']) / len(val_metrics['ssim'])
    }
    
    model.train()
    return avg_metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'model_config': {
            'dim': getattr(model, 'dim', 32),
            'num_blocks': getattr(model, 'num_blocks', [2, 3, 3, 4]),
            'heads': getattr(model, 'heads', [1, 2, 4, 8]),
            'bfn_steps': getattr(model, 'bfn_steps', 10),
            'enable_semantic_guidance': getattr(model, 'enable_semantic_guidance', False),
            'clip_model_name': getattr(model, 'clip_model_name', 'openai/clip-vit-base-patch32'),
            'clip_image_size': getattr(model, 'clip_image_size', 224)
        }
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def train():
    """Main training function."""
    args = parse_args()
    # Debug mode adjustments
    if args.debug:
        args.epochs = 5
        args.num_samples = 100
        args.log_interval = 10
        args.val_interval = 2
        print("Debug mode enabled: reduced epochs and samples")
    
    # Setup device
    device = setup_device(args.device)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 80)
    print("SGBformer Training")
    print("=" * 80)
    print(f"Dataset: {args.dataset_type} ({args.num_samples if args.dataset_type == 'synthetic' else 'real'} samples)")
    print(f"Model: dim={args.dim}, blocks={args.num_blocks}, heads={args.heads}")
    print(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"BFN: {args.bfn_steps} steps, warmup={args.warmup_epochs} epochs")
    print("=" * 80)
    
    # Create dataloaders
    train_loader = get_dataloader(
        dataset_type=args.dataset_type,
        data_root=args.data_root,
        split='train',
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_samples=args.num_samples
    )
    
    val_loader = get_dataloader(
        dataset_type=args.dataset_type,
        data_root=args.data_root,
        split='val',
        batch_size=1,
        image_size=args.image_size,
        num_samples=min(100, args.num_samples // 5)  # Smaller validation set
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    model = SGBformer(
        dim=args.dim,
        num_blocks=args.num_blocks,
        heads=args.heads,
        bfn_steps=args.bfn_steps,
        enable_semantic_guidance=args.enable_semantic,
        clip_model_name=args.clip_model_name,
        clip_image_size=args.clip_image_size
    ).to(device)
    
    # Store config in model for checkpointing
    model.dim = args.dim
    model.num_blocks = args.num_blocks
    model.heads = args.heads
    model.bfn_steps = args.bfn_steps
    model.enable_semantic_guidance = args.enable_semantic
    model.clip_model_name = args.clip_model_name
    model.clip_image_size = args.clip_image_size
    
    
    # Create loss function and scheduler
    criterion = SGBformerLoss(
        charbonnier_weight=args.charbonnier_weight,
        bfn_weight=args.bfn_weight,
        perceptual_weight=args.perceptual_weight,
        enable_perceptual=args.perceptual_weight > 0
    )
    
    loss_scheduler = LossScheduler(criterion, warmup_epochs=args.warmup_epochs)
    
    # Create optimizer and learning rate scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Training loop
    best_psnr = 0.0
    total_iterations = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = defaultdict(list)
        
        # Update loss weights
        loss_scheduler.step(epoch)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"BFN weight: {criterion.bfn_weight:.4f}")
        
        epoch_start_time = time.time()
        
        for batch_idx, (degraded, clean) in enumerate(train_loader):
            degraded = degraded.to(device)
            clean = clean.to(device)
            
            # Forward pass
            outputs = model(degraded, clean)
            
            # Compute loss
            loss, loss_dict = criterion(outputs, clean)
            
            # Modern 2026 approach: robust backward pass with NaN protection
            optimizer.zero_grad()
            
            # Check for NaN loss before backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss detected at batch {batch_idx}, skipping")
                continue
                
            loss.backward()
            
            # Aggressive gradient clipping for stability
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            # Skip update if gradient norm is too large (gradient explosion protection)
            if grad_norm > 10.0:
                print(f"  WARNING: Large gradient norm {grad_norm:.2f}, skipping update")
                continue
                
            optimizer.step()
            
            # Logging
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)
            
            total_iterations += 1
            
            if batch_idx % args.log_interval == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss={loss.item():.4f}, "
                      f"Main={loss_dict['main_loss']:.4f}, "
                      f"BFN={loss_dict['bfn_total']:.4f}")
        
        # End of epoch
        lr_scheduler.step()
        
        # Calculate epoch averages
        epoch_avg = {key: sum(values) / len(values) for key, values in epoch_losses.items()}
        epoch_time = time.time() - epoch_start_time
        
        print(f"  Epoch {epoch+1} Summary ({epoch_time:.1f}s):")
        print(f"    Total Loss: {epoch_avg['total_loss']:.4f}")
        print(f"    Main Loss: {epoch_avg['main_loss']:.4f}")
        print(f"    Coarse Loss: {epoch_avg['coarse_loss']:.4f}")
        print(f"    BFN Loss: {epoch_avg['bfn_total']:.4f}")
        print(f"    BFN Precision: {epoch_avg['bfn_precision']:.4f}")
        
        # Validation
        if (epoch + 1) % args.val_interval == 0:
            print("  Running validation...")
            val_metrics = validate(model, val_loader, device, epoch)
            
            print(f"    Val PSNR: {val_metrics['val_psnr']:.2f} dB")
            print(f"    Val SSIM: {val_metrics['val_ssim']:.4f}")
            
            # Save best model
            if val_metrics['val_psnr'] > best_psnr:
                best_psnr = val_metrics['val_psnr']
                save_checkpoint(
                    model, optimizer, lr_scheduler, epoch,
                    {**epoch_avg, **val_metrics},
                    os.path.join(args.save_dir, 'best_model.pth')
                )
                print(f"    New best PSNR: {best_psnr:.2f} dB")
        
        # Regular checkpoint saving
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, lr_scheduler, epoch,
                epoch_avg,
                os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
    
    # Final checkpoint
    save_checkpoint(
        model, optimizer, lr_scheduler, args.epochs - 1,
        epoch_avg,
        os.path.join(args.save_dir, 'final_model.pth')
    )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved in: {args.save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    train()
