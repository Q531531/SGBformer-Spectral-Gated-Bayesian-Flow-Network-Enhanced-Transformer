

import os
import sys
import argparse
import time
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from models import SGBformer
from utils.dataset import get_dataloader


def calculate_metrics(pred, target):
    """Calculate image quality metrics."""
    # Denormalize to [0, 1] for metrics
    pred_01 = (pred + 1.0) / 2.0
    target_01 = (target + 1.0) / 2.0
    
    # PSNR
    mse = F.mse_loss(pred_01, target_01)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
    
    # SSIM (simplified version)
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SGBformer Testing/Inference')
    
    # Input/Output
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument("--input", type=str, help="Input image or directory (not required for --demo)")
    parser.add_argument("--output", type=str, required=True, help="Output path or directory")
    parser.add_argument('--gt_dir', type=str, default=None,
                       help='Ground truth directory for quantitative evaluation')
    
    # Model arguments (override checkpoint config if needed)
    parser.add_argument('--dim', type=int, default=None,
                       help='Base feature dimension')
    parser.add_argument('--num_blocks', type=int, nargs='+', default=None,
                       help='Number of blocks at each level')
    parser.add_argument('--heads', type=int, nargs='+', default=None,
                       help='Number of attention heads at each level')
    parser.add_argument('--bfn_steps', type=int, default=None,
                       help='Number of BFN refinement steps for inference')
    parser.add_argument('--enable_semantic', dest='enable_semantic', action='store_true',
                      help='Enable semantic guidance (default: disabled)')
    parser.add_argument('--disable_semantic', dest='enable_semantic', action='store_false',
                      help='Disable semantic guidance (default: disabled)')
    parser.set_defaults(enable_semantic=None)
    parser.add_argument('--clip_model_name', type=str, default=None,
                       help='Override CLIP model name (Hugging Face)')
    parser.add_argument('--clip_image_size', type=int, default=None,
                       help='Override CLIP input image size')
    
    # Processing options
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size (images will be resized)')
    parser.add_argument('--preserve_aspect', action='store_true',
                       help='Preserve aspect ratio during resize')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing multiple images')
    
    # Evaluation options
    parser.add_argument('--save_intermediate', action='store_true',
                       help='Save intermediate outputs (coarse, residual, etc.)')
    parser.add_argument('--compute_metrics', action='store_true',
                       help='Compute quantitative metrics (requires --gt_dir)')
    parser.add_argument('--save_comparisons', action='store_true',
                       help='Save comparison grids (input/output/gt)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    # Demo mode
    parser.add_argument('--demo', action='store_true',
                       help='Demo mode with synthetic test images')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup computation device."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    return torch.device(device)


def load_model(checkpoint_path, device, args):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Get model configuration from checkpoint or args
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # Fallback to default config
        config = {
            'dim': 32,
            'num_blocks': [2, 3, 3, 4],
            'heads': [1, 2, 4, 8],
            'bfn_steps': 10,
            'enable_semantic_guidance': False,
            'clip_model_name': 'openai/clip-vit-base-patch32',
            'clip_image_size': 224
        }

    config.setdefault('enable_semantic_guidance', False)
    config.setdefault('clip_model_name', 'openai/clip-vit-base-patch32')
    config.setdefault('clip_image_size', 224)
    
    # Override with command line arguments if provided
    if args.dim is not None:
        config['dim'] = args.dim
    if args.num_blocks is not None:
        config['num_blocks'] = args.num_blocks
    if args.heads is not None:
        config['heads'] = args.heads
    if args.bfn_steps is not None:
        config['bfn_steps'] = args.bfn_steps
    if args.enable_semantic is not None:
        config['enable_semantic_guidance'] = args.enable_semantic
    if args.clip_model_name is not None:
        config['clip_model_name'] = args.clip_model_name
    if args.clip_image_size is not None:
        config['clip_image_size'] = args.clip_image_size
    
    # Create model
    model = SGBformer(
        dim=config['dim'],
        num_blocks=config['num_blocks'],
        heads=config['heads'],
        bfn_steps=config['bfn_steps'],
        enable_semantic_guidance=config.get('enable_semantic_guidance', False),
        clip_model_name=config.get('clip_model_name', 'openai/clip-vit-base-patch32'),
        clip_image_size=config.get('clip_image_size', 224)
    ).to(device)
    
    # Load weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        # Try alternative key names
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            print("Error: Could not find model weights in checkpoint")
            sys.exit(1)
    
    model.eval()
    
    # Print model info
    print("Model loaded successfully!")
    print(f"  Configuration: {config}")
    
    if 'epoch' in checkpoint:
        print(f"  Trained epochs: {checkpoint['epoch'] + 1}")
    if 'metrics' in checkpoint and 'val_psnr' in checkpoint['metrics']:
        print(f"  Best PSNR: {checkpoint['metrics']['val_psnr']:.2f} dB")
    
    return model


def preprocess_image(image_path, image_size, preserve_aspect=False):
    """Load and preprocess single image."""
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None
    
    original_size = image.size
    
    if preserve_aspect:
        # Resize while preserving aspect ratio
        aspect = original_size[0] / original_size[1]
        if aspect > 1:
            new_size = (image_size, int(image_size / aspect))
        else:
            new_size = (int(image_size * aspect), image_size)
        image = image.resize(new_size, Image.BICUBIC)
        
        # Pad to target size
        padded_image = Image.new('RGB', (image_size, image_size), (128, 128, 128))
        paste_x = (image_size - new_size[0]) // 2
        paste_y = (image_size - new_size[1]) // 2
        padded_image.paste(image, (paste_x, paste_y))
        image = padded_image
    else:
        # Direct resize
        image = image.resize((image_size, image_size), Image.BICUBIC)
    
    # Convert to tensor and normalize to [-1, 1]
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    image_tensor = image_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
    
    return image_tensor.unsqueeze(0), original_size


def postprocess_output(output_tensor, original_size=None):
    """Convert model output back to PIL image."""
    # Denormalize from [-1, 1] to [0, 1]
    output = (output_tensor + 1.0) / 2.0
    output = torch.clamp(output, 0, 1)
    
    # Convert to numpy
    if output.dim() == 4:
        output = output.squeeze(0)
    output = output.permute(1, 2, 0).cpu().numpy()
    
    # Convert to PIL image
    output = (output * 255).astype(np.uint8)
    image = Image.fromarray(output)
    
    # Resize to original size if specified
    if original_size is not None:
        image = image.resize(original_size, Image.BICUBIC)
    
    return image


def process_single_image(model, image_path, output_dir, args, device):
    """Process single image with the model."""
    # Load and preprocess
    input_tensor, original_size = preprocess_image(
        image_path, args.image_size, args.preserve_aspect
    )
    
    if input_tensor is None:
        return None
    
    input_tensor = input_tensor.to(device)
    
    # Model inference
    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_tensor)
        inference_time = time.time() - start_time
    
    # Extract results
    final_output = outputs['final_output']
    
    # Save main result
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_restored.png")
    
    restored_image = postprocess_output(final_output, original_size if not args.preserve_aspect else None)
    restored_image.save(output_path)
    
    results = {
        'output_path': output_path,
        'inference_time': inference_time
    }
    
    # Save intermediate results if requested
    if args.save_intermediate:
        coarse_output = outputs['coarse_structure']
        coarse_image = postprocess_output(coarse_output, original_size if not args.preserve_aspect else None)
        coarse_path = os.path.join(output_dir, f"{filename}_coarse.png")
        coarse_image.save(coarse_path)
        results['coarse_path'] = coarse_path
        
        if outputs['refined_residual'] is not None:
            # Visualize residual (scaled for visibility)
            residual = outputs['refined_residual']
            residual_vis = torch.clamp(residual * 5 + 0.5, 0, 1)  # Scale and shift
            residual_image = postprocess_output(residual_vis, original_size if not args.preserve_aspect else None)
            residual_path = os.path.join(output_dir, f"{filename}_residual.png")
            residual_image.save(residual_path)
            results['residual_path'] = residual_path
    
    print(f"  Processed: {os.path.basename(image_path)} ({inference_time:.3f}s)")
    return results


def create_comparison_grid(input_path, output_path, gt_path=None):
    """Create comparison grid image."""
    images = []
    labels = []
    
    # Load input
    input_img = Image.open(input_path).convert('RGB')
    images.append(input_img)
    labels.append('Input')
    
    # Load output
    output_img = Image.open(output_path).convert('RGB')
    images.append(output_img)
    labels.append('SGBformer')
    
    # Load ground truth if available
    if gt_path and os.path.exists(gt_path):
        gt_img = Image.open(gt_path).convert('RGB')
        images.append(gt_img)
        labels.append('Ground Truth')
    
    # Create grid
    img_width, img_height = images[0].size
    grid_width = img_width * len(images)
    grid_height = img_height + 30  # Extra space for labels
    
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    for i, (img, label) in enumerate(zip(images, labels)):
        # Paste image
        x_offset = i * img_width
        grid.paste(img, (x_offset, 0))
        
        # Add label (simplified text)
        # Note: For proper text rendering, consider using PIL.ImageDraw with fonts
    
    return grid


def run_demo(model, device, output_dir, args):
    """Run demo with synthetic test images."""
    print("Running demo mode with synthetic test images...")
    
    # Generate a few synthetic test images
    from utils.dataset import AllWeatherDataset
    
    dataset = AllWeatherDataset(split='test', num_samples=5, image_size=args.image_size)
    
    metrics_list = []
    
    for i in range(len(dataset)):
        degraded, clean = dataset[i]
        
        # Process with model
        degraded_batch = degraded.unsqueeze(0).to(device)
        clean_batch = clean.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(degraded_batch)
            final_output = outputs['final_output']
        
        # Calculate metrics
        metrics = calculate_metrics(final_output, clean_batch)
        metrics_list.append(metrics)
        
        # Save results
        filename = f"demo_{i:03d}"
        
        # Save images
        input_img = postprocess_output(degraded)
        input_img.save(os.path.join(output_dir, f"{filename}_input.png"))
        
        output_img = postprocess_output(final_output)
        output_img.save(os.path.join(output_dir, f"{filename}_output.png"))
        
        gt_img = postprocess_output(clean)
        gt_img.save(os.path.join(output_dir, f"{filename}_gt.png"))
        
        print(f"  Demo {i+1}: PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}")
    
    # Average metrics
    avg_psnr = sum(m['psnr'] for m in metrics_list) / len(metrics_list)
    avg_ssim = sum(m['ssim'] for m in metrics_list) / len(metrics_list)
    
    print(f"\nDemo Results:")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average SSIM: {avg_ssim:.4f}")
    
    return {
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'individual_metrics': metrics_list
    }


def main():
    """Main testing function."""
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 80)
    print("SGBformer Inference")
    print("=" * 80)
    
    # Load model
    model = load_model(args.checkpoint, device, args)
    
    if args.demo:
        # Demo mode
        results = run_demo(model, device, args.output, args)
        
        # Save results
        results_path = os.path.join(args.output, 'demo_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
    else:
        # Process input images - validate input is provided
        if not args.input:
            print("Error: --input is required when not using --demo mode")
            return
            
        if os.path.isfile(args.input):
            # Single image
            print(f"Processing single image: {args.input}")
            result = process_single_image(model, args.input, args.output, args, device)
            print(f"Output saved: {result['output_path']}")
            
        elif os.path.isdir(args.input):
            # Directory of images
            print(f"Processing directory: {args.input}")
            
            # Get all image files
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            for ext in extensions:
                image_files.extend([
                    os.path.join(args.input, f) for f in os.listdir(args.input)
                    if f.lower().endswith(ext)
                ])
            
            print(f"Found {len(image_files)} images")
            
            # Process each image
            results = []
            total_time = 0
            
            for image_path in image_files:
                result = process_single_image(model, image_path, args.output, args, device)
                if result:
                    results.append(result)
                    total_time += result['inference_time']
            
            avg_time = total_time / len(results) if results else 0
            print(f"\nProcessing completed:")
            print(f"  Total images: {len(results)}")
            print(f"  Average time: {avg_time:.3f}s per image")
            print(f"  Outputs saved in: {args.output}")
            
        else:
            print(f"Error: Input path does not exist: {args.input}")
            sys.exit(1)
    
    print("=" * 80)
    print("Inference completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
