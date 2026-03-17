# SGBformer: Spectral-Gated Bayesian Flow Network Enhanced Transformer for All-Weather Image Restoration
This repository contains the PyTorch implementation of SGBformer, a novel transformer-based architecture for all-weather image restoration. 
![结构图](/结构图.png)
## Highlights

- Spectral-gated backbone + BFN residual refinement.
- Optional CLIP semantic guidance (disabled by default).
- Demo mode runs without real data.
```
## Quick Start   Demo Inference  
Test the model with synthetic images:
```bash
# First train a demo model
python train.py --debug --save_dir ./demo_checkpoints

# Then run inference
python test.py --checkpoint ./demo_checkpoints/best_model.pth --demo --output ./demo_results
```

## Usage

### Training

**Basic training with synthetic data:**
```bash
python train.py \
    --dataset_type synthetic \
    --epochs 100 \
    --batch_size 4 \
    --lr 2e-4 \
    --save_dir ./checkpoints
```

**Training with real All-Weather dataset:**
```bash
python train.py \
    --dataset_type real \
    --data_root /path/to/allweather/dataset \
    --epochs 100 \
    --batch_size 4 \
    --lr 2e-4 \
    --save_dir ./checkpoints
```

**Key training arguments:**
- `--dim`: Base feature dimension (default: 32 for lightweight)
- `--num_blocks`: Blocks per level (default: [2,3,3,4])
- `--bfn_steps`: BFN refinement steps (default: 10)
- `--warmup_epochs`: BFN loss warmup period (default: 10)

### Inference

**Single image restoration:**
```bash
python test.py \
    --checkpoint ./checkpoints/best_model.pth \
    --input /path/to/degraded/image.jpg \
    --output ./restored_image.png
```

**Batch processing:**
```bash
python test.py \
    --checkpoint ./checkpoints/best_model.pth \
    --input /path/to/degraded/images/ \
    --output ./restored_images/
```

**With intermediate outputs:**
```bash
python test.py \
    --checkpoint ./checkpoints/best_model.pth \
    --input /path/to/image.jpg \
    --output ./results/ \
    --save_intermediate
```

## Dataset

Real data should be organized as:
```
dataset_root/
├── train/input, train/gt
├── val/input, val/gt
└── test/input, test/gt
```

Run training with:
```bash
python train.py --dataset_type real --data_root /path/to/dataset
```

## CLIP (Semantic Guidance)

- **Default: disabled** (no CLIP download on `--debug`).
- Enable with `--enable_semantic`.
- If enabled, real CLIP is used when `transformers` is installed; otherwise fallback to Mock.
- First run downloads weights to `~/.cache/huggingface` (override with `HF_HOME`).

## Troubleshooting

- **Dataset not found**: check `data_root/<split>/input` and `data_root/<split>/gt`.
- **OOM**: reduce `--batch_size` or `--dim`.
- **Demo check**: `python train.py --debug`.
