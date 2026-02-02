"""
Dataset Module for All-Weather Image Restoration

This module provides dummy dataset loaders for the All-Weather dataset
to enable code testing and demonstration. For actual training, replace
with real dataset paths and preprocessing pipelines.

Key Components:
- AllWeatherDataset: Main dataset class with synthetic weather degradations
- DataLoader utilities with proper transformations
- Synthetic degradation functions for demonstration
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import random


class AllWeatherDataset(Dataset):
    """
    Dummy All-Weather Dataset for code demonstration.
    
    NOTE: This generates synthetic degraded/clean image pairs for testing.
    For actual training, replace with real All-Weather dataset loading logic
    that reads from proper train/val/test splits.
    """
    
    def __init__(self, 
                 data_root="dummy", 
                 split="train", 
                 image_size=256,
                 num_samples=1000):
        """
        Args:
            data_root (str): Root directory (dummy for synthetic data)
            split (str): Dataset split ('train', 'val', 'test')
            image_size (int): Target image size
            num_samples (int): Number of synthetic samples to generate
        """
        self.split = split
        self.image_size = image_size
        self.num_samples = num_samples
        
        # Transform for clean images (normalized to [-1, 1])
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Weather degradation types
        self.weather_types = ['rain', 'snow', 'fog', 'haze']
        
        print(f"Initialized {split} dataset with {num_samples} synthetic samples")
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        """
        Generate synthetic degraded/clean image pair.
        
        Returns:
            tuple: (degraded_image, clean_image) both in [-1, 1] range
        """
        # Generate clean image (random colored patches)
        clean_image = self._generate_clean_image()
        
        # Apply random weather degradation
        weather_type = random.choice(self.weather_types)
        degraded_image = self._apply_weather_degradation(clean_image, weather_type)
        
        # Apply transforms
        if self.transform:
            clean_tensor = self.transform(clean_image)
            degraded_tensor = self.transform(degraded_image)
        
        return degraded_tensor, clean_tensor
    
    def _generate_clean_image(self):
        """Generate synthetic clean image with varied content."""
        # Create base image with random colors and gradients
        img_array = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Add random colored rectangles
        num_rects = random.randint(3, 8)
        for _ in range(num_rects):
            x1, y1 = random.randint(0, self.image_size//2), random.randint(0, self.image_size//2)
            x2, y2 = random.randint(x1, self.image_size), random.randint(y1, self.image_size)
            color = [random.randint(0, 255) for _ in range(3)]
            img_array[y1:y2, x1:x2] = color
        
        # Add gradient overlay
        gradient = np.linspace(0, 1, self.image_size)
        for i in range(3):
            img_array[:, :, i] = np.clip(
                img_array[:, :, i] * (0.7 + 0.3 * gradient[None, :]),
                0, 255
            ).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def _apply_weather_degradation(self, clean_image, weather_type):
        """Apply synthetic weather degradation."""
        img_array = np.array(clean_image).astype(np.float32)
        
        if weather_type == 'rain':
            # Add rain streaks
            degraded = self._add_rain_streaks(img_array)
        elif weather_type == 'snow':
            # Add snow particles  
            degraded = self._add_snow_particles(img_array)
        elif weather_type == 'fog':
            # Add atmospheric fog
            degraded = self._add_fog_effect(img_array)
        elif weather_type == 'haze':
            # Add haze/pollution
            degraded = self._add_haze_effect(img_array)
        else:
            degraded = img_array
            
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        return Image.fromarray(degraded)
    
    def _add_rain_streaks(self, img):
        """Add synthetic rain streaks."""
        h, w, c = img.shape
        rain_img = img.copy()
        
        # Generate random rain streaks
        num_streaks = random.randint(50, 200)
        for _ in range(num_streaks):
            # Random streak parameters
            x = random.randint(0, w-1)
            y_start = random.randint(0, h//3)
            length = random.randint(10, 30)
            thickness = random.randint(1, 2)
            opacity = random.uniform(0.3, 0.8)
            
            # Draw streak
            for i in range(length):
                y = y_start + i
                x_offset = random.randint(-1, 1)  # Slight angle variation
                x_pos = np.clip(x + x_offset, 0, w-1)
                y_pos = np.clip(y, 0, h-1)
                
                if y_pos < h and x_pos < w:
                    # Brighten pixels to simulate rain
                    rain_img[y_pos, max(0,x_pos-thickness):min(w,x_pos+thickness+1)] = \
                        img[y_pos, max(0,x_pos-thickness):min(w,x_pos+thickness+1)] * (1-opacity) + \
                        255 * opacity
        
        return rain_img
    
    def _add_snow_particles(self, img):
        """Add synthetic snow particles."""
        h, w, c = img.shape
        snow_img = img.copy()
        
        # Generate random snow particles
        num_particles = random.randint(100, 500)
        for _ in range(num_particles):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            size = random.randint(1, 4)
            opacity = random.uniform(0.6, 1.0)
            
            # Draw circular snow particle
            y_min, y_max = max(0, y-size), min(h, y+size+1)
            x_min, x_max = max(0, x-size), min(w, x+size+1)
            
            snow_img[y_min:y_max, x_min:x_max] = \
                img[y_min:y_max, x_min:x_max] * (1-opacity) + 255 * opacity
        
        return snow_img
    
    def _add_fog_effect(self, img):
        """Add synthetic fog effect."""
        # Global atmospheric scattering
        fog_strength = random.uniform(0.3, 0.7)
        atmospheric_light = np.array([200, 210, 220])  # Bluish-white
        
        fogged = img * (1 - fog_strength) + atmospheric_light * fog_strength
        return fogged
    
    def _add_haze_effect(self, img):
        """Add synthetic haze effect."""
        # Yellowish haze with reduced contrast
        haze_strength = random.uniform(0.2, 0.5)
        haze_color = np.array([240, 230, 200])  # Yellowish
        
        hazed = img * (1 - haze_strength) + haze_color * haze_strength
        # Reduce contrast
        hazed = hazed * 0.8 + 128 * 0.2
        return hazed


class RealAllWeatherDataset(Dataset):
    """
    Template for real All-Weather dataset loading.
    
    NOTE: This is a template showing how to implement loading from real
    All-Weather dataset files. Uncomment and modify paths as needed.
    """
    
    def __init__(self, data_root, split="train", image_size=256):
        """
        Args:
            data_root (str): Path to All-Weather dataset root
            split (str): Dataset split ('train', 'val', 'test')
            image_size (int): Target image size
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        
        # Define paths based on All-Weather dataset structure
        self.degraded_dir = os.path.join(data_root, split, 'input')
        self.clean_dir = os.path.join(data_root, split, 'gt')
        
        if not os.path.isdir(self.degraded_dir) or not os.path.isdir(self.clean_dir):
            raise FileNotFoundError(
                f"Real dataset not found. Expected '{self.degraded_dir}' and '{self.clean_dir}'."
            )
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Load file lists and filter by common image extensions
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.image_files = sorted([
            f for f in os.listdir(self.degraded_dir)
            if f.lower().endswith(valid_exts)
        ])
        
        # Verify clean images exist
        self.image_files = [
            f for f in self.image_files
            if os.path.exists(os.path.join(self.clean_dir, f))
        ]
        
        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No paired images found in '{self.degraded_dir}' and '{self.clean_dir}'."
            )
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        """
        Load real degraded/clean image pair.
        
        NOTE: Assumes paired filenames in input/ and gt/.
        """
        filename = self.image_files[idx]
        degraded_path = os.path.join(self.degraded_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)
        
        degraded_img = Image.open(degraded_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')
        
        if self.transform:
            degraded_tensor = self.transform(degraded_img)
            clean_tensor = self.transform(clean_img)
        else:
            degraded_tensor = transforms.ToTensor()(degraded_img)
            clean_tensor = transforms.ToTensor()(clean_img)
        
        return degraded_tensor, clean_tensor


def get_dataloader(dataset_type="synthetic", 
                   data_root="dummy",
                   split="train",
                   batch_size=4,
                   image_size=256,
                   num_workers=4,
                   num_samples=1000):
    """
    Create dataloader for All-Weather dataset.
    
    Args:
        dataset_type (str): "synthetic" or "real"
        data_root (str): Path to dataset root
        split (str): Dataset split
        batch_size (int): Batch size
        image_size (int): Target image size
        num_workers (int): Number of dataloader workers
        num_samples (int): Number of synthetic samples (if synthetic)
        
    Returns:
        DataLoader: Configured dataloader
    """
    if dataset_type == "synthetic":
        dataset = AllWeatherDataset(
            data_root=data_root,
            split=split,
            image_size=image_size,
            num_samples=num_samples
        )
    else:
        dataset = RealAllWeatherDataset(
            data_root=data_root,
            split=split,
            image_size=image_size
        )
    
    if len(dataset) == 0:
        raise RuntimeError(
            f"Dataset split '{split}' is empty. Check data_root='{data_root}'."
        )
    
    shuffle = (split == "train")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle
    )
    
    return dataloader


def test_dataset():
    """Test function for dataset functionality."""
    print("Testing AllWeather Dataset...")
    
    # Test synthetic dataset
    dataset = AllWeatherDataset(split="train", num_samples=10)
    
    # Test single sample
    degraded, clean = dataset[0]
    print(f"Degraded image shape: {degraded.shape}")
    print(f"Clean image shape: {clean.shape}")
    print(f"Degraded range: [{degraded.min():.2f}, {degraded.max():.2f}]")
    print(f"Clean range: [{clean.min():.2f}, {clean.max():.2f}]")
    
    # Test dataloader
    dataloader = get_dataloader(
        dataset_type="synthetic",
        split="train", 
        batch_size=4,
        num_samples=20
    )
    
    batch_degraded, batch_clean = next(iter(dataloader))
    print(f"Batch degraded shape: {batch_degraded.shape}")
    print(f"Batch clean shape: {batch_clean.shape}")
    
    print("✓ Dataset test passed!")


if __name__ == "__main__":
    test_dataset()
