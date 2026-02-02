"""
Degradation-Aware Cross-Attention (DAC) with CLIP Module

This module implements the degradation-aware cross-attention mechanism that uses
semantic understanding to guide restoration. The module leverages CLIP embeddings
to provide weather-type awareness for adaptive restoration.

Key Components:
- Mock CLIP Encoder (placeholder for frozen CLIP in actual implementation)
- Cross-Attention mechanism using semantic vectors as K,V and image features as Q
- Multi-scale semantic guidance integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from transformers import CLIPModel
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    CLIPModel = None
    _TRANSFORMERS_AVAILABLE = False


class MockCLIPEncoder(nn.Module):
    """
    Mock CLIP Encoder for code submission.
    
    NOTE: This is a placeholder implementation that generates random embeddings
    to ensure the code runs without downloading large pretrained weights.
    
    In the actual implementation, this should be replaced with a frozen
    pretrained CLIP image encoder (e.g., ViT-B/32) that outputs 512-dim embeddings.
    """
    
    def __init__(self, embed_dim=512):
        """
        Args:
            embed_dim (int): Dimension of output embeddings (default: 512 for CLIP)
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Lightweight CNN to simulate CLIP image encoding
        # This mimics the structure but uses random initialization
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3),     # 224->112
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),    # 112->56
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),   # 56->28
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),  # 28->14
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Initialize to produce reasonable embeddings
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to produce normalized embeddings like CLIP."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                
    def forward(self, x):
        """
        Extract semantic embeddings from degraded images.
        
        Args:
            x (Tensor): Input images [B, 3, H, W] in range [-1, 1]
            
        Returns:
            Tensor: Semantic embeddings [B, embed_dim]
        """
        # Normalize input to [0, 1] range expected by vision models
        x_norm = (x + 1.0) / 2.0
        
        # Extract features
        embeddings = self.encoder(x_norm)
        
        # L2 normalize like CLIP embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings


class RealCLIPEncoder(nn.Module):
    """
    Real CLIP image encoder using Hugging Face transformers.

    This replaces MockCLIPEncoder for production use. We keep CLIP frozen
    and only use its image embeddings for semantic guidance.
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", image_size=224):
        super().__init__()

        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for RealCLIPEncoder. Install with `pip install transformers`."
            )

        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.image_size = image_size
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input images [B, 3, H, W] in range [-1, 1]

        Returns:
            Tensor: CLIP embeddings [B, 512]
        """
        # Convert to [0, 1]
        x = (x + 1.0) / 2.0

        # Resize to CLIP resolution
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)

        # Normalize with CLIP mean/std
        x = (x - self.clip_mean) / self.clip_std

        with torch.no_grad():
            embeddings = self.model.get_image_features(pixel_values=x)

        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings


class DegradationAwareCrossAttention(nn.Module):
    """
    Degradation-Aware Cross-Attention (DAC) Module.
    
    This implements the semantic guidance mechanism described in Eq 11-13 of the paper.
    The key idea is to use CLIP semantic vectors as Keys and Values in cross-attention,
    with image features as Queries, enabling weather-type aware feature modulation.
    """
    
    def __init__(self, feature_dim, semantic_dim=512, num_heads=8, dropout=0.1):
        """
        Args:
            feature_dim (int): Dimension of input image features
            semantic_dim (int): Dimension of semantic embeddings (512 for CLIP)
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.semantic_dim = semantic_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Query projection from image features
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        
        # Key and Value projections from semantic embeddings
        self.k_proj = nn.Linear(semantic_dim, feature_dim)
        self.v_proj = nn.Linear(semantic_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Temperature scaling for attention
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * (self.head_dim ** -0.5))
        
    def forward(self, image_features, semantic_embeddings):
        """
        Cross-attention between image features and semantic embeddings.
        
        This implements the core DAC mechanism:
        Q = W_q * F_img     (image features as queries)
        K = W_k * F_sem     (semantic embeddings as keys)  
        V = W_v * F_sem     (semantic embeddings as values)
        
        Args:
            image_features (Tensor): Image features [B, H*W, feature_dim]
            semantic_embeddings (Tensor): CLIP embeddings [B, semantic_dim]
            
        Returns:
            Tensor: Semantically-guided features [B, H*W, feature_dim]
        """
        B, N, C = image_features.shape  # N = H*W
        
        # Project to Q, K, V
        Q = self.q_proj(image_features)  # [B, N, feature_dim]
        K = self.k_proj(semantic_embeddings.unsqueeze(1))  # [B, 1, feature_dim]  
        V = self.v_proj(semantic_embeddings.unsqueeze(1))  # [B, 1, feature_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
        K = K.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, 1, head_dim]
        V = V.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, 1, head_dim]
        
        # Scaled dot-product attention with learnable temperature
        # Attention scores: Q @ K^T scaled by learnable temperature
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.temperature  # [B, heads, N, 1]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, heads, N, head_dim]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)
        output = self.out_proj(attn_output)
        
        # Residual connection and normalization
        output = self.norm(output + image_features)
        
        return output


class SemanticGuidanceModule(nn.Module):
    """
    Multi-scale semantic guidance module integrating DAC at different resolutions.
    
    This provides semantic guidance at multiple stages of the U-Net backbone,
    enabling both coarse and fine-grained weather-aware adaptations.
    """
    
    def __init__(self, feature_dims, semantic_dim=512):
        """
        Args:
            feature_dims (list): Feature dimensions at different scales
            semantic_dim (int): Semantic embedding dimension
        """
        super().__init__()
        
        # DAC modules for different scales
        self.dac_modules = nn.ModuleList([
            DegradationAwareCrossAttention(dim, semantic_dim) 
            for dim in feature_dims
        ])
        
        # Semantic embedding adapter for different scales
        self.semantic_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(semantic_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(inplace=True)
            ) for dim in feature_dims
        ])
        
    def forward(self, feature_pyramid, semantic_embedding):
        """
        Apply semantic guidance to multi-scale features.
        
        Args:
            feature_pyramid (list): List of features at different scales
            semantic_embedding (Tensor): Global semantic embedding [B, semantic_dim]
            
        Returns:
            list: Semantically-guided features at each scale
        """
        guided_features = []
        
        for i, (features, dac_module, adapter) in enumerate(
            zip(feature_pyramid, self.dac_modules, self.semantic_adapters)
        ):
            B, C, H, W = features.shape
            
            # Adapt semantic embedding for current scale
            adapted_semantic = adapter(semantic_embedding)  # [B, C]
            
            # Flatten spatial dimensions for attention
            features_flat = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
            
            # Apply degradation-aware cross-attention
            guided_flat = dac_module(features_flat, adapted_semantic)
            
            # Reshape back to spatial format
            guided = guided_flat.transpose(1, 2).view(B, C, H, W)
            guided_features.append(guided)
            
        return guided_features


class WeatherClassifier(nn.Module):
    """
    Optional weather type classifier using semantic embeddings.
    
    This can be used for explicit weather type prediction, though in the main
    model the weather understanding is implicit through the semantic guidance.
    """
    
    def __init__(self, semantic_dim=512, num_weather_types=4):
        """
        Args:
            semantic_dim (int): Dimension of input semantic embeddings
            num_weather_types (int): Number of weather categories (rain, snow, fog, haze)
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(semantic_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_weather_types)
        )
        
        self.weather_types = ['rain', 'snow', 'fog', 'clear']
        
    def forward(self, semantic_embeddings):
        """
        Classify weather type from semantic embeddings.
        
        Args:
            semantic_embeddings (Tensor): Semantic embeddings [B, semantic_dim]
            
        Returns:
            Tensor: Weather type logits [B, num_weather_types]
        """
        return self.classifier(semantic_embeddings)


def test_dac_clip():
    """Test function to verify DAC-CLIP functionality."""
    print("Testing Degradation-Aware Cross-Attention with CLIP...")
    
    # Test parameters
    B, H, W = 2, 32, 32
    feature_dims = [64, 128, 256]
    semantic_dim = 512
    
    # Create test inputs
    degraded_image = torch.randn(B, 3, 224, 224)  # Typical input size
    feature_pyramid = [
        torch.randn(B, dim, H // (2**i), W // (2**i)) 
        for i, dim in enumerate(feature_dims)
    ]
    
    # Test Mock CLIP Encoder
    clip_encoder = MockCLIPEncoder(embed_dim=semantic_dim)
    semantic_embeddings = clip_encoder(degraded_image)
    
    print(f"Mock CLIP output shape: {semantic_embeddings.shape}")
    print(f"Embedding norm: {torch.norm(semantic_embeddings, dim=-1).mean():.4f}")
    
    # Test single DAC module
    dac = DegradationAwareCrossAttention(feature_dims[0], semantic_dim)
    test_features = feature_pyramid[0].flatten(2).transpose(1, 2)  # [B, H*W, C]
    guided_features = dac(test_features, semantic_embeddings)
    
    print(f"DAC input shape: {test_features.shape}")
    print(f"DAC output shape: {guided_features.shape}")
    
    # Test Multi-scale Semantic Guidance
    guidance_module = SemanticGuidanceModule(feature_dims, semantic_dim)
    guided_pyramid = guidance_module(feature_pyramid, semantic_embeddings)
    
    print(f"Multi-scale guidance:")
    for i, features in enumerate(guided_pyramid):
        print(f"  Scale {i}: {features.shape}")
    
    # Test Weather Classifier
    weather_classifier = WeatherClassifier(semantic_dim)
    weather_logits = weather_classifier(semantic_embeddings)
    
    print(f"Weather classification logits: {weather_logits.shape}")
    
    # Parameter count
    total_params = (
        sum(p.numel() for p in clip_encoder.parameters()) +
        sum(p.numel() for p in dac.parameters()) +
        sum(p.numel() for p in guidance_module.parameters())
    )
    print(f"Total parameters: {total_params:,}")
    
    print("✓ DAC-CLIP test passed!")


if __name__ == "__main__":
    test_dac_clip()
