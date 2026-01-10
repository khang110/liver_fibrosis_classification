"""Dual-stream image models for liver fibrosis classification.

This module contains CNN architectures that process both B-mode and Nakagami
ultrasound images for binary classification (F0-1 vs F2-4).
"""

import logging
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from models.bmode_models import load_backbone

logger = logging.getLogger(__name__)


class DualStreamMeanPoolingModel(nn.Module):
    """CNN model with mean pooling over dual-stream images (B-mode + Nakagami).
    
    This model processes 3 B-mode and 3 Nakagami images per patient by:
    1. Processing each image independently through separate CNN backbones
    2. Extracting feature vectors from each image
    3. Mean pooling features across the 3 images for each modality
    4. Concatenating B-mode and Nakagami features
    5. Producing a patient-level prediction from the concatenated features
    
    Architecture:
    - Uses two separate CNN backbones (same architecture, separate weights)
      for B-mode and Nakagami images
    - Removes final FC layers to extract feature vectors
    - Mean pools over temporal dimension (3 images) for each modality
    - Concatenates features and applies final classifier
    
    Input shapes:
    - x_bmode: (B, T=3, C, H, W) where T=3 is number of B-mode images
    - x_naka: (B, T=3, C, H, W) where T=3 is number of Nakagami images
    
    Output shape: (B, 2) - patient-level logits
    """
    
    def __init__(
        self,
        backbone_bmode: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
        backbone_naka: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
        pretrained: bool = True,
    ):
        """Initialize the dual-stream mean pooling model.
        
        Args:
            backbone_bmode: Backbone architecture for B-mode images ('resnet18', 'resnet34', 
                'efficientnetv2_b0', 'efficientnetv2_b2'). Default: 'resnet18'.
            backbone_naka: Backbone architecture for Nakagami images ('resnet18', 'resnet34', 
                'efficientnetv2_b0', 'efficientnetv2_b2'). Default: 'resnet18'.
            pretrained: Whether to use ImageNet pretrained weights. Default: True.
        """
        super().__init__()
        
        # Load B-mode backbone
        self.backbone_bmode, feat_dim_b = load_backbone(backbone_bmode, pretrained)
        
        # Load Nakagami backbone
        self.backbone_naka, feat_dim_n = load_backbone(backbone_naka, pretrained)
        
        # Enforce same feature dimension for both backbones
        if feat_dim_b != feat_dim_n:
            raise ValueError(
                f"Feature dimensions must match: B-mode backbone has {feat_dim_b}, "
                f"Nakagami backbone has {feat_dim_n}. Use the same backbone architecture "
                f"for both modalities or ensure they have the same feature dimension."
            )
        
        feature_dim = feat_dim_b
        
        # Remove final FC/classifier layers to extract features (not logits)
        # B-mode backbone
        if hasattr(self.backbone_bmode, 'fc'):  # ResNet
            self.backbone_bmode.fc = nn.Identity()
        elif hasattr(self.backbone_bmode, 'classifier'):  # EfficientNet
            # Keep only the dropout, remove the linear layer
            self.backbone_bmode.classifier = nn.Sequential(
                self.backbone_bmode.classifier[0],  # Dropout
                nn.Identity()
            )
        else:
            raise ValueError(f"Unknown backbone structure for {backbone_bmode}")
        
        # Nakagami backbone
        if hasattr(self.backbone_naka, 'fc'):  # ResNet
            self.backbone_naka.fc = nn.Identity()
        elif hasattr(self.backbone_naka, 'classifier'):  # EfficientNet
            # Keep only the dropout, remove the linear layer
            self.backbone_naka.classifier = nn.Sequential(
                self.backbone_naka.classifier[0],  # Dropout
                nn.Identity()
            )
        else:
            raise ValueError(f"Unknown backbone structure for {backbone_naka}")
        
        self.feature_dim = feature_dim
        
        # Final classifier: concatenated features -> logits
        # Input: 2 * feature_dim (B-mode + Nakagami)
        # Output: 2 (binary classification)
        self.classifier = nn.Linear(2 * feature_dim, 2)
        
        # Store configuration
        self.backbone_bmode_name = backbone_bmode
        self.backbone_naka_name = backbone_naka
        self.pretrained = pretrained
    
    def forward(
        self,
        x_bmode: torch.Tensor,
        x_naka: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x_bmode: Input tensor of shape (B, T=3, C, H, W) where:
                B = batch size
                T = 3 (number of B-mode images per patient)
                C = 3 (RGB channels)
                H, W = 224 (image dimensions)
            x_naka: Input tensor of shape (B, T=3, C, H, W) where:
                B = batch size
                T = 3 (number of Nakagami images per patient)
                C = 3 (RGB channels)
                H, W = 224 (image dimensions)
        
        Returns:
            Tensor of shape (B, 2) containing patient-level logits.
        """
        # Validate input shapes
        if len(x_bmode.shape) != 5:
            raise ValueError(f"Expected x_bmode shape (B, T, C, H, W), got {x_bmode.shape}")
        if len(x_naka.shape) != 5:
            raise ValueError(f"Expected x_naka shape (B, T, C, H, W), got {x_naka.shape}")
        
        B, T, C, H, W = x_bmode.shape
        B_n, T_n, C_n, H_n, W_n = x_naka.shape
        
        if B != B_n:
            raise ValueError(
                f"Batch size mismatch: x_bmode has {B} samples, "
                f"x_naka has {B_n} samples"
            )
        
        # Process B-mode images
        # Flatten temporal dimension: (B*T, C, H, W)
        x_bmode_flat = x_bmode.view(B * T, C, H, W)
        
        # Extract features from B-mode backbone: (B*T, feature_dim)
        features_bmode = self.backbone_bmode(x_bmode_flat)  # (B*T, feature_dim)
        
        # Reshape to group by patient: (B, T, feature_dim)
        features_bmode = features_bmode.view(B, T, self.feature_dim)
        
        # Mean pool over temporal dimension: (B, feature_dim)
        f_bmode = features_bmode.mean(dim=1)  # (B, feature_dim)
        
        # Process Nakagami images
        # Flatten temporal dimension: (B*T, C, H, W)
        x_naka_flat = x_naka.view(B * T_n, C_n, H_n, W_n)
        
        # Extract features from Nakagami backbone: (B*T, feature_dim)
        features_naka = self.backbone_naka(x_naka_flat)  # (B*T, feature_dim)
        
        # Reshape to group by patient: (B, T, feature_dim)
        features_naka = features_naka.view(B, T_n, self.feature_dim)
        
        # Mean pool over temporal dimension: (B, feature_dim)
        f_naka = features_naka.mean(dim=1)  # (B, feature_dim)
        
        # Concatenate features: z = [f_bmode; f_naka]
        z = torch.cat([f_bmode, f_naka], dim=1)  # (B, 2 * feature_dim)
        
        # Final classifier: (B, 2 * feature_dim) -> (B, 2)
        logits = self.classifier(z)  # (B, 2)
        
        return logits


class DualStreamAttentionPoolingModel(nn.Module):
    """CNN model with attention pooling over dual-stream images (B-mode + Nakagami).
    
    This model processes 3 B-mode and 3 Nakagami images per patient by:
    1. Extracting feature vectors from each image using separate CNN backbones
    2. Computing attention weights for each image in each modality
    3. Aggregating features using attention-weighted sum for each modality
    4. Concatenating B-mode and Nakagami features
    5. Producing a patient-level prediction from the concatenated features
    
    Architecture:
    - Uses two separate CNN backbones (same architecture, separate weights)
      for B-mode and Nakagami images
    - Removes final FC layers to extract feature vectors
    - Attention pooling per modality: a_i = v^T * tanh(W f_i + b)
    - Concatenates features and applies final classifier
    
    Input shapes:
    - x_bmode: (B, T=3, C, H, W) where T=3 is number of B-mode images
    - x_naka: (B, T=3, C, H, W) where T=3 is number of Nakagami images
    
    Output shape: (B, 2) - patient-level logits
    """
    
    def __init__(
        self,
        backbone_bmode: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
        backbone_naka: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
        pretrained: bool = True,
        feature_dim: int = 512,
        attention_hidden: int = 128
    ):
        """Initialize the dual-stream attention pooling model.
        
        Args:
            backbone_bmode: Backbone architecture for B-mode images ('resnet18', 'resnet34', 
                'efficientnetv2_b0', 'efficientnetv2_b2'). Default: 'resnet18'.
            backbone_naka: Backbone architecture for Nakagami images ('resnet18', 'resnet34', 
                'efficientnetv2_b0', 'efficientnetv2_b2'). Default: 'resnet18'.
            pretrained: Whether to use ImageNet pretrained weights. Default: True.
            feature_dim: Dimension of feature vectors extracted from each image.
                Should match the backbone's feature dimension. Default: 512.
            attention_hidden: Hidden dimension for attention network. Default: 128.
        """
        super().__init__()
        
        # Load B-mode backbone
        self.backbone_bmode, backbone_feature_dim_b = load_backbone(backbone_bmode, pretrained)
        
        # Load Nakagami backbone
        self.backbone_naka, backbone_feature_dim_n = load_backbone(backbone_naka, pretrained)
        
        # Verify feature_dim matches both backbones
        if feature_dim != backbone_feature_dim_b:
            logger.warning(
                f"feature_dim ({feature_dim}) does not match B-mode backbone feature dim "
                f"({backbone_feature_dim_b}). Using backbone feature dim."
            )
            feature_dim = backbone_feature_dim_b
        
        if feature_dim != backbone_feature_dim_n:
            logger.warning(
                f"feature_dim ({feature_dim}) does not match Nakagami backbone feature dim "
                f"({backbone_feature_dim_n}). Using backbone feature dim."
            )
            # Use the B-mode feature dim if they differ
            if backbone_feature_dim_b != backbone_feature_dim_n:
                raise ValueError(
                    f"B-mode and Nakagami backbones must have the same feature dimension. "
                    f"B-mode: {backbone_feature_dim_b}, Nakagami: {backbone_feature_dim_n}"
                )
            feature_dim = backbone_feature_dim_b
        
        # Remove final FC/classifier layers to extract features (not logits)
        # B-mode backbone
        if hasattr(self.backbone_bmode, 'fc'):  # ResNet
            self.backbone_bmode.fc = nn.Identity()
        elif hasattr(self.backbone_bmode, 'classifier'):  # EfficientNet
            # Keep only the dropout, remove the linear layer
            self.backbone_bmode.classifier = nn.Sequential(
                self.backbone_bmode.classifier[0],  # Dropout
                nn.Identity()
            )
        else:
            raise ValueError(f"Unknown backbone structure for {backbone_bmode}")
        
        # Nakagami backbone
        if hasattr(self.backbone_naka, 'fc'):  # ResNet
            self.backbone_naka.fc = nn.Identity()
        elif hasattr(self.backbone_naka, 'classifier'):  # EfficientNet
            # Keep only the dropout, remove the linear layer
            self.backbone_naka.classifier = nn.Sequential(
                self.backbone_naka.classifier[0],  # Dropout
                nn.Identity()
            )
        else:
            raise ValueError(f"Unknown backbone structure for {backbone_naka}")
        
        self.feature_dim = feature_dim
        
        # Attention networks for each modality
        # a_i = v^T * tanh(W f_i + b)
        # B-mode attention
        self.attention_W_bmode = nn.Linear(feature_dim, attention_hidden)
        self.attention_b_bmode = nn.Parameter(torch.zeros(attention_hidden))
        self.attention_v_bmode = nn.Parameter(torch.randn(attention_hidden))
        
        # Nakagami attention
        self.attention_W_naka = nn.Linear(feature_dim, attention_hidden)
        self.attention_b_naka = nn.Parameter(torch.zeros(attention_hidden))
        self.attention_v_naka = nn.Parameter(torch.randn(attention_hidden))
        
        # Final classifier: concatenated features -> logits
        # Input: 2 * feature_dim (B-mode + Nakagami)
        # Output: 2 (binary classification)
        self.classifier = nn.Linear(2 * feature_dim, 2)
        
        # Store configuration
        self.backbone_bmode_name = backbone_bmode
        self.backbone_naka_name = backbone_naka
        self.pretrained = pretrained
        self.attention_hidden = attention_hidden
    
    def _apply_attention_pooling(
        self,
        features: torch.Tensor,
        attention_W: nn.Linear,
        attention_b: torch.nn.Parameter,
        attention_v: torch.nn.Parameter
    ) -> torch.Tensor:
        """Apply attention pooling to features.
        
        Args:
            features: Tensor of shape (B, T, feature_dim).
            attention_W: Linear layer for attention (feature_dim -> attention_hidden).
            attention_b: Bias parameter (attention_hidden,).
            attention_v: Attention vector (attention_hidden,).
        
        Returns:
            Pooled features of shape (B, feature_dim).
        """
        B, T, feature_dim = features.shape
        
        # Compute attention scores: a_i = v^T * tanh(W f_i + b)
        attention_hidden_features = attention_W(features) + attention_b  # (B, T, attention_hidden)
        attention_hidden_features = torch.tanh(attention_hidden_features)  # (B, T, attention_hidden)
        
        # Compute attention scores: v^T * attention_hidden_features
        attention_scores = torch.matmul(
            attention_hidden_features,
            attention_v.unsqueeze(-1)  # (attention_hidden, 1)
        ).squeeze(-1)  # (B, T)
        
        # Apply softmax to get attention weights: (B, T)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (B, T)
        
        # Compute weighted sum: f_patient = sum_i alpha_i * f_i
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # (B, T, 1)
        patient_features = (attention_weights_expanded * features).sum(dim=1)  # (B, feature_dim)
        
        return patient_features
    
    def forward(
        self,
        x_bmode: torch.Tensor,
        x_naka: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x_bmode: Input tensor of shape (B, T=3, C, H, W) where:
                B = batch size
                T = 3 (number of B-mode images per patient)
                C = 3 (RGB channels)
                H, W = 224 (image dimensions)
            x_naka: Input tensor of shape (B, T=3, C, H, W) where:
                B = batch size
                T = 3 (number of Nakagami images per patient)
                C = 3 (RGB channels)
                H, W = 224 (image dimensions)
        
        Returns:
            Tensor of shape (B, 2) containing patient-level logits.
        """
        # Validate input shapes
        if len(x_bmode.shape) != 5:
            raise ValueError(f"Expected x_bmode shape (B, T, C, H, W), got {x_bmode.shape}")
        if len(x_naka.shape) != 5:
            raise ValueError(f"Expected x_naka shape (B, T, C, H, W), got {x_naka.shape}")
        
        B, T, C, H, W = x_bmode.shape
        B_n, T_n, C_n, H_n, W_n = x_naka.shape
        
        if B != B_n:
            raise ValueError(
                f"Batch size mismatch: x_bmode has {B} samples, "
                f"x_naka has {B_n} samples"
            )
        
        # Process B-mode images
        # Flatten temporal dimension: (B*T, C, H, W)
        x_bmode_flat = x_bmode.view(B * T, C, H, W)
        
        # Extract features from B-mode backbone: (B*T, feature_dim)
        features_bmode = self.backbone_bmode(x_bmode_flat)  # (B*T, feature_dim)
        
        # Reshape to group by patient: (B, T, feature_dim)
        features_bmode = features_bmode.view(B, T, self.feature_dim)
        
        # Apply attention pooling: (B, feature_dim)
        f_bmode_att = self._apply_attention_pooling(
            features_bmode,
            self.attention_W_bmode,
            self.attention_b_bmode,
            self.attention_v_bmode
        )
        
        # Process Nakagami images
        # Flatten temporal dimension: (B*T, C, H, W)
        x_naka_flat = x_naka.view(B * T_n, C_n, H_n, W_n)
        
        # Extract features from Nakagami backbone: (B*T, feature_dim)
        features_naka = self.backbone_naka(x_naka_flat)  # (B*T, feature_dim)
        
        # Reshape to group by patient: (B, T, feature_dim)
        features_naka = features_naka.view(B, T_n, self.feature_dim)
        
        # Apply attention pooling: (B, feature_dim)
        f_naka_att = self._apply_attention_pooling(
            features_naka,
            self.attention_W_naka,
            self.attention_b_naka,
            self.attention_v_naka
        )
        
        # Concatenate features: z = [f_bmode_att; f_naka_att]
        z = torch.cat([f_bmode_att, f_naka_att], dim=1)  # (B, 2 * feature_dim)
        
        # Final classifier: (B, 2 * feature_dim) -> (B, 2)
        logits = self.classifier(z)  # (B, 2)
        
        return logits


def create_dual_stream_mean_model(
    backbone_bmode: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
    backbone_naka: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
    pretrained: bool = True
) -> nn.Module:
    """Create a dual-stream mean pooling model.
    
    Convenience function to instantiate a DualStreamMeanPoolingModel with default
    or specified parameters.
    
    Args:
        backbone_bmode: Backbone architecture for B-mode images ('resnet18', 'resnet34', 
            'efficientnetv2_b0', 'efficientnetv2_b2'). Default: 'resnet18'.
        backbone_naka: Backbone architecture for Nakagami images ('resnet18', 'resnet34', 
            'efficientnetv2_b0', 'efficientnetv2_b2'). Default: 'resnet18'.
        pretrained: Whether to use ImageNet pretrained weights. Default: True.
    
    Returns:
        DualStreamMeanPoolingModel instance ready for training or inference.
    
    Example:
        >>> model = create_dual_stream_mean_model(
        ...     backbone_bmode="resnet18",
        ...     backbone_naka="resnet18",
        ...     pretrained=True
        ... )
        >>> # Input: bmode_imgs (batch_size, 3, 3, 224, 224), nakagami_imgs (batch_size, 3, 3, 224, 224)
        >>> # Output: (batch_size, 2)
        >>> output = model(bmode_imgs, nakagami_imgs)
    """
    model = DualStreamMeanPoolingModel(
        backbone_bmode=backbone_bmode,
        backbone_naka=backbone_naka,
        pretrained=pretrained
    )
    return model


def create_dual_stream_attention_model(
    backbone_bmode: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
    backbone_naka: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
    pretrained: bool = True,
    feature_dim: int = 512,
    attention_hidden: int = 128
) -> nn.Module:
    """Create a dual-stream attention pooling model.
    
    Convenience function to instantiate a DualStreamAttentionPoolingModel with default
    or specified parameters.
    
    Args:
        backbone_bmode: Backbone architecture for B-mode images ('resnet18', 'resnet34', 
            'efficientnetv2_b0', 'efficientnetv2_b2'). Default: 'resnet18'.
        backbone_naka: Backbone architecture for Nakagami images ('resnet18', 'resnet34', 
            'efficientnetv2_b0', 'efficientnetv2_b2'). Default: 'resnet18'.
        pretrained: Whether to use ImageNet pretrained weights. Default: True.
        feature_dim: Dimension of feature vectors extracted from each image.
            Should match the backbone's feature dimension (512 for ResNet18/34).
            Default: 512.
        attention_hidden: Hidden dimension for attention network. Default: 128.
    
    Returns:
        DualStreamAttentionPoolingModel instance ready for training or inference.
    
    Example:
        >>> model = create_dual_stream_attention_model(
        ...     backbone_bmode="resnet18",
        ...     backbone_naka="resnet18",
        ...     pretrained=True,
        ...     feature_dim=512,
        ...     attention_hidden=128
        ... )
        >>> # Input: bmode_imgs (batch_size, 3, 3, 224, 224), nakagami_imgs (batch_size, 3, 3, 224, 224)
        >>> # Output: (batch_size, 2)
        >>> output = model(bmode_imgs, nakagami_imgs)
    """
    model = DualStreamAttentionPoolingModel(
        backbone_bmode=backbone_bmode,
        backbone_naka=backbone_naka,
        pretrained=pretrained,
        feature_dim=feature_dim,
        attention_hidden=attention_hidden
    )
    return model


class DualStreamClinicalFusionModel(nn.Module):
    """Dual-stream CNN + Clinical fusion model for liver fibrosis classification.
    
    This model combines B-mode and Nakagami image features (from dual CNNs) with 
    clinical features to produce patient-level predictions. It supports both mean 
    pooling (E1) and attention pooling (E2) for image feature aggregation.
    
    Architecture:
    - Two separate CNN backbones extract features from B-mode and Nakagami images
    - Image pooling: Mean pooling (E1) or Attention pooling (E2) per modality
    - Clinical branch: MLP processes clinical features
    - Fusion: Concatenates all three branches and applies MLP for final prediction
    
    Input shapes:
    - x_bmode: (B, T=3, C, H, W) where T=3 is number of B-mode images
    - x_naka: (B, T=3, C, H, W) where T=3 is number of Nakagami images
    - clinical_features: (B, K) where K is number of clinical features
    
    Output shape: (B, 2) - patient-level logits
    """
    
    def __init__(
        self,
        backbone_bmode: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
        backbone_naka: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
        pretrained: bool = True,
        feature_dim: int = 512,
        clinical_dim: int = 32,
        fusion_hidden: int = 128,
        pooling: Literal["mean", "attention"] = "mean",
        attention_hidden: int = 128,
        dropout: float = 0.5,
        clinical_input_dim: Optional[int] = None
    ):
        """Initialize the dual-stream + clinical fusion model.
        
        Args:
            backbone_bmode: Backbone architecture for B-mode images ('resnet18', 'resnet34', 
                'efficientnetv2_b0', 'efficientnetv2_b2'). Default: 'resnet18'.
            backbone_naka: Backbone architecture for Nakagami images ('resnet18', 'resnet34', 
                'efficientnetv2_b0', 'efficientnetv2_b2'). Default: 'resnet18'.
            pretrained: Whether to use ImageNet pretrained weights. Default: True.
            feature_dim: Dimension of CNN feature vectors (should match backbone).
                Default: 512.
            clinical_dim: Output dimension for processed clinical features.
                Default: 32.
            fusion_hidden: Hidden dimension for fusion MLP. Default: 128.
            pooling: Pooling method for images: 'mean' (E1) or 'attention' (E2).
                Default: 'mean'.
            attention_hidden: Hidden dimension for attention network (only used
                if pooling='attention'). Default: 128.
            dropout: Dropout probability in fusion MLP. Default: 0.5.
            clinical_input_dim: Number of input clinical features. If None,
                will be inferred from first forward pass. Default: None.
        """
        super().__init__()
        
        # Load B-mode backbone
        self.backbone_bmode, backbone_feature_dim_b = load_backbone(backbone_bmode, pretrained)
        
        # Load Nakagami backbone
        self.backbone_naka, backbone_feature_dim_n = load_backbone(backbone_naka, pretrained)
        
        # Verify feature_dim matches both backbones
        if feature_dim != backbone_feature_dim_b:
            logger.warning(
                f"feature_dim ({feature_dim}) does not match B-mode backbone feature dim "
                f"({backbone_feature_dim_b}). Using backbone feature dim."
            )
            feature_dim = backbone_feature_dim_b
        
        if feature_dim != backbone_feature_dim_n:
            logger.warning(
                f"feature_dim ({feature_dim}) does not match Nakagami backbone feature dim "
                f"({backbone_feature_dim_n}). Using backbone feature dim."
            )
            if backbone_feature_dim_b != backbone_feature_dim_n:
                raise ValueError(
                    f"B-mode and Nakagami backbones must have the same feature dimension. "
                    f"B-mode: {backbone_feature_dim_b}, Nakagami: {backbone_feature_dim_n}"
                )
            feature_dim = backbone_feature_dim_b
        
        # Remove final FC/classifier layers to extract features (not logits)
        # B-mode backbone
        if hasattr(self.backbone_bmode, 'fc'):  # ResNet
            self.backbone_bmode.fc = nn.Identity()
        elif hasattr(self.backbone_bmode, 'classifier'):  # EfficientNet
            self.backbone_bmode.classifier = nn.Sequential(
                self.backbone_bmode.classifier[0],  # Dropout
                nn.Identity()
            )
        else:
            raise ValueError(f"Unknown backbone structure for {backbone_bmode}")
        
        # Nakagami backbone
        if hasattr(self.backbone_naka, 'fc'):  # ResNet
            self.backbone_naka.fc = nn.Identity()
        elif hasattr(self.backbone_naka, 'classifier'):  # EfficientNet
            self.backbone_naka.classifier = nn.Sequential(
                self.backbone_naka.classifier[0],  # Dropout
                nn.Identity()
            )
        else:
            raise ValueError(f"Unknown backbone structure for {backbone_naka}")
        
        self.feature_dim = feature_dim
        self.pooling = pooling
        
        # Image pooling branch (applied to both modalities)
        if pooling == "mean":
            # Mean pooling: no additional parameters needed
            self.image_pooling = None
        elif pooling == "attention":
            # Attention pooling: separate attention for each modality
            # B-mode attention
            self.attention_W_bmode = nn.Linear(feature_dim, attention_hidden)
            self.attention_b_bmode = nn.Parameter(torch.zeros(attention_hidden))
            self.attention_v_bmode = nn.Parameter(torch.randn(attention_hidden))
            # Nakagami attention
            self.attention_W_naka = nn.Linear(feature_dim, attention_hidden)
            self.attention_b_naka = nn.Parameter(torch.zeros(attention_hidden))
            self.attention_v_naka = nn.Parameter(torch.randn(attention_hidden))
            self.attention_hidden = attention_hidden
        else:
            raise ValueError(
                f"Unsupported pooling method: {pooling}. "
                f"Supported: 'mean', 'attention'"
            )
        
        # Clinical branch: MLP to process clinical features
        # Input dimension will be set during first forward pass or can be specified
        self.clinical_projection = None  # Will be initialized in forward if needed
        self.clinical_dim = clinical_dim
        self._clinical_input_dim = None  # Will be set from first input
        
        # Fusion MLP
        # Input: 2 * feature_dim (B-mode + Nakagami) + clinical_dim (processed clinical)
        fusion_input_dim = 2 * feature_dim + clinical_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 2)
        )
        
        # Store configuration
        self.backbone_bmode_name = backbone_bmode
        self.backbone_naka_name = backbone_naka
        self.pretrained = pretrained
        self.fusion_hidden = fusion_hidden
        self.dropout = dropout
    
    def _initialize_clinical_projection(self, clinical_input_dim: int):
        """Initialize clinical projection layer.
        
        Args:
            clinical_input_dim: Number of input clinical features.
        """
        if self.clinical_projection is None or self._clinical_input_dim != clinical_input_dim:
            self.clinical_projection = nn.Sequential(
                nn.Linear(clinical_input_dim, self.clinical_dim),
                nn.ReLU()
            )
            self._clinical_input_dim = clinical_input_dim
            # Move to same device as other parameters
            if next(self.fusion_mlp.parameters()).is_cuda:
                self.clinical_projection = self.clinical_projection.cuda()
    
    def _pool_image_features(
        self,
        image_features: torch.Tensor,
        attention_W: Optional[nn.Linear] = None,
        attention_b: Optional[torch.nn.Parameter] = None,
        attention_v: Optional[torch.nn.Parameter] = None
    ) -> torch.Tensor:
        """Pool image features across temporal dimension.
        
        Args:
            image_features: Tensor of shape (B, T, feature_dim).
            attention_W: Linear layer for attention (only if pooling='attention').
            attention_b: Bias parameter for attention (only if pooling='attention').
            attention_v: Attention vector (only if pooling='attention').
        
        Returns:
            Pooled features of shape (B, feature_dim).
        """
        B, T, feature_dim = image_features.shape
        
        if self.pooling == "mean":
            # Mean pooling: (B, T, feature_dim) -> (B, feature_dim)
            f_img = image_features.mean(dim=1)
        elif self.pooling == "attention":
            # Attention pooling: same as A2
            # Compute attention scores: a_i = v^T * tanh(W f_i + b)
            attention_hidden_features = attention_W(image_features) + attention_b
            attention_hidden_features = torch.tanh(attention_hidden_features)
            
            # Compute attention scores: v^T * attention_hidden_features
            attention_scores = torch.matmul(
                attention_hidden_features,
                attention_v.unsqueeze(-1)
            ).squeeze(-1)  # (B, T)
            
            # Apply softmax to get attention weights
            attention_weights = torch.softmax(attention_scores, dim=1)  # (B, T)
            
            # Compute weighted sum
            attention_weights_expanded = attention_weights.unsqueeze(-1)  # (B, T, 1)
            f_img = (attention_weights_expanded * image_features).sum(dim=1)  # (B, feature_dim)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        return f_img
    
    def forward(
        self,
        x_bmode: torch.Tensor,
        x_naka: torch.Tensor,
        clinical_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x_bmode: Input tensor of shape (B, T=3, C, H, W) where:
                B = batch size
                T = 3 (number of B-mode images per patient)
                C = 3 (RGB channels)
                H, W = 224 (image dimensions)
            x_naka: Input tensor of shape (B, T=3, C, H, W) where:
                B = batch size
                T = 3 (number of Nakagami images per patient)
                C = 3 (RGB channels)
                H, W = 224 (image dimensions)
            clinical_features: Clinical feature tensor of shape (B, K) where:
                B = batch size
                K = number of clinical features per patient
        
        Returns:
            Tensor of shape (B, 2) containing patient-level logits.
        """
        # Validate input shapes
        if len(x_bmode.shape) != 5:
            raise ValueError(f"Expected x_bmode shape (B, T, C, H, W), got {x_bmode.shape}")
        if len(x_naka.shape) != 5:
            raise ValueError(f"Expected x_naka shape (B, T, C, H, W), got {x_naka.shape}")
        if len(clinical_features.shape) != 2:
            raise ValueError(
                f"Expected clinical_features shape (B, K), got {clinical_features.shape}"
            )
        
        B, T, C, H, W = x_bmode.shape
        B_n, T_n, C_n, H_n, W_n = x_naka.shape
        B_clin, K = clinical_features.shape
        
        if B != B_n or B != B_clin:
            raise ValueError(
                f"Batch size mismatch: x_bmode has {B} samples, "
                f"x_naka has {B_n} samples, clinical_features has {B_clin} samples"
            )
        
        # Initialize clinical projection if needed
        self._initialize_clinical_projection(K)
        
        # Process B-mode images
        # Flatten temporal dimension: (B*T, C, H, W)
        x_bmode_flat = x_bmode.view(B * T, C, H, W)
        
        # Extract features from B-mode backbone: (B*T, feature_dim)
        features_bmode = self.backbone_bmode(x_bmode_flat)  # (B*T, feature_dim)
        
        # Reshape to group by patient: (B, T, feature_dim)
        features_bmode = features_bmode.view(B, T, self.feature_dim)
        
        # Pool B-mode features: (B, feature_dim)
        if self.pooling == "mean":
            f_bmode = self._pool_image_features(features_bmode)
        else:  # attention
            f_bmode = self._pool_image_features(
                features_bmode,
                self.attention_W_bmode,
                self.attention_b_bmode,
                self.attention_v_bmode
            )
        
        # Process Nakagami images
        # Flatten temporal dimension: (B*T, C, H, W)
        x_naka_flat = x_naka.view(B * T_n, C_n, H_n, W_n)
        
        # Extract features from Nakagami backbone: (B*T, feature_dim)
        features_naka = self.backbone_naka(x_naka_flat)  # (B*T, feature_dim)
        
        # Reshape to group by patient: (B, T, feature_dim)
        features_naka = features_naka.view(B, T_n, self.feature_dim)
        
        # Pool Nakagami features: (B, feature_dim)
        if self.pooling == "mean":
            f_naka = self._pool_image_features(features_naka)
        else:  # attention
            f_naka = self._pool_image_features(
                features_naka,
                self.attention_W_naka,
                self.attention_b_naka,
                self.attention_v_naka
            )
        
        # Clinical branch: Process clinical features
        # MLP: (B, K) -> (B, clinical_dim)
        g_clin = self.clinical_projection(clinical_features)  # (B, clinical_dim)
        
        # Fusion: Concatenate all features
        # z = [f_bmode; f_naka; g_clin]: (B, 2 * feature_dim + clinical_dim)
        z = torch.cat([f_bmode, f_naka, g_clin], dim=1)  # (B, 2 * feature_dim + clinical_dim)
        
        # Fusion MLP: (B, 2 * feature_dim + clinical_dim) -> (B, 2)
        logits = self.fusion_mlp(z)  # (B, 2)
        return logits


def create_dual_stream_clinical_fusion_model(
    backbone_bmode: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
    backbone_naka: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
    pretrained: bool = True,
    feature_dim: int = 512,
    clinical_dim: int = 32,
    fusion_hidden: int = 128,
    pooling: Literal["mean", "attention"] = "mean",
    attention_hidden: int = 128,
    dropout: float = 0.5,
    clinical_input_dim: Optional[int] = None
) -> nn.Module:
    """Create a dual-stream + clinical fusion model.
    
    Convenience function to instantiate a DualStreamClinicalFusionModel with default
    or specified parameters.
    
    Args:
        backbone_bmode: Backbone architecture for B-mode images ('resnet18', 'resnet34', 
            'efficientnetv2_b0', 'efficientnetv2_b2'). Default: 'resnet18'.
        backbone_naka: Backbone architecture for Nakagami images ('resnet18', 'resnet34', 
            'efficientnetv2_b0', 'efficientnetv2_b2'). Default: 'resnet18'.
        pretrained: Whether to use ImageNet pretrained weights. Default: True.
        feature_dim: Dimension of CNN feature vectors. Default: 512.
        clinical_dim: Output dimension for processed clinical features.
            Default: 32.
        fusion_hidden: Hidden dimension for fusion MLP. Default: 128.
        pooling: Pooling method: 'mean' (E1) or 'attention' (E2). Default: 'mean'.
        attention_hidden: Hidden dimension for attention (only if pooling='attention').
            Default: 128.
        dropout: Dropout probability in fusion MLP. Default: 0.5.
        clinical_input_dim: Number of input clinical features. If None,
            will be inferred from first forward pass. Default: None.
    
    Returns:
        DualStreamClinicalFusionModel instance ready for training or inference.
    
    Example:
        >>> # E1: Mean pooling
        >>> model_e1 = create_dual_stream_clinical_fusion_model(
        ...     pooling="mean",
        ...     clinical_input_dim=8
        ... )
        >>> # E2: Attention pooling
        >>> model_e2 = create_dual_stream_clinical_fusion_model(
        ...     pooling="attention",
        ...     clinical_input_dim=8
        ... )
        >>> # Input: bmode_imgs (batch_size, 3, 3, 224, 224), 
        >>> #       nakagami_imgs (batch_size, 3, 3, 224, 224),
        >>> #       clinical (batch_size, 8)
        >>> # Output: (batch_size, 2)
        >>> output = model_e1(bmode_imgs, nakagami_imgs, clinical_features)
    """
    model = DualStreamClinicalFusionModel(
        backbone_bmode=backbone_bmode,
        backbone_naka=backbone_naka,
        pretrained=pretrained,
        feature_dim=feature_dim,
        clinical_dim=clinical_dim,
        fusion_hidden=fusion_hidden,
        pooling=pooling,
        attention_hidden=attention_hidden,
        dropout=dropout,
        clinical_input_dim=clinical_input_dim
    )
    
    # Initialize clinical projection if input dimension is provided
    if clinical_input_dim is not None:
        model._initialize_clinical_projection(clinical_input_dim)
    
    return model

