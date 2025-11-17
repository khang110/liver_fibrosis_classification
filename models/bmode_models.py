"""B-mode image models for liver fibrosis classification.

This module contains CNN architectures that process B-mode ultrasound images
for binary classification (F0-1 vs F2-4).
"""

import logging
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


def load_backbone(
    backbone: str,
    pretrained: bool = True
) -> Tuple[nn.Module, int]:
    """Load a backbone model and return it with its feature dimension.
    
    Args:
        backbone: Backbone name. Supported: 'resnet18', 'resnet34', 
            'efficientnetv2_b0', 'efficientnetv2_b2'.
        pretrained: Whether to use ImageNet pretrained weights. Default: True.
    
    Returns:
        Tuple of (backbone_model, feature_dimension).
    
    Raises:
        ValueError: If backbone name is not supported.
    """
    weights = "DEFAULT" if pretrained else None
    
    if backbone == "resnet18":
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features
    elif backbone == "resnet34":
        model = models.resnet34(weights=weights)
        feature_dim = model.fc.in_features
    elif backbone == "efficientnetv2_b0":
        # Use efficientnet_b0 (EfficientNet v1 B0)
        model = models.efficientnet_b0(weights=weights)
        feature_dim = model.classifier[1].in_features
    elif backbone == "efficientnetv2_b2":
        # Use efficientnet_b2 (EfficientNet v1 B2)
        model = models.efficientnet_b2(weights=weights)
        feature_dim = model.classifier[1].in_features
    else:
        raise ValueError(
            f"Unsupported backbone: {backbone}. "
            f"Supported: 'resnet18', 'resnet34', 'efficientnetv2_b0', 'efficientnetv2_b2'"
        )
    
    return model, feature_dim


class BModeMeanPoolingModel(nn.Module):
    """CNN model with mean pooling over multiple B-mode images per patient.
    
    This model processes 3 B-mode images per patient by:
    1. Processing each image independently through a CNN backbone
    2. Obtaining a logit for each image
    3. Averaging the logits across the 3 images to get a patient-level prediction
    
    Architecture:
    - Uses a ResNet backbone (ResNet18 or ResNet34) with ImageNet pretrained weights
    - Replaces the final fully connected layer to output a single logit per image
    - Mean pools over the temporal dimension (3 images) to get patient-level logit
    
    Input shape: (B, T=3, C, H, W) where:
        B = batch size
        T = 3 (number of B-mode images per patient)
        C = 3 (RGB channels)
        H, W = 224 (image height and width after preprocessing)
    
    Output shape: (B,) - patient-level logits
    """
    
    def __init__(
        self,
        backbone: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
        pretrained: bool = True,
        num_classes: int = 1
    ):
        """Initialize the B-mode mean pooling model.
        
        Args:
            backbone: Backbone architecture to use ('resnet18', 'resnet34', 
                'efficientnetv2_b0', 'efficientnetv2_b2').
            pretrained: Whether to use ImageNet pretrained weights. Default: True.
            num_classes: Number of output classes. Default: 1 (binary classification).
        """
        super().__init__()
        
        # Load backbone
        self.backbone, num_features = load_backbone(backbone, pretrained)
        
        # Replace the final fully connected layer to output a single logit per image
        # Handle different backbone structures
        if hasattr(self.backbone, 'fc'):  # ResNet
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif hasattr(self.backbone, 'classifier'):  # EfficientNet
            # EfficientNet classifier is Sequential(Dropout, Linear)
            self.backbone.classifier = nn.Sequential(
                self.backbone.classifier[0],  # Keep dropout
                nn.Linear(num_features, num_classes)
            )
        else:
            raise ValueError(f"Unknown backbone structure for {backbone}")
        
        # Store backbone name for reference
        self.backbone_name = backbone
        self.pretrained = pretrained
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, T=3, C, H, W) where:
                B = batch size
                T = 3 (number of B-mode images per patient)
                C = 3 (RGB channels)
                H, W = 224 (image dimensions)
        
        Returns:
            Tensor of shape (B,) containing patient-level logits.
            Each logit is the mean of the 3 image-level logits for that patient.
        """
        # x shape: (B, T=3, C, H, W)
        B, T, C, H, W = x.shape
        
        # Reshape to process all images: (B*T, C, H, W)
        x = x.view(B * T, C, H, W)
        
        # Pass through backbone: (B*T, 1) - one logit per image
        image_logits = self.backbone(x)  # Shape: (B*T, 1)
        
        # Reshape back to (B, T, 1) to group by patient
        image_logits = image_logits.view(B, T, 1)
        
        # Mean pool over the temporal dimension (T=3) to get patient-level logit
        # Shape: (B, 1)
        patient_logits = image_logits.mean(dim=1)
        
        # Squeeze to (B,) for binary classification
        patient_logits = patient_logits.squeeze(dim=1)
        
        return patient_logits


def create_bmode_mean_model(
    backbone: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
    pretrained: bool = True
) -> nn.Module:
    """Create a B-mode mean pooling model.
    
    Convenience function to instantiate a BModeMeanPoolingModel with default
    or specified parameters.
    
    Args:
        backbone: ResNet architecture to use ('resnet18' or 'resnet34').
            Default: 'resnet18'.
        pretrained: Whether to use ImageNet pretrained weights. Default: True.
    
    Returns:
        BModeMeanPoolingModel instance ready for training or inference.
    
    Example:
        >>> model = create_bmode_mean_model(backbone="resnet18", pretrained=True)
        >>> # Input: (batch_size, 3, 3, 224, 224)
        >>> # Output: (batch_size,)
        >>> output = model(images)
    """
    model = BModeMeanPoolingModel(
        backbone=backbone,
        pretrained=pretrained,
        num_classes=1
    )
    return model


class BModeAttentionPoolingModel(nn.Module):
    """CNN model with attention pooling over multiple B-mode images per patient.
    
    This model processes 3 B-mode images per patient by:
    1. Extracting feature vectors from each image using a CNN backbone
    2. Computing attention weights for each image based on its features
    3. Aggregating features using attention-weighted sum
    4. Producing a patient-level prediction from the aggregated features
    
    Architecture:
    - Uses a ResNet backbone (ResNet18 or ResNet34) with ImageNet pretrained weights
    - Removes the final FC layer to extract feature vectors
    - Attention mechanism: a_i = v^T * tanh(W f_i + b)
    - Softmax to get attention weights: alpha_i = softmax(a_i)
    - Weighted sum: f_patient = sum_i alpha_i * f_i
    - Final linear layer to output patient-level logit
    
    Input shape: (B, T=3, C, H, W) where:
        B = batch size
        T = 3 (number of B-mode images per patient)
        C = 3 (RGB channels)
        H, W = 224 (image height and width after preprocessing)
    
    Output shape: (B,) - patient-level logits
    """
    
    def __init__(
        self,
        backbone: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
        pretrained: bool = True,
        feature_dim: int = 512,
        attention_hidden: int = 128
    ):
        """Initialize the B-mode attention pooling model.
        
        Args:
            backbone: Backbone architecture to use ('resnet18', 'resnet34', 
                'efficientnetv2_b0', 'efficientnetv2_b2').
            pretrained: Whether to use ImageNet pretrained weights. Default: True.
            feature_dim: Dimension of feature vectors extracted from each image.
                Should match the backbone's feature dimension. Default: 512.
            attention_hidden: Hidden dimension for attention network. Default: 128.
        """
        super().__init__()
        
        # Load backbone
        self.backbone, backbone_feature_dim = load_backbone(backbone, pretrained)
        
        # Verify feature_dim matches backbone
        if feature_dim != backbone_feature_dim:
            logger.warning(
                f"feature_dim ({feature_dim}) does not match backbone feature dim "
                f"({backbone_feature_dim}). Using backbone feature dim."
            )
            feature_dim = backbone_feature_dim
        
        # Remove the final FC/classifier layer to extract features
        if hasattr(self.backbone, 'fc'):  # ResNet
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):  # EfficientNet
            # Keep only the dropout, remove the linear layer
            self.backbone.classifier = nn.Sequential(
                self.backbone.classifier[0],  # Dropout
                nn.Identity()
            )
        else:
            raise ValueError(f"Unknown backbone structure for {backbone}")
        
        self.feature_dim = feature_dim
        
        # Attention network
        # a_i = v^T * tanh(W f_i + b)
        # W: (attention_hidden, feature_dim)
        # b: (attention_hidden,)
        # v: (attention_hidden,)
        self.attention_W = nn.Linear(feature_dim, attention_hidden)
        self.attention_b = nn.Parameter(torch.zeros(attention_hidden))
        self.attention_v = nn.Parameter(torch.randn(attention_hidden))
        
        # Final classification layer
        self.fc = nn.Linear(feature_dim, 1)
        
        # Store backbone name for reference
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.attention_hidden = attention_hidden
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, T=3, C, H, W) where:
                B = batch size
                T = 3 (number of B-mode images per patient)
                C = 3 (RGB channels)
                H, W = 224 (image dimensions)
        
        Returns:
            Tensor of shape (B,) containing patient-level logits.
            Each logit is computed from attention-weighted aggregation of image features.
        """
        # x shape: (B, T=3, C, H, W)
        B, T, C, H, W = x.shape
        
        # Reshape to process all images: (B*T, C, H, W)
        x = x.view(B * T, C, H, W)
        
        # Extract features from backbone: (B*T, feature_dim)
        # Backbone removes final FC, so we get feature vectors
        features = self.backbone(x)  # Shape: (B*T, feature_dim)
        
        # Reshape to group by patient: (B, T, feature_dim)
        features = features.view(B, T, self.feature_dim)
        
        # Compute attention scores for each image
        # a_i = v^T * tanh(W f_i + b)
        # features: (B, T, feature_dim)
        # attention_W(features): (B, T, attention_hidden)
        # attention_b: (attention_hidden,) -> broadcast to (B, T, attention_hidden)
        attention_hidden_features = self.attention_W(features) + self.attention_b  # (B, T, attention_hidden)
        attention_hidden_features = torch.tanh(attention_hidden_features)  # (B, T, attention_hidden)
        
        # Compute attention scores: v^T * attention_hidden_features
        # attention_v: (attention_hidden,) -> (attention_hidden, 1) for matmul
        # attention_hidden_features: (B, T, attention_hidden)
        # Result: (B, T, 1) -> (B, T)
        attention_scores = torch.matmul(
            attention_hidden_features,
            self.attention_v.unsqueeze(-1)  # (attention_hidden, 1)
        ).squeeze(-1)  # (B, T)
        
        # Apply softmax to get attention weights: (B, T)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (B, T)
        
        # Compute weighted sum: f_patient = sum_i alpha_i * f_i
        # attention_weights: (B, T) -> (B, T, 1)
        # features: (B, T, feature_dim)
        # patient_features: (B, feature_dim)
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # (B, T, 1)
        patient_features = (attention_weights_expanded * features).sum(dim=1)  # (B, feature_dim)
        
        # Apply final linear layer to get patient-level logits: (B, 1)
        patient_logits = self.fc(patient_features)  # (B, 1)
        
        # Squeeze to (B,) for binary classification
        patient_logits = patient_logits.squeeze(-1)  # (B,)
        
        return patient_logits


def create_bmode_attention_model(
    backbone: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
    pretrained: bool = True,
    feature_dim: int = 512,
    attention_hidden: int = 128
) -> nn.Module:
    """Create a B-mode attention pooling model.
    
    Convenience function to instantiate a BModeAttentionPoolingModel with default
    or specified parameters.
    
    Args:
        backbone: ResNet architecture to use ('resnet18' or 'resnet34').
            Default: 'resnet18'.
        pretrained: Whether to use ImageNet pretrained weights. Default: True.
        feature_dim: Dimension of feature vectors extracted from each image.
            Should match the backbone's feature dimension (512 for ResNet18/34).
            Default: 512.
        attention_hidden: Hidden dimension for attention network. Default: 128.
    
    Returns:
        BModeAttentionPoolingModel instance ready for training or inference.
    
    Example:
        >>> model = create_bmode_attention_model(
        ...     backbone="resnet18",
        ...     pretrained=True,
        ...     feature_dim=512,
        ...     attention_hidden=128
        ... )
        >>> # Input: (batch_size, 3, 3, 224, 224)
        >>> # Output: (batch_size,)
        >>> output = model(images)
    """
    model = BModeAttentionPoolingModel(
        backbone=backbone,
        pretrained=pretrained,
        feature_dim=feature_dim,
        attention_hidden=attention_hidden
    )
    return model


class BModeRadiomicsFusionModel(nn.Module):
    """CNN + Radiomics fusion model for liver fibrosis classification.
    
    This model combines B-mode image features (from CNN) with radiomics features
    to produce patient-level predictions. It uses a two-branch architecture:
    
    1. CNN branch: Extracts features from 3 B-mode images using mean pooling
    2. Radiomics branch: Processes radiomics feature vectors
    3. Fusion: Concatenates both branches and applies MLP for final prediction
    
    Architecture:
    - CNN backbone (ResNet18/34) extracts features from each image
    - Mean pooling over 3 images to get patient-level image features
    - Linear layer + ReLU processes radiomics features
    - Concatenation of image and radiomics features
    - Fusion MLP: Linear -> ReLU -> Dropout -> Linear -> logits
    
    Input shapes:
    - x: (B, T=3, C, H, W) where T=3 is number of B-mode images
    - rad_features: (B, R) where R is number of radiomics features
    
    Output shape: (B,) - patient-level logits
    """
    
    def __init__(
        self,
        backbone: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
        pretrained: bool = True,
        feature_dim: int = 512,
        radiomics_dim: int = 64,
        fusion_hidden: int = 128,
        dropout: float = 0.5
    ):
        """Initialize the B-mode + radiomics fusion model.
        
        Args:
            backbone: Backbone architecture to use ('resnet18', 'resnet34', 
                'efficientnetv2_b0', 'efficientnetv2_b2').
            pretrained: Whether to use ImageNet pretrained weights. Default: True.
            feature_dim: Dimension of CNN feature vectors (should match backbone).
                Default: 512.
            radiomics_dim: Output dimension for processed radiomics features.
                Default: 64.
            fusion_hidden: Hidden dimension for fusion MLP. Default: 128.
            dropout: Dropout probability in fusion MLP. Default: 0.5.
        """
        super().__init__()
        
        # Load backbone
        self.backbone, backbone_feature_dim = load_backbone(backbone, pretrained)
        
        # Verify feature_dim matches backbone
        if feature_dim != backbone_feature_dim:
            logger.warning(
                f"feature_dim ({feature_dim}) does not match backbone feature dim "
                f"({backbone_feature_dim}). Using backbone feature dim."
            )
            feature_dim = backbone_feature_dim
        
        # Remove the final FC/classifier layer to extract features
        if hasattr(self.backbone, 'fc'):  # ResNet
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):  # EfficientNet
            # Keep only the dropout, remove the linear layer
            self.backbone.classifier = nn.Sequential(
                self.backbone.classifier[0],  # Dropout
                nn.Identity()
            )
        else:
            raise ValueError(f"Unknown backbone structure for {backbone}")
        
        self.feature_dim = feature_dim
        
        # Radiomics branch: Linear layer + ReLU
        # Input dimension will be set during first forward pass or can be specified
        # For now, we'll make it flexible
        self.radiomics_projection = None  # Will be initialized in forward if needed
        self.radiomics_dim = radiomics_dim
        self._radiomics_input_dim = None  # Will be set from first input
        
        # Fusion MLP
        # Input: feature_dim (CNN) + radiomics_dim (processed radiomics)
        fusion_input_dim = feature_dim + radiomics_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1)
        )
        
        # Store configuration
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.fusion_hidden = fusion_hidden
        self.dropout = dropout
    
    def _initialize_radiomics_projection(self, radiomics_input_dim: int):
        """Initialize radiomics projection layer.
        
        Args:
            radiomics_input_dim: Number of input radiomics features.
        """
        if self.radiomics_projection is None or self._radiomics_input_dim != radiomics_input_dim:
            self.radiomics_projection = nn.Sequential(
                nn.Linear(radiomics_input_dim, self.radiomics_dim),
                nn.ReLU()
            )
            self._radiomics_input_dim = radiomics_input_dim
            # Move to same device as other parameters
            if next(self.fusion_mlp.parameters()).is_cuda:
                self.radiomics_projection = self.radiomics_projection.cuda()
    
    def forward(
        self,
        x: torch.Tensor,
        rad_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, T=3, C, H, W) where:
                B = batch size
                T = 3 (number of B-mode images per patient)
                C = 3 (RGB channels)
                H, W = 224 (image dimensions)
            rad_features: Radiomics feature tensor of shape (B, R) where:
                B = batch size
                R = number of radiomics features per patient
        
        Returns:
            Tensor of shape (B,) containing patient-level logits.
        """
        # Validate input shapes
        if len(x.shape) != 5:
            raise ValueError(f"Expected x shape (B, T, C, H, W), got {x.shape}")
        if len(rad_features.shape) != 2:
            raise ValueError(f"Expected rad_features shape (B, R), got {rad_features.shape}")
        
        B, T, C, H, W = x.shape
        B_rad, R = rad_features.shape
        
        if B != B_rad:
            raise ValueError(
                f"Batch size mismatch: x has {B} samples, "
                f"rad_features has {B_rad} samples"
            )
        
        # Initialize radiomics projection if needed
        self._initialize_radiomics_projection(R)
        
        # CNN branch: Extract features from images
        # Reshape to process all images: (B*T, C, H, W)
        x_flat = x.view(B * T, C, H, W)
        
        # Extract features from backbone: (B*T, feature_dim)
        image_features = self.backbone(x_flat)  # (B*T, feature_dim)
        
        # Reshape to group by patient: (B, T, feature_dim)
        image_features = image_features.view(B, T, self.feature_dim)
        
        # Mean pool over temporal dimension: (B, feature_dim)
        f_img = image_features.mean(dim=1)  # (B, feature_dim)
        
        # Radiomics branch: Process radiomics features
        # Linear + ReLU: (B, R) -> (B, radiomics_dim)
        g_rad = self.radiomics_projection(rad_features)  # (B, radiomics_dim)
        
        # Fusion: Concatenate image and radiomics features
        # z = [f_img; g_rad]: (B, feature_dim + radiomics_dim)
        z = torch.cat([f_img, g_rad], dim=1)  # (B, feature_dim + radiomics_dim)
        
        # Fusion MLP: (B, feature_dim + radiomics_dim) -> (B, 1)
        logits = self.fusion_mlp(z)  # (B, 1)
        
        # Squeeze to (B,) for binary classification
        logits = logits.squeeze(-1)  # (B,)
        
        return logits


def create_bmode_radiomics_fusion_model(
    backbone: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
    pretrained: bool = True,
    feature_dim: int = 512,
    radiomics_dim: int = 64,
    fusion_hidden: int = 128,
    dropout: float = 0.5,
    radiomics_input_dim: Optional[int] = None
) -> nn.Module:
    """Create a B-mode + radiomics fusion model.
    
    Convenience function to instantiate a BModeRadiomicsFusionModel with default
    or specified parameters.
    
    Args:
        backbone: ResNet architecture to use ('resnet18' or 'resnet34').
            Default: 'resnet18'.
        pretrained: Whether to use ImageNet pretrained weights. Default: True.
        feature_dim: Dimension of CNN feature vectors. Default: 512.
        radiomics_dim: Output dimension for processed radiomics features.
            Default: 64.
        fusion_hidden: Hidden dimension for fusion MLP. Default: 128.
        dropout: Dropout probability in fusion MLP. Default: 0.5.
        radiomics_input_dim: Number of input radiomics features. If None,
            will be inferred from first forward pass. Default: None.
    
    Returns:
        BModeRadiomicsFusionModel instance ready for training or inference.
    
    Example:
        >>> model = create_bmode_radiomics_fusion_model(
        ...     backbone="resnet18",
        ...     pretrained=True,
        ...     radiomics_input_dim=10
        ... )
        >>> # Input: images (batch_size, 3, 3, 224, 224), radiomics (batch_size, 10)
        >>> # Output: (batch_size,)
        >>> output = model(images, radiomics)
    """
    model = BModeRadiomicsFusionModel(
        backbone=backbone,
        pretrained=pretrained,
        feature_dim=feature_dim,
        radiomics_dim=radiomics_dim,
        fusion_hidden=fusion_hidden,
        dropout=dropout
    )
    
    # Initialize radiomics projection if input dimension is provided
    if radiomics_input_dim is not None:
        model._initialize_radiomics_projection(radiomics_input_dim)
    
    return model


class BModeClinicalFusionModel(nn.Module):
    """CNN + Clinical fusion model for liver fibrosis classification.
    
    This model combines B-mode image features (from CNN) with clinical features
    to produce patient-level predictions. It supports both mean pooling (C1)
    and attention pooling (C2) for image feature aggregation.
    
    Architecture:
    - CNN backbone (ResNet18/34) extracts features from each of 3 images
    - Image pooling: Mean pooling (C1) or Attention pooling (C2)
    - Clinical branch: MLP processes clinical features
    - Fusion: Concatenates both branches and applies MLP for final prediction
    
    Input shapes:
    - x: (B, T=3, C, H, W) where T=3 is number of B-mode images
    - clinical_features: (B, K) where K is number of clinical features
    
    Output shape: (B,) - patient-level logits
    """
    
    def __init__(
        self,
        backbone: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
        pretrained: bool = True,
        feature_dim: int = 512,
        clinical_dim: int = 32,
        fusion_hidden: int = 128,
        pooling: Literal["mean", "attention"] = "mean",
        attention_hidden: int = 128,
        dropout: float = 0.5,
        clinical_input_dim: Optional[int] = None
    ):
        """Initialize the B-mode + clinical fusion model.
        
        Args:
            backbone: Backbone architecture to use ('resnet18', 'resnet34', 
                'efficientnetv2_b0', 'efficientnetv2_b2').
            pretrained: Whether to use ImageNet pretrained weights. Default: True.
            feature_dim: Dimension of CNN feature vectors (should match backbone).
                Default: 512.
            clinical_dim: Output dimension for processed clinical features.
                Default: 32.
            fusion_hidden: Hidden dimension for fusion MLP. Default: 128.
            pooling: Pooling method for images: 'mean' (C1) or 'attention' (C2).
                Default: 'mean'.
            attention_hidden: Hidden dimension for attention network (only used
                if pooling='attention'). Default: 128.
            dropout: Dropout probability in fusion MLP. Default: 0.5.
            clinical_input_dim: Number of input clinical features. If None,
                will be inferred from first forward pass. Default: None.
        """
        super().__init__()
        
        # Load backbone
        self.backbone, backbone_feature_dim = load_backbone(backbone, pretrained)
        
        # Verify feature_dim matches backbone
        if feature_dim != backbone_feature_dim:
            logger.warning(
                f"feature_dim ({feature_dim}) does not match backbone feature dim "
                f"({backbone_feature_dim}). Using backbone feature dim."
            )
            feature_dim = backbone_feature_dim
        
        # Remove the final FC/classifier layer to extract features
        if hasattr(self.backbone, 'fc'):  # ResNet
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):  # EfficientNet
            # Keep only the dropout, remove the linear layer
            self.backbone.classifier = nn.Sequential(
                self.backbone.classifier[0],  # Dropout
                nn.Identity()
            )
        else:
            raise ValueError(f"Unknown backbone structure for {backbone}")
        
        self.feature_dim = feature_dim
        self.pooling = pooling
        
        # Image pooling branch
        if pooling == "mean":
            # Mean pooling: no additional parameters needed
            self.image_pooling = None
        elif pooling == "attention":
            # Attention pooling: same as A2
            self.attention_W = nn.Linear(feature_dim, attention_hidden)
            self.attention_b = nn.Parameter(torch.zeros(attention_hidden))
            self.attention_v = nn.Parameter(torch.randn(attention_hidden))
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
        # Input: feature_dim (CNN) + clinical_dim (processed clinical)
        fusion_input_dim = feature_dim + clinical_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1)
        )
        
        # Store configuration
        self.backbone_name = backbone
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
    
    def _pool_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        """Pool image features across temporal dimension.
        
        Args:
            image_features: Tensor of shape (B, T, feature_dim).
        
        Returns:
            Pooled features of shape (B, feature_dim).
        """
        B, T, feature_dim = image_features.shape
        
        if self.pooling == "mean":
            # Mean pooling: (B, T, feature_dim) -> (B, feature_dim)
            f_img = image_features.mean(dim=1)
        elif self.pooling == "attention":
            # Attention pooling: same as A2
            # Compute attention scores
            attention_hidden_features = self.attention_W(image_features) + self.attention_b
            attention_hidden_features = torch.tanh(attention_hidden_features)
            
            # Compute attention scores: v^T * attention_hidden_features
            attention_scores = torch.matmul(
                attention_hidden_features,
                self.attention_v.unsqueeze(-1)
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
        x: torch.Tensor,
        clinical_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, T=3, C, H, W) where:
                B = batch size
                T = 3 (number of B-mode images per patient)
                C = 3 (RGB channels)
                H, W = 224 (image dimensions)
            clinical_features: Clinical feature tensor of shape (B, K) where:
                B = batch size
                K = number of clinical features per patient
        
        Returns:
            Tensor of shape (B,) containing patient-level logits.
        """
        # Validate input shapes
        if len(x.shape) != 5:
            raise ValueError(f"Expected x shape (B, T, C, H, W), got {x.shape}")
        if len(clinical_features.shape) != 2:
            raise ValueError(
                f"Expected clinical_features shape (B, K), got {clinical_features.shape}"
            )
        
        B, T, C, H, W = x.shape
        B_clin, K = clinical_features.shape
        
        if B != B_clin:
            raise ValueError(
                f"Batch size mismatch: x has {B} samples, "
                f"clinical_features has {B_clin} samples"
            )
        
        # Initialize clinical projection if needed
        self._initialize_clinical_projection(K)
        
        # CNN branch: Extract features from images
        # Reshape to process all images: (B*T, C, H, W)
        x_flat = x.view(B * T, C, H, W)
        
        # Extract features from backbone: (B*T, feature_dim)
        image_features = self.backbone(x_flat)  # (B*T, feature_dim)
        
        # Reshape to group by patient: (B, T, feature_dim)
        image_features = image_features.view(B, T, self.feature_dim)
        
        # Pool image features: (B, feature_dim)
        f_img = self._pool_image_features(image_features)
        
        # Clinical branch: Process clinical features
        # MLP: (B, K) -> (B, clinical_dim)
        g_clin = self.clinical_projection(clinical_features)  # (B, clinical_dim)
        
        # Fusion: Concatenate image and clinical features
        # z = [f_img; g_clin]: (B, feature_dim + clinical_dim)
        z = torch.cat([f_img, g_clin], dim=1)  # (B, feature_dim + clinical_dim)
        
        # Fusion MLP: (B, feature_dim + clinical_dim) -> (B, 1)
        logits = self.fusion_mlp(z)  # (B, 1)
        
        # Squeeze to (B,) for binary classification
        logits = logits.squeeze(-1)  # (B,)
        
        return logits


def create_bmode_clinical_fusion_model(
    backbone: Literal["resnet18", "resnet34", "efficientnetv2_b0", "efficientnetv2_b2"] = "resnet18",
    pretrained: bool = True,
    feature_dim: int = 512,
    clinical_dim: int = 32,
    fusion_hidden: int = 128,
    pooling: Literal["mean", "attention"] = "mean",
    attention_hidden: int = 128,
    dropout: float = 0.5,
    clinical_input_dim: Optional[int] = None
) -> nn.Module:
    """Create a B-mode + clinical fusion model.
    
    Convenience function to instantiate a BModeClinicalFusionModel with default
    or specified parameters.
    
    Args:
        backbone: ResNet architecture to use ('resnet18' or 'resnet34').
            Default: 'resnet18'.
        pretrained: Whether to use ImageNet pretrained weights. Default: True.
        feature_dim: Dimension of CNN feature vectors. Default: 512.
        clinical_dim: Output dimension for processed clinical features.
            Default: 32.
        fusion_hidden: Hidden dimension for fusion MLP. Default: 128.
        pooling: Pooling method: 'mean' (C1) or 'attention' (C2). Default: 'mean'.
        attention_hidden: Hidden dimension for attention (only if pooling='attention').
            Default: 128.
        dropout: Dropout probability in fusion MLP. Default: 0.5.
        clinical_input_dim: Number of input clinical features. If None,
            will be inferred from first forward pass. Default: None.
    
    Returns:
        BModeClinicalFusionModel instance ready for training or inference.
    
    Example:
        >>> # C1: Mean pooling
        >>> model_c1 = create_bmode_clinical_fusion_model(
        ...     pooling="mean",
        ...     clinical_input_dim=5
        ... )
        >>> # C2: Attention pooling
        >>> model_c2 = create_bmode_clinical_fusion_model(
        ...     pooling="attention",
        ...     clinical_input_dim=5
        ... )
        >>> # Input: images (batch_size, 3, 3, 224, 224), clinical (batch_size, 5)
        >>> # Output: (batch_size,)
        >>> output = model_c1(images, clinical_features)
    """
    model = BModeClinicalFusionModel(
        backbone=backbone,
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

