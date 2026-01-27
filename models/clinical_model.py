"""Clinical-only model for liver fibrosis classification.

This module contains a simple MLP architecture for processing clinical features.
"""

import torch
import torch.nn as nn


class ClinicalModel(nn.Module):
    """Simple MLP for clinical data classification.
    
    Architecture:
    - LayerNorm
    - Linear -> ReLU -> Dropout
    - Linear (Output)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        """Initialize the model.
        
        Args:
            input_dim: Number of input clinical features.
            hidden_dim: Hidden dimension size.
            num_classes: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, input_dim).
            
        Returns:
            Logits tensor of shape (B, num_classes).
        """
        return self.network(x)
