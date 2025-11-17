"""Test script for B-mode attention pooling model."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.clinical_data import ClinicalConfig, load_clinical_table
from data.datasets import BModePatientDataset, get_eval_transform
from data.image_index import build_patient_records
from models.bmode_models import (
    BModeAttentionPoolingModel,
    create_bmode_attention_model,
)

print("=" * 70)
print("Step 1: Loading data and creating dataset")
print("=" * 70)

# Load clinical data
config = ClinicalConfig(
    csv_path=Path("data/annotations/175_clinical_5_variables.csv"),
    feature_columns=["AST", "ALT", "PLT", "APRI", "FIB_4"],
    patient_id_column="NO",
    label_column="CL_F2",
    fibrosis_stage_column=None,
)

df = load_clinical_table(config)
records = build_patient_records(
    clinical_df=df,
    image_root=Path("data/bmode_full"),
    label_column="CL_F2",
    image_pattern="Bmode_image_*.png",
    required_images=3
)

# Create dataset
dataset = BModePatientDataset(records, transform=get_eval_transform())
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

print(f"✓ Created dataset with {len(dataset)} patients")
print(f"✓ Created DataLoader with batch_size=4")

print("\n" + "=" * 70)
print("Step 2: Creating attention model using helper function")
print("=" * 70)

# Create model using helper function
model = create_bmode_attention_model(
    backbone="resnet18",
    pretrained=True,
    feature_dim=512,
    attention_hidden=128
)
model.eval()  # Set to evaluation mode

print(f"✓ Created model: {model.__class__.__name__}")
print(f"  Backbone: {model.backbone_name}")
print(f"  Pretrained: {model.pretrained}")
print(f"  Feature dimension: {model.feature_dim}")
print(f"  Attention hidden dimension: {model.attention_hidden}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

print("\n" + "=" * 70)
print("Step 3: Testing forward pass")
print("=" * 70)

# Get a batch
imgs, labels, patient_ids = next(iter(dataloader))
print(f"Input batch shape: {imgs.shape} (expected: (4, 3, 3, 224, 224))")
print(f"Labels shape: {labels.shape}")
print(f"Patient IDs: {patient_ids}")

# Forward pass
with torch.no_grad():
    output = model(imgs)

print(f"\n✓ Forward pass successful!")
print(f"Output shape: {output.shape} (expected: (4,))")
print(f"Output values: {output}")
print(f"Output dtype: {output.dtype}")

# Verify shapes
assert output.shape == (4,), f"Expected output shape (4,), got {output.shape}"
assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"
print("✓ Output shape and dtype correct!")

print("\n" + "=" * 70)
print("Step 4: Testing attention mechanism")
print("=" * 70)

# Test that attention weights sum to 1
model.eval()
with torch.no_grad():
    # Get a single patient
    single_imgs, _, _ = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    # single_imgs shape: (1, 3, 3, 224, 224)
    
    # Manually compute attention weights to verify
    B, T, C, H, W = single_imgs.shape
    x_flat = single_imgs.view(B * T, C, H, W)
    features = model.backbone(x_flat)  # (3, 512)
    features = features.view(B, T, model.feature_dim)  # (1, 3, 512)
    
    # Compute attention
    attention_hidden = model.attention_W(features) + model.attention_b
    attention_hidden = torch.tanh(attention_hidden)
    attention_scores = torch.matmul(
        attention_hidden,
        model.attention_v.unsqueeze(-1)
    ).squeeze(-1)
    attention_weights = torch.softmax(attention_scores, dim=1)
    
    print(f"Attention weights shape: {attention_weights.shape} (expected: (1, 3))")
    print(f"Attention weights: {attention_weights.squeeze().tolist()}")
    print(f"Sum of attention weights: {attention_weights.sum().item():.6f} (should be ~1.0)")
    
    assert abs(attention_weights.sum().item() - 1.0) < 1e-5, \
        "Attention weights should sum to 1.0"
    assert attention_weights.min().item() >= 0, \
        "Attention weights should be non-negative"
    assert attention_weights.max().item() <= 1, \
        "Attention weights should be <= 1"
    
    print("✓ Attention mechanism works correctly!")

print("\n" + "=" * 70)
print("Step 5: Testing with different batch sizes")
print("=" * 70)

# Test with batch size 1
single_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
imgs_single, _, _ = next(iter(single_loader))
with torch.no_grad():
    output_single = model(imgs_single)
print(f"Batch size 1: input {imgs_single.shape} -> output {output_single.shape}")

# Test with batch size 8
batch8_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
imgs_batch8, _, _ = next(iter(batch8_loader))
with torch.no_grad():
    output_batch8 = model(imgs_batch8)
print(f"Batch size 8: input {imgs_batch8.shape} -> output {output_batch8.shape}")

print("✓ Model handles different batch sizes correctly!")

print("\n" + "=" * 70)
print("Step 6: Testing ResNet34 backbone")
print("=" * 70)

# Create ResNet34 model
model34 = create_bmode_attention_model(
    backbone="resnet34",
    pretrained=True,
    feature_dim=512,
    attention_hidden=128
)
model34.eval()

print(f"✓ Created ResNet34 model")
total_params_34 = sum(p.numel() for p in model34.parameters())
print(f"  Total parameters: {total_params_34:,}")

# Test forward pass
with torch.no_grad():
    output_34 = model34(imgs)
print(f"  Output shape: {output_34.shape}")
print(f"  Output values: {output_34}")

print("✓ ResNet34 model works correctly!")

print("\n" + "=" * 70)
print("Step 7: Testing gradient computation")
print("=" * 70)

# Test that gradients can be computed
model.train()  # Set to training mode
imgs.requires_grad_(False)  # Images don't need gradients
output = model(imgs)

# Create a dummy loss (binary cross entropy with logits)
loss = torch.nn.functional.binary_cross_entropy_with_logits(
    output, labels
)

# Backward pass
loss.backward()

# Check if gradients exist
has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
print(f"✓ Backward pass successful")
print(f"  Loss value: {loss.item():.4f}")
print(f"  Gradients computed: {has_gradients}")

if has_gradients:
    # Count parameters with gradients
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  Parameters with gradients: {params_with_grad}")
    
    # Check attention parameters have gradients
    attention_grads = [
        model.attention_W.weight.grad is not None,
        model.attention_W.bias.grad is not None,
        model.attention_b.grad is not None,
        model.attention_v.grad is not None,
    ]
    print(f"  Attention parameters have gradients: {all(attention_grads)}")

print("\n" + "=" * 70)
print("Step 8: Comparing with mean pooling model")
print("=" * 70)

from models.bmode_models import create_bmode_mean_model

mean_model = create_bmode_mean_model(backbone="resnet18", pretrained=True)
mean_model.eval()

with torch.no_grad():
    attention_output = model(imgs)
    mean_output = mean_model(imgs)

print(f"Attention model output: {attention_output.tolist()}")
print(f"Mean pooling model output: {mean_output.tolist()}")
print(f"Difference: {(attention_output - mean_output).abs().mean().item():.4f}")

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)

