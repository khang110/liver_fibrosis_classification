"""Test script for B-mode + radiomics fusion model."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.clinical_data import ClinicalConfig, load_clinical_table
from data.datasets import BModePatientDataset, get_eval_transform
from data.image_index import build_patient_records
from data.radiomics_features import build_patient_radiomics
from models.bmode_models import create_bmode_radiomics_fusion_model

print("=" * 70)
print("Step 1: Loading data and creating datasets")
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
print("Step 2: Extracting radiomics features")
print("=" * 70)

# Extract radiomics features
radiomics_df = build_patient_radiomics(records, roi_masks=None)
print(f"✓ Extracted radiomics for {len(radiomics_df)} patients")
print(f"Radiomics shape: {radiomics_df.shape}")
print(f"Radiomics columns: {list(radiomics_df.columns)}")

# Get radiomics input dimension
radiomics_input_dim = len(radiomics_df.columns)
print(f"Radiomics input dimension: {radiomics_input_dim}")

print("\n" + "=" * 70)
print("Step 3: Creating fusion model")
print("=" * 70)

# Create model
model = create_bmode_radiomics_fusion_model(
    backbone="resnet18",
    pretrained=True,
    feature_dim=512,
    radiomics_dim=64,
    fusion_hidden=128,
    dropout=0.5,
    radiomics_input_dim=radiomics_input_dim
)
model.eval()

print(f"✓ Created model: {model.__class__.__name__}")
print(f"  Backbone: {model.backbone_name}")
print(f"  Feature dimension: {model.feature_dim}")
print(f"  Radiomics dimension: {model.radiomics_dim}")
print(f"  Fusion hidden: {model.fusion_hidden}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

print("\n" + "=" * 70)
print("Step 4: Testing forward pass")
print("=" * 70)

# Get a batch
imgs, labels, patient_ids = next(iter(dataloader))
print(f"Images shape: {imgs.shape} (expected: (4, 3, 3, 224, 224))")
print(f"Labels shape: {labels.shape}")
print(f"Patient IDs: {patient_ids}")

# Get corresponding radiomics features
# Note: patient_ids are strings, need to match with radiomics_df index
radiomics_batch = []
for pid in patient_ids:
    if pid in radiomics_df.index:
        radiomics_batch.append(radiomics_df.loc[pid].values)
    else:
        raise ValueError(f"Patient {pid} not found in radiomics DataFrame")

radiomics_tensor = torch.tensor(np.array(radiomics_batch), dtype=torch.float32)
print(f"Radiomics shape: {radiomics_tensor.shape} (expected: (4, {radiomics_input_dim}))")

# Forward pass
with torch.no_grad():
    output = model(imgs, radiomics_tensor)

print(f"\n✓ Forward pass successful!")
print(f"Output shape: {output.shape} (expected: (4,))")
print(f"Output values: {output}")
print(f"Output dtype: {output.dtype}")

# Verify shapes
assert output.shape == (4,), f"Expected output shape (4,), got {output.shape}"
assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"
print("✓ Output shape and dtype correct!")

print("\n" + "=" * 70)
print("Step 5: Testing with different batch sizes")
print("=" * 70)

# Test with batch size 1
single_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
imgs_single, _, patient_ids_single = next(iter(single_loader))
radiomics_single = torch.tensor(
    [radiomics_df.loc[patient_ids_single[0]].values],
    dtype=torch.float32
)

with torch.no_grad():
    output_single = model(imgs_single, radiomics_single)
print(f"Batch size 1: images {imgs_single.shape}, radiomics {radiomics_single.shape} -> output {output_single.shape}")

# Test with batch size 8
batch8_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
imgs_batch8, _, patient_ids_batch8 = next(iter(batch8_loader))
radiomics_batch8 = torch.tensor(
    np.array([radiomics_df.loc[pid].values for pid in patient_ids_batch8]),
    dtype=torch.float32
)

with torch.no_grad():
    output_batch8 = model(imgs_batch8, radiomics_batch8)
print(f"Batch size 8: images {imgs_batch8.shape}, radiomics {radiomics_batch8.shape} -> output {output_batch8.shape}")

print("✓ Model handles different batch sizes correctly!")

print("\n" + "=" * 70)
print("Step 6: Testing gradient computation")
print("=" * 70)

# Test that gradients can be computed
model.train()
imgs.requires_grad_(False)
radiomics_tensor.requires_grad_(False)
output = model(imgs, radiomics_tensor)

# Create a dummy loss
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
    # Check that all branches have gradients
    backbone_grads = any(p.grad is not None for p in model.backbone.parameters() if p.requires_grad)
    radiomics_grads = model.radiomics_projection is not None and any(
        p.grad is not None for p in model.radiomics_projection.parameters() if p.requires_grad
    )
    fusion_grads = any(p.grad is not None for p in model.fusion_mlp.parameters() if p.requires_grad)
    
    print(f"  Backbone has gradients: {backbone_grads}")
    print(f"  Radiomics projection has gradients: {radiomics_grads}")
    print(f"  Fusion MLP has gradients: {fusion_grads}")

print("\n" + "=" * 70)
print("Step 7: Testing model architecture components")
print("=" * 70)

model.eval()
with torch.no_grad():
    # Test CNN branch separately
    B, T, C, H, W = imgs.shape
    x_flat = imgs.view(B * T, C, H, W)
    image_features = model.backbone(x_flat)
    image_features = image_features.view(B, T, model.feature_dim)
    f_img = image_features.mean(dim=1)
    print(f"CNN branch output shape: {f_img.shape} (expected: ({B}, {model.feature_dim}))")
    
    # Test radiomics branch separately
    g_rad = model.radiomics_projection(radiomics_tensor)
    print(f"Radiomics branch output shape: {g_rad.shape} (expected: ({B}, {model.radiomics_dim}))")
    
    # Test fusion
    z = torch.cat([f_img, g_rad], dim=1)
    print(f"Concatenated features shape: {z.shape} (expected: ({B}, {model.feature_dim + model.radiomics_dim}))")
    
    logits = model.fusion_mlp(z)
    print(f"Fusion MLP output shape: {logits.shape} (expected: ({B}, 1))")

print("✓ All architecture components work correctly!")

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)

