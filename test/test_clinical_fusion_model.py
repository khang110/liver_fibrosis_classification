"""Test script for B-mode + clinical fusion models (C1 and C2)."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.clinical_data import (
    ClinicalConfig,
    ClinicalPreprocessor,
    load_clinical_table,
)
from data.datasets import BModePatientDataset, get_eval_transform
from data.image_index import build_patient_records
from models.bmode_models import create_bmode_clinical_fusion_model

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
print("Step 2: Preprocessing clinical features")
print("=" * 70)

# Create preprocessor and transform clinical data
preprocessor = ClinicalPreprocessor(
    numeric_features=["AST", "ALT", "PLT", "APRI", "FIB_4"],
    categorical_features=[],
)

preprocessor.fit(df)
X_clinical, _, _ = preprocessor.transform(df, label_column="CL_F2")

print(f"✓ Preprocessed clinical features")
print(f"Clinical features shape: {X_clinical.shape}")
print(f"Number of features: {X_clinical.shape[1]}")

clinical_input_dim = X_clinical.shape[1]
print(f"Clinical input dimension: {clinical_input_dim}")

print("\n" + "=" * 70)
print("Step 3: Testing C1 model (Mean pooling)")
print("=" * 70)

# Create C1 model (mean pooling)
model_c1 = create_bmode_clinical_fusion_model(
    backbone="resnet18",
    pretrained=True,
    feature_dim=512,
    clinical_dim=32,
    fusion_hidden=128,
    pooling="mean",
    dropout=0.5,
    clinical_input_dim=clinical_input_dim
)
model_c1.eval()

print(f"✓ Created C1 model: {model_c1.__class__.__name__}")
print(f"  Pooling: {model_c1.pooling}")
print(f"  Backbone: {model_c1.backbone_name}")
print(f"  Clinical dimension: {model_c1.clinical_dim}")

total_params = sum(p.numel() for p in model_c1.parameters())
print(f"  Total parameters: {total_params:,}")

# Get a batch
imgs, labels, patient_ids = next(iter(dataloader))
print(f"\nImages shape: {imgs.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Patient IDs: {patient_ids}")

# Get corresponding clinical features
clinical_batch = []
for pid in patient_ids:
    # Convert to int if needed (patient_ids are strings, df index might be int)
    try:
        pid_int = int(pid)
    except ValueError:
        pid_int = pid
    
    # Find index in df
    if pid_int in df.index:
        idx = df.index.get_loc(pid_int)
        clinical_batch.append(X_clinical[idx])
    elif pid in df.index:
        idx = df.index.get_loc(pid)
        clinical_batch.append(X_clinical[idx])
    else:
        raise ValueError(f"Patient {pid} (int: {pid_int}) not found in clinical DataFrame")

clinical_tensor = torch.tensor(np.array(clinical_batch), dtype=torch.float32)
print(f"Clinical features shape: {clinical_tensor.shape}")

# Forward pass
with torch.no_grad():
    output_c1 = model_c1(imgs, clinical_tensor)

print(f"\n✓ C1 forward pass successful!")
print(f"Output shape: {output_c1.shape} (expected: (4,))")
print(f"Output values: {output_c1}")

assert output_c1.shape == (4,), f"Expected output shape (4,), got {output_c1.shape}"
print("✓ C1 output shape correct!")

print("\n" + "=" * 70)
print("Step 4: Testing C2 model (Attention pooling)")
print("=" * 70)

# Create C2 model (attention pooling)
model_c2 = create_bmode_clinical_fusion_model(
    backbone="resnet18",
    pretrained=True,
    feature_dim=512,
    clinical_dim=32,
    fusion_hidden=128,
    pooling="attention",
    attention_hidden=128,
    dropout=0.5,
    clinical_input_dim=clinical_input_dim
)
model_c2.eval()

print(f"✓ Created C2 model: {model_c2.__class__.__name__}")
print(f"  Pooling: {model_c2.pooling}")
print(f"  Backbone: {model_c2.backbone_name}")
print(f"  Clinical dimension: {model_c2.clinical_dim}")

total_params_c2 = sum(p.numel() for p in model_c2.parameters())
print(f"  Total parameters: {total_params_c2:,}")

# Forward pass
with torch.no_grad():
    output_c2 = model_c2(imgs, clinical_tensor)

print(f"\n✓ C2 forward pass successful!")
print(f"Output shape: {output_c2.shape} (expected: (4,))")
print(f"Output values: {output_c2}")

assert output_c2.shape == (4,), f"Expected output shape (4,), got {output_c2.shape}"
print("✓ C2 output shape correct!")

print("\n" + "=" * 70)
print("Step 5: Comparing C1 and C2 outputs")
print("=" * 70)

print(f"C1 output: {output_c1.tolist()}")
print(f"C2 output: {output_c2.tolist()}")
print(f"Difference: {(output_c1 - output_c2).abs().mean().item():.4f}")

print("\n" + "=" * 70)
print("Step 6: Testing gradient computation")
print("=" * 70)

# Test C1 gradients
model_c1.train()
imgs.requires_grad_(False)
clinical_tensor.requires_grad_(False)
output = model_c1(imgs, clinical_tensor)

loss = torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
loss.backward()

has_gradients = any(p.grad is not None for p in model_c1.parameters() if p.requires_grad)
print(f"✓ C1 backward pass successful")
print(f"  Loss: {loss.item():.4f}")
print(f"  Gradients computed: {has_gradients}")

# Test C2 gradients
model_c2.train()
output_c2_grad = model_c2(imgs, clinical_tensor)
loss_c2 = torch.nn.functional.binary_cross_entropy_with_logits(output_c2_grad, labels)
loss_c2.backward()

has_gradients_c2 = any(p.grad is not None for p in model_c2.parameters() if p.requires_grad)
print(f"✓ C2 backward pass successful")
print(f"  Loss: {loss_c2.item():.4f}")
print(f"  Gradients computed: {has_gradients_c2}")

print("\n" + "=" * 70)
print("Step 7: Testing model architecture components")
print("=" * 70)

model_c1.eval()
with torch.no_grad():
    B, T, C, H, W = imgs.shape
    x_flat = imgs.view(B * T, C, H, W)
    image_features = model_c1.backbone(x_flat)
    image_features = image_features.view(B, T, model_c1.feature_dim)
    
    # Test pooling
    f_img_mean = model_c1._pool_image_features(image_features)
    print(f"C1 (mean) pooled features shape: {f_img_mean.shape}")
    
    # Test clinical branch
    g_clin = model_c1.clinical_projection(clinical_tensor)
    print(f"Clinical features shape: {g_clin.shape}")
    
    # Test fusion
    z = torch.cat([f_img_mean, g_clin], dim=1)
    print(f"Concatenated features shape: {z.shape}")
    
    logits = model_c1.fusion_mlp(z)
    print(f"Fusion MLP output shape: {logits.shape}")

# Test C2 attention pooling
model_c2.eval()
with torch.no_grad():
    image_features_c2 = model_c2.backbone(x_flat)
    image_features_c2 = image_features_c2.view(B, T, model_c2.feature_dim)
    
    f_img_attention = model_c2._pool_image_features(image_features_c2)
    print(f"C2 (attention) pooled features shape: {f_img_attention.shape}")
    
    # Verify attention weights sum to 1
    attention_hidden = model_c2.attention_W(image_features_c2) + model_c2.attention_b
    attention_hidden = torch.tanh(attention_hidden)
    attention_scores = torch.matmul(
        attention_hidden,
        model_c2.attention_v.unsqueeze(-1)
    ).squeeze(-1)
    attention_weights = torch.softmax(attention_scores, dim=1)
    print(f"Attention weights sum: {attention_weights.sum(dim=1)} (should be ~1.0)")

print("\n" + "=" * 70)
print("Step 8: Testing with different batch sizes")
print("=" * 70)

# Test batch size 1
single_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
imgs_single, _, patient_ids_single = next(iter(single_loader))
# Get clinical feature for single patient
pid_single = patient_ids_single[0]
try:
    pid_single_int = int(pid_single)
except ValueError:
    pid_single_int = pid_single

if pid_single_int in df.index:
    idx_single = df.index.get_loc(pid_single_int)
elif pid_single in df.index:
    idx_single = df.index.get_loc(pid_single)
else:
    raise ValueError(f"Patient {pid_single} not found")

clinical_single = torch.tensor(
    [X_clinical[idx_single]],
    dtype=torch.float32
)

with torch.no_grad():
    output_c1_single = model_c1(imgs_single, clinical_single)
    output_c2_single = model_c2(imgs_single, clinical_single)

print(f"Batch size 1 - C1: {output_c1_single.shape}, C2: {output_c2_single.shape}")

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)

