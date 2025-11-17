"""Test script for B-mode patient dataset."""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.clinical_data import ClinicalConfig, load_clinical_table
from data.datasets import BModePatientDataset, get_train_transform, get_eval_transform
from data.image_index import build_patient_records

logging.basicConfig(level=logging.INFO)

print("=" * 70)
print("Step 1: Loading clinical data and building patient records")
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
print(f"✓ Loaded {len(df)} patients from clinical data")

# Build patient records
records = build_patient_records(
    clinical_df=df,
    image_root=Path("data/bmode_full"),
    label_column="CL_F2",
    image_pattern="Bmode_image_*.png",
    required_images=3
)
print(f"✓ Created {len(records)} patient records")

print("\n" + "=" * 70)
print("Step 2: Creating dataset with eval transform")
print("=" * 70)

# Create dataset with eval transform (no augmentation)
eval_dataset = BModePatientDataset(
    patient_records=records,
    transform=get_eval_transform()
)

print(f"✓ Created dataset with {len(eval_dataset)} patients")
print(f"Dataset length: {len(eval_dataset)}")

# Test getting a single item
print("\n" + "=" * 70)
print("Step 3: Testing single item retrieval")
print("=" * 70)

imgs, label, patient_id = eval_dataset[0]
print(f"✓ Retrieved sample for patient {patient_id}")
print(f"Images tensor shape: {imgs.shape} (expected: (3, 3, 224, 224))")
print(f"Label: {label.item()} ({'F2-4' if label.item() == 1.0 else 'F0-1'})")
print(f"Label dtype: {label.dtype}")
print(f"Patient ID type: {type(patient_id)}")

# Verify tensor properties
assert imgs.shape == (3, 3, 224, 224), f"Expected (3, 3, 224, 224), got {imgs.shape}"
assert label.dtype == torch.float32, f"Expected float32, got {label.dtype}"
assert label.item() in [0.0, 1.0], f"Label should be 0.0 or 1.0, got {label.item()}"
print("✓ All tensor properties correct!")

print("\n" + "=" * 70)
print("Step 4: Testing dataset with train transform")
print("=" * 70)

# Create dataset with train transform (with augmentation)
train_dataset = BModePatientDataset(
    patient_records=records[:10],  # Use subset for testing
    transform=get_train_transform()
)

print(f"✓ Created training dataset with {len(train_dataset)} patients")

# Get a few samples to verify augmentation works
for i in range(3):
    imgs, label, patient_id = train_dataset[i]
    print(f"  Patient {patient_id}: images shape {imgs.shape}, label {label.item()}")

print("\n" + "=" * 70)
print("Step 5: Testing DataLoader")
print("=" * 70)

# Create DataLoader
dataloader = DataLoader(
    eval_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0  # Set to 0 for testing, increase for actual training
)

print(f"✓ Created DataLoader with batch_size=4")
print(f"Number of batches: {len(dataloader)}")

# Get a batch
batch_imgs, batch_labels, batch_patient_ids = next(iter(dataloader))
print(f"\nBatch shapes:")
print(f"  Images: {batch_imgs.shape} (expected: (4, 3, 3, 224, 224))")
print(f"  Labels: {batch_labels.shape} (expected: (4,))")
print(f"  Patient IDs: {len(batch_patient_ids)} items")
print(f"  Patient IDs: {batch_patient_ids}")

assert batch_imgs.shape == (4, 3, 3, 224, 224), f"Expected (4, 3, 3, 224, 224), got {batch_imgs.shape}"
assert batch_labels.shape == (4,), f"Expected (4,), got {batch_labels.shape}"
print("✓ Batch shapes correct!")

print("\n" + "=" * 70)
print("Step 6: Testing with shuffled DataLoader")
print("=" * 70)

shuffled_loader = DataLoader(
    eval_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

batch_imgs, batch_labels, batch_patient_ids = next(iter(shuffled_loader))
print(f"✓ Shuffled batch - Patient IDs: {batch_patient_ids}")
print(f"  Labels: {batch_labels.tolist()}")

print("\n" + "=" * 70)
print("Step 7: Testing image value ranges after normalization")
print("=" * 70)

# Check that normalized images are in reasonable range
imgs, _, _ = eval_dataset[0]
print(f"Image tensor stats:")
print(f"  Min: {imgs.min().item():.4f}")
print(f"  Max: {imgs.max().item():.4f}")
print(f"  Mean: {imgs.mean().item():.4f}")
print(f"  Std: {imgs.std().item():.4f}")

# After ImageNet normalization, values should be roughly in [-3, 3] range
# (though can be outside due to different image statistics)
print("  (After ImageNet normalization, values typically in [-3, 3] range)")

print("\n" + "=" * 70)
print("Step 8: Testing label distribution")
print("=" * 70)

# Count labels in dataset
label_counts = {0: 0, 1: 0}
for i in range(len(eval_dataset)):
    _, label, _ = eval_dataset[i]
    label_counts[int(label.item())] += 1

print(f"Label distribution:")
print(f"  Class 0 (F0-1): {label_counts[0]} patients")
print(f"  Class 1 (F2-4): {label_counts[1]} patients")
print(f"  Total: {sum(label_counts.values())} patients")

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)

