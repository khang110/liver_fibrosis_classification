"""Test script for radiomics feature extraction."""

import logging
from pathlib import Path

import pandas as pd

from data.clinical_data import ClinicalConfig, load_clinical_table
from data.image_index import build_patient_records
from data.radiomics_features import build_patient_radiomics, extract_bmode_radiomics

logging.basicConfig(level=logging.INFO)

print("=" * 70)
print("Step 1: Loading data and building patient records")
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

print(f"✓ Created {len(records)} patient records")

print("\n" + "=" * 70)
print("Step 2: Testing single image radiomics extraction")
print("=" * 70)

# Test on a single image
test_image = records[0].bmode_paths[0]
print(f"Testing on image: {test_image}")

try:
    features = extract_bmode_radiomics(test_image, roi_mask=None)
    print(f"✓ Successfully extracted {len(features)} features")
    print("\nFeature values:")
    for feature_name, value in features.items():
        print(f"  {feature_name}: {value:.6f}")
except Exception as e:
    print(f"✗ Error: {e}")
    raise

print("\n" + "=" * 70)
print("Step 3: Building patient radiomics (first 10 patients)")
print("=" * 70)

# Test on a subset of patients
test_records = records[:10]
try:
    radiomics_df = build_patient_radiomics(test_records, roi_masks=None)
    print(f"✓ Successfully extracted radiomics for {len(radiomics_df)} patients")
    print(f"\nDataFrame shape: {radiomics_df.shape}")
    print(f"Index name: {radiomics_df.index.name}")
    print(f"\nFeature columns:")
    print(list(radiomics_df.columns))
    print(f"\nFirst few rows:")
    print(radiomics_df.head())
    
    # Check feature statistics
    print(f"\nFeature statistics:")
    print(radiomics_df.describe())
    
except Exception as e:
    print(f"✗ Error: {e}")
    raise

print("\n" + "=" * 70)
print("Step 4: Testing with all patients")
print("=" * 70)

try:
    all_radiomics_df = build_patient_radiomics(records, roi_masks=None)
    print(f"✓ Successfully extracted radiomics for {len(all_radiomics_df)} patients")
    print(f"Coverage: {len(all_radiomics_df)}/{len(records)} patients")
    print(f"\nFeature summary:")
    print(all_radiomics_df.describe())
    
except Exception as e:
    print(f"✗ Error: {e}")
    raise

print("\n" + "=" * 70)
print("Step 5: Verifying feature consistency")
print("=" * 70)

# Check that all patients have the same features
expected_features = [
    'intensity_mean', 'intensity_std', 'intensity_min', 'intensity_max',
    'intensity_skewness', 'intensity_kurtosis',
    'glcm_contrast', 'glcm_correlation', 'glcm_energy', 'glcm_homogeneity'
]

missing_features = [f for f in expected_features if f not in all_radiomics_df.columns]
if missing_features:
    print(f"✗ Missing features: {missing_features}")
else:
    print(f"✓ All expected features present: {len(expected_features)} features")

# Check for NaN values
nan_counts = all_radiomics_df.isna().sum()
if nan_counts.sum() > 0:
    print(f"⚠ Found NaN values:")
    print(nan_counts[nan_counts > 0])
else:
    print("✓ No NaN values found")

# Check feature ranges
print(f"\nFeature ranges:")
for feature in expected_features:
    min_val = all_radiomics_df[feature].min()
    max_val = all_radiomics_df[feature].max()
    print(f"  {feature}: [{min_val:.4f}, {max_val:.4f}]")

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)

