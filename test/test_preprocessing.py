"""Test script for clinical data preprocessing functionality."""

from pathlib import Path

import numpy as np

from data.clinical_data import (
    ClinicalConfig,
    ClinicalPreprocessor,
    get_feature_tensor_for_patient,
    load_clinical_table,
)

# Load the 5-variable clinical data
print("=" * 70)
print("Step 1: Loading clinical data")
print("=" * 70)

config = ClinicalConfig(
    csv_path=Path("data/annotations/175_clinical_5_variables.csv"),
    feature_columns=["AST", "ALT", "PLT", "APRI", "FIB_4"],
    patient_id_column="NO",
    label_column="CL_F2",
    fibrosis_stage_column=None,
)

df = load_clinical_table(config)
print(f"✓ Loaded {len(df)} patients")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# Create and fit preprocessor
print("\n" + "=" * 70)
print("Step 2: Creating and fitting preprocessor")
print("=" * 70)

# All features in this dataset are numeric
preprocessor = ClinicalPreprocessor(
    numeric_features=["AST", "ALT", "PLT", "APRI", "FIB_4"],
    categorical_features=[],  # No categorical features in this dataset
)

preprocessor.fit(df)
print("✓ Preprocessor fitted successfully")
print(f"\nNumeric feature statistics:")
for feature in preprocessor.numeric_features:
    print(f"  {feature}: mean={preprocessor.numeric_means[feature]:.4f}, "
          f"std={preprocessor.numeric_stds[feature]:.4f}")

print(f"\nFeature names after encoding: {preprocessor.feature_names}")

# Transform the data
print("\n" + "=" * 70)
print("Step 3: Transforming data")
print("=" * 70)

X, y_binary, fibrosis_stage = preprocessor.transform(df, label_column="CL_F2")
print(f"✓ Transformed data successfully")
print(f"X shape: {X.shape} (N_patients={X.shape[0]}, D_features={X.shape[1]})")
print(f"y_binary shape: {y_binary.shape}")
print(f"fibrosis_stage shape: {fibrosis_stage.shape}")
print(f"\nX statistics (first 5 patients, first 3 features):")
print(X[:5, :3])
print(f"\nLabel distribution:")
unique, counts = np.unique(y_binary, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count} patients")

# Test helper function for single patient
print("\n" + "=" * 70)
print("Step 4: Testing single patient feature extraction")
print("=" * 70)

patient_id = 1
feature_tensor = get_feature_tensor_for_patient(patient_id, df, preprocessor)
print(f"✓ Extracted features for patient {patient_id}")
print(f"Feature tensor shape: {feature_tensor.shape}")
print(f"Feature tensor dtype: {feature_tensor.dtype}")
print(f"Feature tensor values: {feature_tensor}")

# Example with categorical features (using full clinical CSV)
print("\n" + "=" * 70)
print("Step 5: Example with categorical features")
print("=" * 70)

try:
    config_full = ClinicalConfig(
        csv_path=Path("data/annotations/175_clinical.csv"),
        feature_columns=["AGE", "Sex", "BMI", "AST", "ALT", "PLT", "APRI", "FIB_4"],
        patient_id_column="NO",
        label_column="CL_F2",
        fibrosis_stage_column=None,
    )
    
    df_full = load_clinical_table(config_full)
    print(f"✓ Loaded full clinical data with {len(df_full)} patients")
    print(f"Columns: {list(df_full.columns)}")
    
    # Check Sex column values
    print(f"\nSex column unique values: {sorted(df_full['Sex'].dropna().unique())}")
    
    # Create preprocessor with categorical features
    preprocessor_full = ClinicalPreprocessor(
        numeric_features=["AGE", "BMI", "AST", "ALT", "PLT", "APRI", "FIB_4"],
        categorical_features=["Sex"],  # Sex is categorical
    )
    
    preprocessor_full.fit(df_full)
    print("✓ Preprocessor fitted with categorical features")
    print(f"\nCategorical mappings:")
    for feature, mapping in preprocessor_full.categorical_mappings.items():
        print(f"  {feature}: {mapping}")
    
    print(f"\nFeature names: {preprocessor_full.feature_names}")
    print(f"Total feature dimension: {len(preprocessor_full.feature_names)}")
    
    # Transform
    X_full, y_full, _ = preprocessor_full.transform(df_full, label_column="CL_F2")
    print(f"\n✓ Transformed full data")
    print(f"X_full shape: {X_full.shape}")
    print(f"First patient features (showing first 10): {X_full[0, :10]}")
    
except FileNotFoundError:
    print("Full clinical CSV not found, skipping categorical example")

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)

