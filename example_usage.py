"""Example usage of the clinical data access layer.

This script demonstrates how to use the ClinicalConfig and load_clinical_table
functions to load clinical data from CSV files.
"""

from pathlib import Path

from data.clinical_data import ClinicalConfig, load_clinical_table

# Example 1: Using the 5-variable clinical CSV
# This matches the actual structure of 175_clinical_5_variables.csv
config_5var = ClinicalConfig(
    csv_path=Path("data/annotations/175_clinical_5_variables.csv"),
    feature_columns=["AST", "ALT", "PLT", "APRI", "FIB_4"],
    patient_id_column="NO",  # Patient ID column in the CSV
    label_column="CL_F2",   # Binary label: 0=F0-1, 1=F2-4
    fibrosis_stage_column=None,  # Not present in this CSV
)

# Load the data
print("=" * 60)
print("Example 1: Loading 5-variable clinical data")
print("=" * 60)
try:
    df_5var = load_clinical_table(config_5var)
    print("✓ Successfully loaded clinical data!")
    print(f"\nDataset shape: {df_5var.shape}")
    print(f"Index name: {df_5var.index.name}")
    print(f"Columns: {list(df_5var.columns)}")
    print(f"\nFirst 5 patients:")
    print(df_5var.head())
    print(f"\nLabel distribution (CL_F2):")
    print(df_5var["CL_F2"].value_counts().sort_index())
    print(f"\nAccess a specific patient:")
    print(f"Patient 1: {df_5var.loc[1]}")
except Exception as e:
    print(f"✗ Error loading data: {e}")

# Example 2: Using with different column names (for future use)
# This shows how you would adapt it to a CSV with standard column names
print("\n" + "=" * 60)
print("Example 2: Configuration for standard column names")
print("=" * 60)
print("""
# If your CSV has standard column names like:
# patient_id, label_binary, fibrosis_stage, age, sex, BMI, AST, ALT, ...

config_standard = ClinicalConfig(
    csv_path=Path("path/to/clinical_data.csv"),
    feature_columns=["age", "sex", "BMI", "AST", "ALT", "platelets", 
                     "albumin", "bilirubin"],
    patient_id_column="patient_id",      # Default
    label_column="label_binary",          # Default
    fibrosis_stage_column="fibrosis_stage",  # Default
)

df = load_clinical_table(config_standard)
""")

