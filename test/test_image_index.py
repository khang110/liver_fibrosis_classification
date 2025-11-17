"""Test script for image indexing functionality."""

import logging
from pathlib import Path

from data.clinical_data import ClinicalConfig, load_clinical_table
from data.image_index import build_patient_records, PatientRecord

# Set up logging to see debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

print("=" * 70)
print("Step 1: Loading clinical data")
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
print(f"Sample patient IDs: {list(df.index[:5])}")

print("\n" + "=" * 70)
print("Step 2: Building patient records with B-mode images")
print("=" * 70)

# Build patient records
image_root = Path("data/bmode_full")

try:
    records = build_patient_records(
        clinical_df=df,
        image_root=image_root,
        label_column="CL_F2",
        fibrosis_stage_column=None,
        image_pattern="Bmode_image_*.png",  # Match actual pattern
        required_images=3
    )
    
    print(f"\n✓ Successfully created {len(records)} patient records")
    print(f"Coverage: {len(records)}/{len(df)} patients ({100*len(records)/len(df):.1f}%)")
    
except Exception as e:
    print(f"✗ Error building records: {e}")
    raise

print("\n" + "=" * 70)
print("Step 3: Inspecting sample patient records")
print("=" * 70)

# Show details for first few patients
for i, record in enumerate(records[:5]):
    print(f"\nPatient {record.patient_id}:")
    print(f"  Label: {record.label_binary} ({'F2-4' if record.label_binary == 1 else 'F0-1'})")
    print(f"  Fibrosis stage: {record.fibrosis_stage}")
    print(f"  Images ({len(record.bmode_paths)}):")
    for j, img_path in enumerate(record.bmode_paths, 1):
        print(f"    {j}. {img_path.name} ({img_path.exists()})")

print("\n" + "=" * 70)
print("Step 4: Statistics")
print("=" * 70)

# Count labels
label_counts = {}
for record in records:
    label = record.label_binary
    label_counts[label] = label_counts.get(label, 0) + 1

print("Label distribution:")
for label, count in sorted(label_counts.items()):
    print(f"  Class {label} ({'F2-4' if label == 1 else 'F0-1'}): {count} patients")

# Verify all images exist
all_exist = all(img.exists() for record in records for img in record.bmode_paths)
print(f"\nAll images exist: {all_exist}")

# Check for any issues
issues = []
for record in records:
    if len(record.bmode_paths) != 3:
        issues.append(f"Patient {record.patient_id}: {len(record.bmode_paths)} images")
    for img in record.bmode_paths:
        if not img.exists():
            issues.append(f"Patient {record.patient_id}: missing {img.name}")

if issues:
    print(f"\n⚠ Found {len(issues)} issues:")
    for issue in issues[:10]:
        print(f"  {issue}")
else:
    print("\n✓ No issues found - all records are valid!")

print("\n" + "=" * 70)
print("Step 5: Testing custom image finder (example)")
print("=" * 70)

# Example of using a custom finder function
def custom_image_finder(patient_dir: Path):
    """Custom function to find images - could filter, sort differently, etc."""
    images = sorted(patient_dir.glob("Bmode_image_*.png"))
    # Could add custom logic here, e.g., filter by size, check metadata, etc.
    return images

# Test with a single patient
test_patient_id = records[0].patient_id
test_patient_dir = image_root / test_patient_id
custom_images = custom_image_finder(test_patient_dir)
print(f"Custom finder for patient {test_patient_id}: found {len(custom_images)} images")
print(f"Images: {[img.name for img in custom_images]}")

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)

