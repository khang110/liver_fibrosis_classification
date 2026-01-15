"""
Visualization script to compare fold distributions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
CLINICAL_CSV = "data/annotations/175_clinical_5_variables.csv"
PATIENT_ID_COL = "NO"
LABEL_COL = "CL_F2"
CLINICAL_FEATURES = ['AST', 'ALT', 'PLT']
N_FOLDS = 5
SEED = 42
OUTPUT_DIR = Path("logs/fold_debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Creating visualization plots...")

# Load data
df = pd.read_csv(CLINICAL_CSV)
patient_ids = df[PATIENT_ID_COL].values
labels = df[LABEL_COL].values
clinical_data = df[CLINICAL_FEATURES].values

# Create stratified K-Fold splits
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Create comprehensive plot
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Clinical Features Distribution: Problem Folds (4,5) vs Good Folds (1,2,3)', 
             fontsize=16, fontweight='bold')

colors = {
    1: '#2ecc71',  # Green - good
    2: '#3498db',  # Blue - good
    3: '#9b59b6',  # Purple - good
    4: '#e74c3c',  # Red - problem
    5: '#e67e22',  # Orange - problem
}

# Plot each feature
for feat_idx, feat in enumerate(CLINICAL_FEATURES):
    # Validation set distributions (left column)
    ax_val = axes[feat_idx, 0]
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels), 1):
        val_values = clinical_data[val_idx, feat_idx]
        
        # Use different line styles for problem folds
        linestyle = '--' if fold_idx in [4, 5] else '-'
        linewidth = 3 if fold_idx in [4, 5] else 1.5
        
        ax_val.hist(val_values, bins=20, alpha=0.3, color=colors[fold_idx], 
                   label=f'Fold {fold_idx}', edgecolor='black')
    
    ax_val.set_xlabel(f'{feat} Value', fontsize=11)
    ax_val.set_ylabel('Frequency', fontsize=11)
    ax_val.set_title(f'{feat} - Validation Sets Distribution', fontsize=12, fontweight='bold')
    ax_val.legend(loc='upper right', fontsize=9)
    ax_val.grid(True, alpha=0.3)
    
    # Box plots showing outliers (right column)
    ax_box = axes[feat_idx, 1]
    
    val_data_by_fold = []
    fold_labels = []
    boxcolors = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels), 1):
        val_values = clinical_data[val_idx, feat_idx]
        val_data_by_fold.append(val_values)
        fold_labels.append(f'F{fold_idx}')
        boxcolors.append(colors[fold_idx])
    
    bp = ax_box.boxplot(val_data_by_fold, labels=fold_labels, patch_artist=True,
                        showmeans=True, meanline=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], boxcolors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    # Highlight problem folds
    for i in [3, 4]:  # Fold 4 and 5 (0-indexed)
        bp['boxes'][i].set_linewidth(3)
        bp['boxes'][i].set_edgecolor('red')
    
    ax_box.set_xlabel('Fold', fontsize=11)
    ax_box.set_ylabel(f'{feat} Value', fontsize=11)
    ax_box.set_title(f'{feat} - Box Plot with Outliers', fontsize=12, fontweight='bold')
    ax_box.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = OUTPUT_DIR / "fold_comparison_distributions.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# Create second plot: Class-wise distributions
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle('Clinical Features by Class: Fold 4 vs Fold 5 vs Overall', 
             fontsize=16, fontweight='bold')

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels), 1):
    if fold_idx not in [4, 5]:
        continue
    
    col_offset = 0 if fold_idx == 4 else 1
    
    val_df = df.iloc[val_idx]
    
    for feat_idx, feat in enumerate(CLINICAL_FEATURES):
        ax = axes[feat_idx, col_offset]
        
        class0_data = val_df[val_df[LABEL_COL] == 0][feat].values
        class1_data = val_df[val_df[LABEL_COL] == 1][feat].values
        
        ax.hist(class0_data, bins=15, alpha=0.6, color='blue', label='Class 0 (No F2)', edgecolor='black')
        ax.hist(class1_data, bins=15, alpha=0.6, color='red', label='Class 1 (F2+)', edgecolor='black')
        
        ax.set_xlabel(f'{feat} Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Fold {fold_idx} - {feat}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

# Overall distribution in third column
for feat_idx, feat in enumerate(CLINICAL_FEATURES):
    ax = axes[feat_idx, 2]
    
    class0_data = df[df[LABEL_COL] == 0][feat].values
    class1_data = df[df[LABEL_COL] == 1][feat].values
    
    ax.hist(class0_data, bins=15, alpha=0.6, color='blue', label='Class 0 (No F2)', edgecolor='black')
    ax.hist(class1_data, bins=15, alpha=0.6, color='red', label='Class 1 (F2+)', edgecolor='black')
    
    ax.set_xlabel(f'{feat} Value', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'Overall - {feat}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = OUTPUT_DIR / "fold_class_distributions.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# Create summary statistics table plot
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Fold', 'Val Size', 'Class 0', 'Class 1', 
                   'AST Mean±SD', 'ALT Mean±SD', 'PLT Mean±SD', 
                   'AUC (from log)'])

# AUCs from training log
aucs = {1: 0.8712, 2: 0.7424, 3: 0.7935, 4: 0.6304, 5: 0.6377}

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels), 1):
    val_labels_fold = labels[val_idx]
    val_clinical_fold = clinical_data[val_idx]
    
    class0_count = sum(val_labels_fold == 0)
    class1_count = sum(val_labels_fold == 1)
    
    ast_mean = np.mean(val_clinical_fold[:, 0])
    ast_std = np.std(val_clinical_fold[:, 0])
    alt_mean = np.mean(val_clinical_fold[:, 1])
    alt_std = np.std(val_clinical_fold[:, 1])
    plt_mean = np.mean(val_clinical_fold[:, 2])
    plt_std = np.std(val_clinical_fold[:, 2])
    
    auc = aucs[fold_idx]
    
    row = [
        f'Fold {fold_idx}',
        f'{len(val_idx)}',
        f'{class0_count}',
        f'{class1_count}',
        f'{ast_mean:.1f}±{ast_std:.1f}',
        f'{alt_mean:.1f}±{alt_std:.1f}',
        f'{plt_mean:.1f}±{plt_std:.1f}',
        f'{auc:.4f}'
    ]
    
    table_data.append(row)

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.08, 0.08, 0.08, 0.08, 0.15, 0.15, 0.15, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(len(table_data[0])):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white')

# Highlight problem folds
for row_idx in [4, 5]:  # Fold 4 and 5
    for col_idx in range(len(table_data[0])):
        cell = table[(row_idx, col_idx)]
        cell.set_facecolor('#ffcccc')
        cell.set_edgecolor('red')
        cell.set_linewidth(2)

ax.set_title('Validation Set Summary Statistics by Fold', 
             fontsize=14, fontweight='bold', pad=20)

output_path = OUTPUT_DIR / "fold_summary_table.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}/")
print(f"  - fold_comparison_distributions.png")
print(f"  - fold_class_distributions.png")
print(f"  - fold_summary_table.png")
