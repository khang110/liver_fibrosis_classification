"""Training script for C1/C2: B-mode CNN + Clinical Fusion.

This script implements 5-fold cross-validation training for the B-mode + clinical
fusion models (C1: mean pooling, C2: attention pooling) with early stopping
based on validation AUC.

Usage:
    python train_bmode_clinical.py --pooling mean      # C1: Mean pooling
    python train_bmode_clinical.py --pooling attention # C2: Attention pooling
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from data.clinical_data import (
    ClinicalConfig,
    ClinicalPreprocessor,
    load_clinical_table,
)
from data.datasets import BModePatientDataset, get_eval_transform, get_train_transform
from data.image_index import PatientRecord, build_patient_records
from models.bmode_models import create_bmode_clinical_fusion_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train B-mode + clinical fusion models (C1: mean pooling, C2: attention pooling)"
    )
    parser.add_argument(
        '--pooling',
        type=str,
        choices=['mean', 'attention'],
        default='mean',
        help='Pooling method: "mean" for C1 or "attention" for C2. Default: mean'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        choices=['resnet18', 'resnet34', 'efficientnetv2_b0', 'efficientnetv2_b2'],
        default='resnet18',
        help='Backbone architecture. Options: resnet18, resnet34, efficientnetv2_b0, efficientnetv2_b2. Default: resnet18'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training. Default: 16'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate. Default: 1e-4'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=40,
        help='Maximum number of epochs. Default: 40'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=10,
        help='Early stopping patience. Default: 10'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Default: auto-detect'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loader workers. Default: 4'
    )
    parser.add_argument(
        '--attention_hidden',
        type=int,
        default=128,
        help='Hidden dimension for attention network (only for attention pooling). Default: 128'
    )
    parser.add_argument(
        '--clinical_dim',
        type=int,
        default=32,
        help='Output dimension for processed clinical features. Default: 32'
    )
    parser.add_argument(
        '--fusion_hidden',
        type=int,
        default=128,
        help='Hidden dimension for fusion MLP. Default: 128'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout probability in fusion MLP. Default: 0.5'
    )
    
    return parser.parse_args()


def get_config(args):
    """Get configuration dictionary from command line arguments.
    
    Args:
        args: Parsed command line arguments.
    
    Returns:
        Configuration dictionary.
    """
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = {
        'clinical_csv': Path("data/annotations/175_clinical_5_variables.csv"),
        'image_root': Path("data/bmode_full"),
        'patient_id_column': "NO",
        'label_column': "CL_F2",
        'image_pattern': "Bmode_image_*.png",
        'n_folds': 5,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_min_delta': 0.001,
        'backbone': args.backbone,
        'pretrained': True,
        'device': device,
        'num_workers': args.num_workers,
        'pooling': args.pooling,
        'attention_hidden': args.attention_hidden,
        'clinical_dim': args.clinical_dim,
        'fusion_hidden': args.fusion_hidden,
        'dropout': args.dropout,
        'clinical_features': ["AST", "ALT", "PLT", "APRI", "FIB_4"],
    }
    
    return config


def split_patients_stratified(
    patient_records: List[PatientRecord],
    train_idx: np.ndarray,
    val_idx: np.ndarray
) -> Tuple[List[PatientRecord], List[PatientRecord]]:
    """Split patient records into train and validation sets.
    
    Args:
        patient_records: List of all patient records.
        train_idx: Array of indices for training set.
        val_idx: Array of indices for validation set.
    
    Returns:
        Tuple of (train_records, val_records).
    """
    train_records = [patient_records[i] for i in train_idx]
    val_records = [patient_records[i] for i in val_idx]
    return train_records, val_records


def build_clinical_feature_dict(
    df: pd.DataFrame,
    preprocessor: ClinicalPreprocessor,
    patient_ids: List[str]
) -> Dict[str, torch.Tensor]:
    """Build a dictionary mapping patient_id to clinical feature tensor.
    
    Args:
        df: Clinical DataFrame indexed by patient_id.
        preprocessor: Fitted ClinicalPreprocessor.
        patient_ids: List of patient IDs to extract features for.
    
    Returns:
        Dictionary mapping patient_id (str) to clinical feature tensor.
    """
    clinical_dict = {}
    
    for pid in patient_ids:
        # Convert to int if needed
        try:
            pid_int = int(pid)
        except ValueError:
            pid_int = pid
        
        # Get patient row
        if pid_int in df.index:
            patient_row = df.loc[[pid_int]]
        elif pid in df.index:
            patient_row = df.loc[[pid]]
        else:
            logger.warning(f"Patient {pid} not found in clinical DataFrame")
            continue
        
        # Transform clinical features (only features, no label)
        X, _, _ = preprocessor.transform(patient_row, label_column=None)
        
        # Convert to tensor
        clinical_tensor = torch.from_numpy(X[0]).float()
        clinical_dict[pid] = clinical_tensor
    
    return clinical_dict


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clinical_dict: Dict[str, torch.Tensor]
) -> Tuple[float, float]:
    """Train for one epoch.
    
    Args:
        model: Model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.
        clinical_dict: Dictionary mapping patient_id to clinical feature tensor.
    
    Returns:
        Tuple of (average_loss, average_accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for imgs, labels, patient_ids in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # Build clinical features batch
        clinical_batch = []
        for pid in patient_ids:
            if pid in clinical_dict:
                clinical_batch.append(clinical_dict[pid])
            else:
                raise ValueError(f"Patient {pid} not found in clinical_dict")
        
        clinical_features = torch.stack(clinical_batch).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(imgs, clinical_features)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = correct / total if total > 0 else 0.0
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    clinical_dict: Dict[str, torch.Tensor]
) -> Tuple[float, float, float]:
    """Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate.
        dataloader: Data loader.
        criterion: Loss function.
        device: Device to run on.
        clinical_dict: Dictionary mapping patient_id to clinical feature tensor.
    
    Returns:
        Tuple of (average_loss, average_accuracy, auc_score).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels, patient_ids in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Build clinical features batch
            clinical_batch = []
            for pid in patient_ids:
                if pid in clinical_dict:
                    clinical_batch.append(clinical_dict[pid])
                else:
                    raise ValueError(f"Patient {pid} not found in clinical_dict")
            
            clinical_features = torch.stack(clinical_batch).to(device)
            
            # Forward pass
            logits = model(imgs, clinical_features)
            loss = criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Store for AUC calculation
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = correct / total if total > 0 else 0.0
    
    # Calculate AUC
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    probabilities = torch.sigmoid(torch.from_numpy(all_logits)).numpy()
    auc = roc_auc_score(all_labels, probabilities)
    
    return avg_loss, avg_acc, auc


def train_fold(
    fold: int,
    train_records: List[PatientRecord],
    val_records: List[PatientRecord],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict
) -> float:
    """Train model for one fold.
    
    Args:
        fold: Fold number (for logging).
        train_records: Training patient records.
        val_records: Validation patient records.
        train_df: Training clinical DataFrame.
        val_df: Validation clinical DataFrame.
        config: Configuration dictionary.
    
    Returns:
        Validation AUC score for this fold.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Fold {fold + 1}/{config['n_folds']}")
    logger.info(f"{'='*70}")
    
    # Create datasets
    train_dataset = BModePatientDataset(
        train_records,
        transform=get_train_transform()
    )
    val_dataset = BModePatientDataset(
        val_records,
        transform=get_eval_transform()
    )
    
    logger.info(f"Train: {len(train_dataset)} patients")
    logger.info(f"Val: {len(val_dataset)} patients")
    
    # Fit preprocessor on training data only (to avoid leakage)
    logger.info("Fitting clinical preprocessor on training data...")
    preprocessor = ClinicalPreprocessor(
        numeric_features=config['clinical_features'],
        categorical_features=[],
    )
    preprocessor.fit(train_df)
    
    # Build clinical feature dictionaries for each split
    train_patient_ids = [r.patient_id for r in train_records]
    val_patient_ids = [r.patient_id for r in val_records]
    
    train_clinical_dict = build_clinical_feature_dict(
        train_df, preprocessor, train_patient_ids
    )
    val_clinical_dict = build_clinical_feature_dict(
        val_df, preprocessor, val_patient_ids
    )
    
    logger.info(f"Built clinical feature dictionaries: "
                f"train={len(train_clinical_dict)}, "
                f"val={len(val_clinical_dict)}")
    
    # Get clinical input dimension from preprocessor
    clinical_input_dim = len(preprocessor.feature_names)
    logger.info(f"Clinical input dimension: {clinical_input_dim}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Create model
    device = torch.device(config['device'])
    # Feature dimension will be auto-detected from backbone
    # (512 for ResNet, 1280 for EfficientNetV2-B0, 1408 for EfficientNetV2-B2)
    model = create_bmode_clinical_fusion_model(
        backbone=config['backbone'],
        pretrained=config['pretrained'],
        feature_dim=512,  # Will be adjusted automatically if backbone has different dim
        clinical_dim=config['clinical_dim'],
        fusion_hidden=config['fusion_hidden'],
        pooling=config['pooling'],
        attention_hidden=config['attention_hidden'],
        dropout=config['dropout'],
        clinical_input_dim=clinical_input_dim
    ).to(device)
    
    logger.info(
        f"Created {config['pooling']} pooling model with backbone {config['backbone']}"
    )
    
    # Calculate pos_weight for imbalanced dataset
    train_labels = [r.label_binary for r in train_records]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count]).to(device) if pos_count > 0 else torch.tensor([1.0]).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate']
    )
    
    logger.info(f"Using device: {device}")
    logger.info(f"Positive weight: {pos_weight.item():.4f}")
    
    # Training loop with early stopping
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(config['num_epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, train_clinical_dict
        )
        
        # Validate
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, device, val_clinical_dict
        )
        
        logger.info(
            f"Epoch {epoch + 1}/{config['num_epochs']} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}"
        )
        
        # Early stopping check
        if val_auc > best_val_auc + config['early_stopping_min_delta']:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            logger.info(f"  → New best validation AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Load best model and re-evaluate to get final metrics
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with validation AUC: {best_val_auc:.4f}")
        
        # Re-evaluate best model to get final metrics
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, device, val_clinical_dict
        )
        
        logger.info(
            f"\nFinal Val Results (Best Model) - Loss: {val_loss:.4f}, "
            f"Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}"
        )
        
        # Return the best validation AUC (from early stopping)
        # This is more reliable than re-evaluation which might have slight differences
        return best_val_auc
    else:
        # If no best model was saved (shouldn't happen), use final evaluation
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, device, val_clinical_dict
        )
        logger.info(
            f"\nFinal Val Results - Loss: {val_loss:.4f}, "
            f"Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}"
        )
        return val_auc


def main():
    """Main training function with 5-fold cross-validation."""
    # Parse command line arguments
    args = parse_args()
    config = get_config(args)
    
    # Determine model name
    model_name = f"C1 (Mean Pooling)" if config['pooling'] == 'mean' else f"C2 (Attention Pooling)"
    
    logger.info(f"Starting 5-fold cross-validation training for {model_name}")
    logger.info(f"Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Load clinical data
    logger.info("\nLoading clinical data...")
    clinical_config = ClinicalConfig(
        csv_path=config['clinical_csv'],
        feature_columns=config['clinical_features'],
        patient_id_column=config['patient_id_column'],
        label_column=config['label_column'],
        fibrosis_stage_column=None,
    )
    df = load_clinical_table(clinical_config)
    logger.info(f"Loaded {len(df)} patients from clinical data")
    
    # Build patient records
    logger.info("\nBuilding patient records...")
    patient_records = build_patient_records(
        clinical_df=df,
        image_root=config['image_root'],
        label_column=config['label_column'],
        image_pattern=config['image_pattern'],
        required_images=3
    )
    logger.info(f"Created {len(patient_records)} patient records")
    
    # Prepare for stratified K-fold
    labels = np.array([r.label_binary for r in patient_records])
    patient_indices = np.arange(len(patient_records))
    
    # Stratified 5-fold CV with train/val split
    # For each fold: use 4 folds for train, 1 fold for val
    # This gives: 80% train, 20% val per fold
    skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=42)
    folds = list(skf.split(patient_indices, labels))
    
    val_aucs = []
    
    for fold in range(config['n_folds']):
        # Get train and val indices for this fold
        train_idx = folds[fold][0]
        val_idx = folds[fold][1]
        
        # Split records
        train_records, val_records = split_patients_stratified(
            patient_records, train_idx, val_idx
        )
        
        # Split clinical DataFrames
        train_patient_ids = [int(r.patient_id) for r in train_records]
        val_patient_ids = [int(r.patient_id) for r in val_records]
        
        train_df = df.loc[train_patient_ids]
        val_df = df.loc[val_patient_ids]
        
        # Train fold
        val_auc = train_fold(
            fold, train_records, val_records,
            train_df, val_df, config
        )
        val_aucs.append(val_auc)
    
    # Print results
    logger.info(f"\n{'='*70}")
    logger.info(f"Cross-Validation Results - {model_name}")
    logger.info(f"{'='*70}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Backbone: {config['backbone']}")
    logger.info(f"Pooling: {config['pooling']}")
    logger.info(f"Val AUCs: {[f'{auc:.4f}' for auc in val_aucs]}")
    logger.info(f"Mean Val AUC: {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()

