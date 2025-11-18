"""Training script for A1/A2: B-mode CNN + Mean/Attention Pooling.

This script implements 5-fold cross-validation training for the B-mode image
classification models (A1: mean pooling, A2: attention pooling) with early
stopping based on validation AUC.

Usage:
    python train_bmode_a1.py --model_type mean      # A1: Mean pooling
    python train_bmode_a1.py --model_type attention # A2: Attention pooling
"""

import os
# Set CuBLAS workspace config for deterministic behavior (must be before torch import)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import copy
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from data.clinical_data import ClinicalConfig, load_clinical_table
from data.datasets import BModePatientDataset, get_eval_transform, get_train_transform
from data.image_index import PatientRecord, build_patient_records
from models.bmode_models import (
    create_bmode_attention_model,
    create_bmode_mean_model,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train B-mode CNN models (A1: mean pooling, A2: attention pooling)"
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['mean', 'attention'],
        default='mean',
        help='Model type: "mean" for A1 (mean pooling) or "attention" for A2 (attention pooling). Default: mean'
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
        help='Batch size for training. Default: 8'
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
        help='Maximum number of epochs. Default: 100'
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
        help='Hidden dimension for attention network (only for attention model). Default: 128'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility. Default: 42'
    )
    parser.add_argument(
        '--no_cv',
        action='store_true',
        help='Disable cross-validation and use a single train/val split (80/20). Default: False (use 5-fold CV)'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Validation split ratio when --no_cv is used. Default: 0.2 (20%%)'
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
        'model_type': args.model_type,
        'attention_hidden': args.attention_hidden,
        'seed': args.seed,
        'no_cv': args.no_cv,
        'val_split': args.val_split,
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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch.
    
    Args:
        model: Model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.
    
    Returns:
        Tuple of (average_loss, average_accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for imgs, labels, _ in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(imgs)
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
    device: torch.device
) -> Tuple[float, float, float]:
    """Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate.
        dataloader: Data loader.
        criterion: Loss function.
        device: Device to run on.
    
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
        for imgs, labels, _ in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(imgs)
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
    config: dict
) -> float:
    """Train model for one fold.
    
    Args:
        fold: Fold number (for logging).
        train_records: Training patient records.
        val_records: Validation patient records.
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
    
    # Create model based on model_type
    device = torch.device(config['device'])
    model_type = config['model_type']
    
    if model_type == 'mean':
        # A1: Mean pooling model
        model = create_bmode_mean_model(
            backbone=config['backbone'],
            pretrained=config['pretrained']
        ).to(device)
        logger.info(f"Created A1 model (mean pooling) with backbone {config['backbone']}")
    elif model_type == 'attention':
        # A2: Attention pooling model
        # Feature dimension will be auto-detected from backbone
        model = create_bmode_attention_model(
            backbone=config['backbone'],
            pretrained=config['pretrained'],
            feature_dim=512,  # Will be adjusted automatically if backbone has different dim
            attention_hidden=config['attention_hidden']
        ).to(device)
        logger.info(
            f"Created A2 model (attention pooling) with backbone {config['backbone']}, "
            f"attention_hidden={config['attention_hidden']}"
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'mean' or 'attention'.")
    
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
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, device
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
            best_model_state = copy.deepcopy(model.state_dict())
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
            model, val_loader, criterion, device
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
            model, val_loader, criterion, device
        )
        logger.info(
            f"\nFinal Val Results - Loss: {val_loss:.4f}, "
            f"Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}"
        )
        return val_auc


def main():
    """Main training function with optional cross-validation."""
    # Parse command line arguments
    args = parse_args()
    config = get_config(args)

    # Ensure reproducibility
    set_global_seed(config['seed'])
    
    # Determine model name
    model_name = "A1 (Mean Pooling)" if config['model_type'] == 'mean' else "A2 (Attention Pooling)"
    
    if config['no_cv']:
        logger.info(f"Starting single train/val split training for {model_name}")
    else:
        logger.info(f"Starting 5-fold cross-validation training for {model_name}")
    logger.info(f"Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Load clinical data
    logger.info("\nLoading clinical data...")
    clinical_config = ClinicalConfig(
        csv_path=config['clinical_csv'],
        feature_columns=["AST", "ALT", "PLT", "APRI", "FIB_4"],
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
    
    # Prepare for splitting
    labels = np.array([r.label_binary for r in patient_records])
    patient_indices = np.arange(len(patient_records))
    
    if config['no_cv']:
        # Single train/val split
        train_idx, val_idx = train_test_split(
            patient_indices,
            test_size=config['val_split'],
            stratify=labels,
            random_state=config['seed']
        )
        
        # Split records
        train_records, val_records = split_patients_stratified(
            patient_records, train_idx, val_idx
        )
        
        logger.info(f"\nTrain/Val Split:")
        logger.info(f"  Train: {len(train_records)} patients ({len(train_records)/len(patient_records)*100:.1f}%)")
        logger.info(f"  Val:   {len(val_records)} patients ({len(val_records)/len(patient_records)*100:.1f}%)")
        
        # Train single model
        val_auc = train_fold(0, train_records, val_records, config)
        
        # Print results
        logger.info(f"\n{'='*70}")
        logger.info(f"Training Results - {model_name}")
        logger.info(f"{'='*70}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Backbone: {config['backbone']}")
        logger.info(f"Val AUC: {val_auc:.4f}")
        logger.info(f"{'='*70}")
    else:
        # Stratified 5-fold CV with train/val split
        # For each fold: use 4 folds for train, 1 fold for val
        # This gives: 80% train, 20% val per fold
        skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=config['seed'])
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
            
            # Train fold
            val_auc = train_fold(fold, train_records, val_records, config)
            val_aucs.append(val_auc)
        
        # Print results
        logger.info(f"\n{'='*70}")
        logger.info(f"Cross-Validation Results - {model_name}")
        logger.info(f"{'='*70}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Backbone: {config['backbone']}")
        logger.info(f"Val AUCs: {[f'{auc:.4f}' for auc in val_aucs]}")
        logger.info(f"Mean Val AUC: {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
        logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()

