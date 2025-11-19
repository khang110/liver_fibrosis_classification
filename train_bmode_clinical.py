"""Training script for C1/C2: B-mode CNN + Clinical Fusion.

This script implements 5-fold cross-validation training for the B-mode + clinical
fusion models (C1: mean pooling, C2: attention pooling) with early stopping
based on validation AUC.

Usage:
    python train_bmode_clinical.py --pooling mean      # C1: Mean pooling
    python train_bmode_clinical.py --pooling attention # C2: Attention pooling
"""

import os
# Set CuBLAS workspace config for deterministic behavior (must be before torch import)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import copy
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from data.clinical_data import (
    ClinicalConfig,
    ClinicalPreprocessor,
    load_clinical_table,
)
from data.datasets import BModePatientDataset, get_eval_transform, get_train_transform
from data.image_index import PatientRecord, build_patient_records
from models.bmode_models import create_bmode_clinical_fusion_model, load_backbone

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BestThresholdMetrics = Dict[str, Any]


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
        default=12,
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
    parser.add_argument(
        '--log_dir',
        type=str,
        default='runs/bmode_clinical',
        help='Root directory for TensorBoard logs. Default: runs/bmode_clinical'
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
        'seed': args.seed,
        'no_cv': args.no_cv,
        'val_split': args.val_split,
        'log_dir': Path(args.log_dir),
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
    clinical_dict: Dict[str, torch.Tensor],
    return_confusion: bool = False
) -> Tuple[float, float, float, Optional[np.ndarray], Optional[BestThresholdMetrics]]:
    """Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate.
        dataloader: Data loader.
        criterion: Loss function.
        device: Device to run on.
        clinical_dict: Dictionary mapping patient_id to clinical feature tensor.
    
    Returns:
        Tuple of (average_loss, average_accuracy, auc_score, confusion_matrix?, best_threshold_metrics).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_labels = []
    all_predictions = []
    
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
            all_predictions.append(predictions.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = correct / total if total > 0 else 0.0
    
    # Calculate AUC
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    probabilities = torch.sigmoid(torch.from_numpy(all_logits)).numpy()
    auc = roc_auc_score(all_labels, probabilities)
    best_threshold_metrics = find_best_threshold(probabilities, all_labels)
    confusion = None
    if return_confusion and best_threshold_metrics is not None:
        confusion = best_threshold_metrics['confusion_matrix']
    
    return avg_loss, avg_acc, auc, confusion, best_threshold_metrics


def find_best_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> Optional[BestThresholdMetrics]:
    """Find the threshold that maximizes balanced accuracy and return metrics."""
    if probabilities.size == 0:
        return None
    
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    
    best_metrics = None
    best_balanced_acc = -1.0
    
    for thr in thresholds:
        preds = (probabilities >= thr).astype(int)
        balanced_acc = balanced_accuracy_score(labels, preds)
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            acc = (preds == labels).mean()
            precision = precision_score(labels, preds, zero_division=0)
            recall = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
            conf = confusion_matrix(labels, preds, labels=[0, 1])
            best_metrics = {
                'threshold': float(thr),
                'balanced_accuracy': float(balanced_acc),
                'accuracy': float(acc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'confusion_matrix': conf,
            }
    
    return best_metrics


def log_best_threshold_metrics(metrics: BestThresholdMetrics) -> None:
    """Log detailed information about the best decision threshold."""
    logger.info(
        "Best Threshold: %.2f | Balanced Acc: %.4f | Acc: %.4f | "
        "Precision: %.4f | Recall: %.4f | F1: %.4f",
        metrics['threshold'],
        metrics['balanced_accuracy'],
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
    )
    confusion = metrics.get('confusion_matrix')
    if confusion is not None:
        logger.info(
            "Confusion Matrix at Best Threshold (rows=actual [0,1], cols=predicted [0,1]):\n%s",
            confusion
        )


def log_best_threshold_scalars(
    writer: SummaryWriter,
    metrics: BestThresholdMetrics,
    global_step: int
) -> None:
    """Write best-threshold metrics to TensorBoard."""
    writer.add_scalar('BestThreshold/value', metrics['threshold'], global_step)
    writer.add_scalar('BalancedAccuracy/val_best', metrics['balanced_accuracy'], global_step)
    writer.add_scalar('Accuracy/val_best', metrics['accuracy'], global_step)
    writer.add_scalar('Precision/val_best', metrics['precision'], global_step)
    writer.add_scalar('Recall/val_best', metrics['recall'], global_step)
    writer.add_scalar('F1/val_best', metrics['f1'], global_step)


def create_confusion_matrix_figure(
    matrix: np.ndarray,
    class_names: Tuple[str, str] = ("Negative", "Positive")
) -> plt.Figure:
    """Create a matplotlib figure visualizing the confusion matrix."""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='Actual',
        xlabel='Predicted',
        title='Validation Confusion Matrix'
    )
    
    thresh = matrix.max() / 2 if matrix.max() > 0 else 0.5
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, int(matrix[i, j]),
                ha='center', va='center',
                color='white' if matrix[i, j] > thresh else 'black'
            )
    
    fig.tight_layout()
    return fig


def train(
    fold: int,
    train_records: List[PatientRecord],
    val_records: List[PatientRecord],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
    writer: Optional[SummaryWriter] = None
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
    if config['no_cv']:
        logger.info(f"{'='*20}Training single model{'='*20}")
    else:
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
    # Auto-detect feature dimension from backbone to avoid warnings
    _, backbone_feature_dim = load_backbone(config['backbone'], config['pretrained'])
    logger.info(f"Backbone feature dimension: {backbone_feature_dim}")
    
    model = create_bmode_clinical_fusion_model(
        backbone=config['backbone'],
        pretrained=config['pretrained'],
        feature_dim=backbone_feature_dim,  # Use correct dimension for backbone
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
    logger.info(f"Positive weight: {pos_weight.item():.4f}\n")
    
    # Training loop with early stopping
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    best_epoch = -1
    
    for epoch in range(config['num_epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, train_clinical_dict
        )
        
        # Validate
        val_loss, val_acc, val_auc, _, best_threshold_metrics = evaluate(
            model, val_loader, criterion, device, val_clinical_dict
        )
        
        logger.info(
            f"Epoch {epoch + 1}/{config['num_epochs']} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc (0.5): {val_acc:.4f}, Val AUC: {val_auc:.4f}"
        )
        if best_threshold_metrics is not None:
            logger.info(
                f"  Best Threshold {best_threshold_metrics['threshold']:.2f} → "
                f"Balanced Acc: {best_threshold_metrics['balanced_accuracy']:.4f}, "
                f"Acc: {best_threshold_metrics['accuracy']:.4f}, "
                f"Precision: {best_threshold_metrics['precision']:.4f}, "
                f"Recall: {best_threshold_metrics['recall']:.4f}, "
                f"F1: {best_threshold_metrics['f1']:.4f}"
            )

        if writer is not None:
            step = epoch + 1
            writer.add_scalar('Loss/train', train_loss, step)
            writer.add_scalar('Loss/val', val_loss, step)
            writer.add_scalar('Accuracy/train', train_acc, step)
            writer.add_scalar('Accuracy/val_threshold_0.5', val_acc, step)
            writer.add_scalar('AUC/val', val_auc, step)
            if best_threshold_metrics is not None:
                writer.add_scalar('Accuracy/val_best', best_threshold_metrics['accuracy'], step)
                writer.add_scalar('BalancedAccuracy/val_best', best_threshold_metrics['balanced_accuracy'], step)
                writer.add_scalar('Recall/val_best', best_threshold_metrics['recall'], step)
                writer.add_scalar('Precision/val_best', best_threshold_metrics['precision'], step)
                writer.add_scalar('F1/val_best', best_threshold_metrics['f1'], step)
                writer.add_scalar('BestThreshold/value', best_threshold_metrics['threshold'], step)
        
        # Early stopping check
        if val_auc > best_val_auc + config['early_stopping_min_delta']:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            logger.info(f"  → New best validation AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    if writer is not None:
        writer.add_scalar('Best/val_auc', best_val_auc, best_epoch if best_epoch != -1 else config['num_epochs'])
        writer.flush()
    
    # Load best model and re-evaluate to get final metrics
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with validation AUC: {best_val_auc:.4f}")
        
        # Re-evaluate best model to get final metrics
        val_loss, val_acc, val_auc, confusion, best_threshold_metrics = evaluate(
            model, val_loader, criterion, device, val_clinical_dict, return_confusion=True
        )
        
        logger.info(
            f"Final Val Results (Best Model) - Loss: {val_loss:.4f}, "
            f"Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}"
        )
        
        # Return the best validation AUC (from early stopping)
        # This is more reliable than re-evaluation which might have slight differences
        if best_threshold_metrics is not None:
            log_best_threshold_metrics(best_threshold_metrics)
            if writer is not None:
                log_best_threshold_scalars(
                    writer,
                    best_threshold_metrics,
                    best_epoch if best_epoch != -1 else config['num_epochs']
                )
        if confusion is not None and writer is not None:
            fig = create_confusion_matrix_figure(confusion)
            writer.add_figure(
                'ConfusionMatrix/val',
                fig,
                global_step=best_epoch if best_epoch != -1 else config['num_epochs']
            )
            plt.close(fig)
        return best_val_auc
    else:
        # If no best model was saved (shouldn't happen), use final evaluation
        val_loss, val_acc, val_auc, confusion, best_threshold_metrics = evaluate(
            model, val_loader, criterion, device, val_clinical_dict, return_confusion=True
        )
        logger.info(
            f"\nFinal Val Results - Loss: {val_loss:.4f}, "
            f"Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}"
        )
        if best_threshold_metrics is not None:
            log_best_threshold_metrics(best_threshold_metrics)
            if writer is not None:
                log_best_threshold_scalars(writer, best_threshold_metrics, config['num_epochs'])
        if confusion is not None and writer is not None:
            fig = create_confusion_matrix_figure(confusion)
            writer.add_figure(
                'ConfusionMatrix/val',
                fig,
                global_step=config['num_epochs']
            )
            plt.close(fig)
        return val_auc


def main():
    """Main training function with optional cross-validation."""
    # Parse command line arguments
    args = parse_args()
    config = get_config(args)

    # Ensure reproducibility
    set_global_seed(config['seed'])
    
    # Determine model name
    model_name = f"C1 (Mean Pooling)" if config['pooling'] == 'mean' else f"C2 (Attention Pooling)"
    
    config['log_dir'].mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = (
        model_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
    )
    
    if config['no_cv']:
        logger.info(f"Starting single train/val split training for {model_name}")
    else:
        logger.info(f"Starting 5-fold cross-validation training for {model_name}")
    logger.info(f"Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Load clinical data
    logger.info("Loading clinical data...")
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
    logger.info("Building patient records...")
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
        
        # Split clinical DataFrames
        train_patient_ids = [int(r.patient_id) for r in train_records]
        val_patient_ids = [int(r.patient_id) for r in val_records]
        
        train_df = df.loc[train_patient_ids]
        val_df = df.loc[val_patient_ids]
        
        logger.info(f"Train/Val Split:")
        logger.info(f"  Train: {len(train_records)} patients ({len(train_records)/len(patient_records)*100:.1f}%)")
        logger.info(f"  Val:   {len(val_records)} patients ({len(val_records)/len(patient_records)*100:.1f}%)")
        
        # Train single model
        run_name = f"{model_slug}_single_{config['backbone']}_{timestamp}"
        writer = SummaryWriter(log_dir=str(config['log_dir'] / run_name))
        try:
            val_auc = train(
                0, train_records, val_records,
                train_df, val_df, config, writer
            )
        finally:
            writer.close()
        
        # Print results
        logger.info(f"{'='*70}\n")
        logger.info(f"Training Results - {model_name}")
        logger.info(f"{'='*70}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Backbone: {config['backbone']}")
        logger.info(f"Pooling: {config['pooling']}")
        logger.info(f"Val AUC: {val_auc:.4f}")
        logger.info(f"{'='*70}")
    else:
        # Stratified 5-fold CV with train/val split
        # For each fold: use 4 folds for train, 1 fold for val
        # This gives: 80% train, 20% val per fold
        skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=config['seed'])
        folds = list(skf.split(patient_indices, labels))
        
        val_aucs = []
        run_base = f"{model_slug}_cv_{config['backbone']}_{timestamp}"
        
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
            fold_run_name = f"{run_base}_fold{fold + 1}"
            writer = SummaryWriter(log_dir=str(config['log_dir'] / fold_run_name))
            try:
                val_auc = train(
                    fold, train_records, val_records,
                    train_df, val_df, config, writer
                )
            finally:
                writer.close()
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

