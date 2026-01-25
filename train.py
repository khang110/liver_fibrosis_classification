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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import matplotlib.pyplot as plt

from data.clinical_data import ClinicalConfig, load_clinical_table
from data.datasets import BModePatientDataset, get_eval_transform, get_train_transform, get_train_transform_advanced
from data.image_index import PatientRecord, build_patient_records
from models.bmode_models import (
    create_bmode_attention_model,
    create_bmode_mean_model,
    load_backbone,
)

os.makedirs("logs", exist_ok=True)  # optional, create folder
os.makedirs("weights", exist_ok=True)
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_train.log", encoding="utf-8"),  # save to file
        logging.StreamHandler()                                   # print to console
    ]
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
    parser.add_argument(
        '--log_dir',
        type=str,
        default='runs/bmode',
        help='Root directory for TensorBoard logs. Default: runs/bmode'
    )
    parser.add_argument(
        '--use_advanced_aug',
        action='store_true',
        help='Use advanced data augmentation. Default: False (use basic augmentation)'
    )
    parser.add_argument(
        '--use_lr_scheduler',
        action='store_true',
        help='Use cosine annealing LR scheduler. Default: False'
    )
    parser.add_argument(
        '--use_tta',
        action='store_true',
        help='Use test-time augmentation during validation. Default: False'
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
        'clinical_csv': Path("data/annotations/175_clinical_11_variables_F3.csv"),
        'image_root': Path("data/nakagami_full"),
        'patient_id_column': "NO",
        'label_column': "CL_F3",
        'image_pattern': "Nakagami_image_*.png",  #Nakagami_image_ // Bmode_image_
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
        'log_dir': Path(args.log_dir),
        'use_advanced_aug': args.use_advanced_aug,
        'use_lr_scheduler': args.use_lr_scheduler,
        'use_tta': args.use_tta,
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
) -> Tuple[float, float, Optional[float], Optional[BestThresholdMetrics]]:
    """Train for one epoch.
    
    Args:
        model: Model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.
    
    Returns:
        Tuple of (average_loss, average_accuracy, train_auc, train_threshold_metrics).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    
    for imgs, labels, _ in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = correct / total if total > 0 else 0.0
    train_auc = None
    train_threshold_metrics = None
    if all_labels:
        logits_concat = np.concatenate(all_logits)
        labels_concat = np.concatenate(all_labels)
        try:
            probabilities = torch.softmax(torch.from_numpy(logits_concat), dim=1).numpy()[:, 1]
            train_auc = float(roc_auc_score(labels_concat, probabilities))
            train_threshold_metrics = find_best_threshold(probabilities, labels_concat)
        except ValueError:
            train_auc = None
            train_threshold_metrics = None
    return avg_loss, avg_acc, train_auc, train_threshold_metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_confusion: bool = False
) -> Tuple[float, float, float, Optional[np.ndarray], Optional[BestThresholdMetrics]]:
    """Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate.
        dataloader: Data loader.
        criterion: Loss function.
        device: Device to run on.
        return_confusion: Whether to return confusion matrix. Default: False.
    
    Returns:
        Tuple of (average_loss, average_accuracy, auc_score, confusion_matrix?, best_threshold_metrics).
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
            labels = labels.to(device).long()
            
            # Forward pass
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Store for AUC calculation
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = correct / total if total > 0 else 0.0
    
    # Calculate AUC using positive-class probability
    all_logits = np.concatenate(all_logits)  # (N, 2)
    all_labels = np.concatenate(all_labels)  # (N,)
    probabilities = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()[:, 1]
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
            
            # Calculate Sensitivity (Recall), Specificity, PPV (Precision), NPV
            # Confusion matrix: [[TN, FP], [FN, TP]]
            tn, fp, fn, tp = conf.ravel()
            
            # Sensitivity (Recall) = TP / (TP + FN)
            sensitivity = recall  # Same as recall
            
            # Specificity = TN / (TN + FP)
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            
            # PPV (Positive Predictive Value) = Precision = TP / (TP + FP)
            ppv = precision  # Same as precision
            
            # NPV (Negative Predictive Value) = TN / (TN + FN)
            npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
            
            best_metrics = {
                'threshold': float(thr),
                'balanced_accuracy': float(balanced_acc),
                'accuracy': float(acc),
                'precision': float(precision),
                'recall': float(recall),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'ppv': float(ppv),
                'npv': float(npv),
                'f1': float(f1),
                'confusion_matrix': conf,
            }
    
    return best_metrics


def log_best_threshold_metrics(metrics: BestThresholdMetrics, prefix: str = "") -> None:
    """Log detailed information about the best decision threshold."""
    prefix_str = f"{prefix} " if prefix else ""
    logger.info(
        "%sBest Threshold: %.2f | Balanced Acc: %.4f | Acc: %.4f | "
        "Precision: %.4f | Recall: %.4f | Sensitivity: %.4f | Specificity: %.4f | "
        "PPV: %.4f | NPV: %.4f | F1: %.4f",
        prefix_str,
        metrics['threshold'],
        metrics['balanced_accuracy'],
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['sensitivity'],
        metrics['specificity'],
        metrics['ppv'],
        metrics['npv'],
        metrics['f1'],
    )
    confusion = metrics.get('confusion_matrix')
    if confusion is not None:
        logger.info(
            "%sConfusion Matrix at Best Threshold (rows=actual [0,1], cols=predicted [0,1]):\n%s",
            prefix_str,
            confusion
        )


def log_best_threshold_scalars(
    writer: SummaryWriter,
    metrics: BestThresholdMetrics,
    global_step: int,
    prefix: str = 'val'
) -> None:
    """Write best-threshold metrics to TensorBoard."""
    writer.add_scalar(f'BestThreshold/{prefix}', metrics['threshold'], global_step)
    writer.add_scalar(f'BalancedAccuracy/{prefix}_best', metrics['balanced_accuracy'], global_step)
    writer.add_scalar(f'Accuracy/{prefix}_best', metrics['accuracy'], global_step)
    writer.add_scalar(f'Precision/{prefix}_best', metrics['precision'], global_step)
    writer.add_scalar(f'Recall/{prefix}_best', metrics['recall'], global_step)
    writer.add_scalar(f'Sensitivity/{prefix}_best', metrics['sensitivity'], global_step)
    writer.add_scalar(f'Specificity/{prefix}_best', metrics['specificity'], global_step)
    writer.add_scalar(f'PPV/{prefix}_best', metrics['ppv'], global_step)
    writer.add_scalar(f'NPV/{prefix}_best', metrics['npv'], global_step)
    writer.add_scalar(f'F1/{prefix}_best', metrics['f1'], global_step)


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
        title='Prediction'
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
    config: dict,
    writer: Optional[SummaryWriter] = None,
    run_name: Optional[str] = None
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
    if config['no_cv']:
        logger.info(f"{'='*20}Training single model{'='*20}")
    else:
        logger.info(f"\n{'='*70}")
        logger.info(f"Fold {fold + 1}/{config['n_folds']}")
        logger.info(f"{'='*70}")
    
    # Create datasets
    train_transform = get_train_transform_advanced() if config.get('use_advanced_aug', False) else get_train_transform()
    train_dataset = BModePatientDataset(
        train_records,
        transform=train_transform
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
        # Detect backbone feature dimension to avoid mismatches (e.g., EfficientNetV2)
        _, backbone_feature_dim = load_backbone(
            backbone=config['backbone'],
            pretrained=config['pretrained']
        )
        model = create_bmode_attention_model(
            backbone=config['backbone'],
            pretrained=config['pretrained'],
            feature_dim=backbone_feature_dim,
            attention_hidden=config['attention_hidden']
        ).to(device)
        logger.info(
            f"Created A2 model (attention pooling) with backbone {config['backbone']}, "
            f"attention_hidden={config['attention_hidden']}, feature_dim={backbone_feature_dim}"
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'mean' or 'attention'.")
    
    # Calculate class weights for imbalanced dataset
    train_labels = [r.label_binary for r in train_records]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    if pos_count > 0:
        class_weights = torch.tensor([1.0, neg_count / pos_count], dtype=torch.float32).to(device)
    else:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    
    # Loss and optimizer (multi-class cross-entropy with class weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate']
    )
    
    # Learning rate scheduler (optional)
    scheduler = None
    if config.get('use_lr_scheduler', False):
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=1e-6
        )
        logger.info("Using CosineAnnealingLR scheduler")
    
    logger.info(f"Using device: {device}")
    logger.info(f"Class weights: {class_weights.cpu().numpy().tolist()}")
    
    # Training loop with early stopping
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    best_epoch = -1
    
    for epoch in range(config['num_epochs']):
        # Train
        train_loss, train_acc, train_auc, train_threshold_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        train_auc_str = f"{train_auc:.4f}" if train_auc is not None else "nan"
        # Validate
        val_loss, val_acc, val_auc, _, val_threshold_metrics = evaluate(
            model, val_loader, criterion, device
        )
        
        logger.info(
            f"Epoch {epoch + 1}/{config['num_epochs']} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Train AUC: {train_auc_str} - "
            f"Val Loss: {val_loss:.4f}, Val Acc (0.5): {val_acc:.4f}, Val AUC: {val_auc:.4f}"
        )
        if train_threshold_metrics is not None:
            logger.info(
                f"  Train Best Threshold {train_threshold_metrics['threshold']:.2f} → "
                f"Balanced Acc: {train_threshold_metrics['balanced_accuracy']:.4f}, "
                f"Acc: {train_threshold_metrics['accuracy']:.4f}, "
                f"Sensitivity: {train_threshold_metrics['sensitivity']:.4f}, "
                f"Specificity: {train_threshold_metrics['specificity']:.4f}, "
                f"PPV: {train_threshold_metrics['ppv']:.4f}, "
                f"NPV: {train_threshold_metrics['npv']:.4f}, "
                f"F1: {train_threshold_metrics['f1']:.4f}"
            )
        if val_threshold_metrics is not None:
            logger.info(
                f"  Val Best Threshold {val_threshold_metrics['threshold']:.2f} → "
                f"Balanced Acc: {val_threshold_metrics['balanced_accuracy']:.4f}, "
                f"Acc: {val_threshold_metrics['accuracy']:.4f}, "
                f"Sensitivity: {val_threshold_metrics['sensitivity']:.4f}, "
                f"Specificity: {val_threshold_metrics['specificity']:.4f}, "
                f"PPV: {val_threshold_metrics['ppv']:.4f}, "
                f"NPV: {val_threshold_metrics['npv']:.4f}, "
                f"F1: {val_threshold_metrics['f1']:.4f}"
            )
        
        # Log to TensorBoard
        if writer is not None:
            step = epoch + 1
            writer.add_scalar('Loss/train', train_loss, step)
            writer.add_scalar('Loss/val', val_loss, step)
            writer.add_scalar('Accuracy/train', train_acc, step)
            writer.add_scalar('Accuracy/val_threshold_0.5', val_acc, step)
            writer.add_scalar('AUC/val', val_auc, step)
            if train_auc is not None:
                writer.add_scalar('AUC/train', train_auc, step)
            if train_threshold_metrics is not None:
                log_best_threshold_scalars(writer, train_threshold_metrics, step, prefix='train')
            if val_threshold_metrics is not None:
                log_best_threshold_scalars(writer, val_threshold_metrics, step, prefix='val')
        
        # Step scheduler after each epoch
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"  Current LR: {current_lr:.6f}")
        
        # Early stopping check
        if val_auc > best_val_auc + config['early_stopping_min_delta']:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            logger.info(f"  → New best validation AUC: {best_val_auc:.4f}")
            
            if run_name:
                save_path = Path("weights") / f"{run_name}_best.pth"
                torch.save(best_model_state, save_path)
                logger.info(f"  → Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Log best AUC to TensorBoard
    if writer is not None:
        writer.add_scalar('Best/val_auc', best_val_auc, best_epoch if best_epoch != -1 else config['num_epochs'])
        writer.flush()
    
    # Load best model and re-evaluate to get final metrics
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with validation AUC: {best_val_auc:.4f}")
        
        # Re-evaluate best model on both train and val sets
        train_loss, train_acc, train_auc, train_confusion, train_threshold_metrics = evaluate(
            model, train_loader, criterion, device, return_confusion=True
        )
        val_loss, val_acc, val_auc, val_confusion, val_threshold_metrics = evaluate(
            model, val_loader, criterion, device, return_confusion=True
        )
        
        logger.info(
            f"Final Train Results (Best Model) at epoch {best_epoch} - Loss: {train_loss:.4f}, "
            f"Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}"
        )
        logger.info(
            f"Final Val Results (Best Model) at epoch {best_epoch} - Loss: {val_loss:.4f}, "
            f"Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}"
        )
        
        # Log train metrics
        if train_threshold_metrics is not None:
            log_best_threshold_metrics(train_threshold_metrics, prefix="Train")
            if writer is not None:
                log_best_threshold_scalars(
                    writer,
                    train_threshold_metrics,
                    best_epoch if best_epoch != -1 else config['num_epochs'],
                    prefix='train'
                )
        if train_confusion is not None and writer is not None:
            fig = create_confusion_matrix_figure(train_confusion)
            writer.add_figure(
                'ConfusionMatrix/train',
                fig,
                global_step=best_epoch if best_epoch != -1 else config['num_epochs']
            )
            plt.close(fig)
        
        # Log val metrics
        if val_threshold_metrics is not None:
            log_best_threshold_metrics(val_threshold_metrics, prefix="Val")
            if writer is not None:
                log_best_threshold_scalars(
                    writer,
                    val_threshold_metrics,
                    best_epoch if best_epoch != -1 else config['num_epochs'],
                    prefix='val'
                )
        if val_confusion is not None and writer is not None:
            fig = create_confusion_matrix_figure(val_confusion)
            writer.add_figure(
                'ConfusionMatrix/val',
                fig,
                global_step=best_epoch if best_epoch != -1 else config['num_epochs']
            )
            plt.close(fig)
        return best_val_auc
    else:
        # If no best model was saved (shouldn't happen), use final evaluation
        train_loss, train_acc, train_auc, train_confusion, train_threshold_metrics = evaluate(
            model, train_loader, criterion, device, return_confusion=True
        )
        val_loss, val_acc, val_auc, val_confusion, val_threshold_metrics = evaluate(
            model, val_loader, criterion, device, return_confusion=True
        )
        logger.info(
            f"\nFinal Train Results - Loss: {train_loss:.4f}, "
            f"Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}"
        )
        logger.info(
            f"\nFinal Val Results - Loss: {val_loss:.4f}, "
            f"Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}"
        )
        if train_threshold_metrics is not None:
            log_best_threshold_metrics(train_threshold_metrics, prefix="Train")
            if writer is not None:
                log_best_threshold_scalars(writer, train_threshold_metrics, config['num_epochs'], prefix='train')
        if train_confusion is not None and writer is not None:
            fig = create_confusion_matrix_figure(train_confusion)
            writer.add_figure(
                'ConfusionMatrix/train',
                fig,
                global_step=config['num_epochs']
            )
            plt.close(fig)
        if val_threshold_metrics is not None:
            log_best_threshold_metrics(val_threshold_metrics, prefix="Val")
            if writer is not None:
                log_best_threshold_scalars(writer, val_threshold_metrics, config['num_epochs'], prefix='val')
        if val_confusion is not None and writer is not None:
            fig = create_confusion_matrix_figure(val_confusion)
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
    model_name = "A1 (Mean Pooling)" if config['model_type'] == 'mean' else "A2 (Attention Pooling)"
    
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
        feature_columns=["AST", "ALT", "PLT", "APRI", "FIB_4"],
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
        
        logger.info(f"Train/Val Split:")
        logger.info(f"  Train: {len(train_records)} patients ({len(train_records)/len(patient_records)*100:.1f}%)")
        logger.info(f"  Val:   {len(val_records)} patients ({len(val_records)/len(patient_records)*100:.1f}%)")
        
        # Train single model
        run_name = f"{model_slug}_single_{config['backbone']}_{timestamp}"
        writer = SummaryWriter(log_dir=str(config['log_dir'] / run_name))
        try:
            val_auc = train(0, train_records, val_records, config, writer, run_name=run_name)
        finally:
            writer.close()
        
        # Print results
        logger.info(f"{'='*70}\n")
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
        run_base = f"{model_slug}_cv_{config['backbone']}_{timestamp}"
        
        for fold in range(config['n_folds']):
            # Get train and val indices for this fold
            train_idx = folds[fold][0]
            val_idx = folds[fold][1]
            
            # Split records
            train_records, val_records = split_patients_stratified(
                patient_records, train_idx, val_idx
            )
            
            # Train fold
            fold_run_name = f"{run_base}_fold{fold + 1}"
            writer = SummaryWriter(log_dir=str(config['log_dir'] / fold_run_name))
            try:
                val_auc = train(fold, train_records, val_records, config, writer, run_name=fold_run_name)
            finally:
                writer.close()
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

