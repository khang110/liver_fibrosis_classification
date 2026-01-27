"""Training script for Clinical-Only Model.

This script implements 5-fold cross-validation training for the clinical-only MLP model.
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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from data.clinical_data import (
    ClinicalConfig,
    ClinicalPreprocessor,
    load_clinical_table,
)
from data.clinical_dataset import ClinicalDataset
from models.clinical_model import ClinicalModel
from data.image_index import PatientRecord, build_patient_records # Kept for patient splitting logic if needed, or we can just split the dataframe directly. 
# Actually, let's look at how train_clinical.py does it. It uses build_patient_records to get the patient list.
# But for clinical only, we might just want to use the CSV.
# However, to keep folds consistent with other experiments, it's BEST to use the same patient records if possible, 
# or at least use the same splitting logic. 
# Let's assume we can just load the clinical table and split the IDs.

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_train_only_clinical.log", encoding="utf-8"),
        logging.StreamHandler()
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
        description="Train Clinical-Only Model"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training. Default: 32'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate. Default: 1e-3'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Maximum number of epochs. Default: 100'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=20,
        help='Early stopping patience. Default: 20'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Default: auto-detect'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=64,
        help='Hidden dimension for MLP. Default: 64'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout probability. Default: 0.5'
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
        help='Disable cross-validation and use a single train/val split (80/20). Default: False'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Validation split ratio when --no_cv is used. Default: 0.2'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay (L2 regularization). Default: 1e-4'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='runs/clinical_only',
        help='Root directory for TensorBoard logs. Default: runs/clinical_only'
    )
    
    return parser.parse_args()


def get_config(args):
    """Get configuration dictionary."""
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = {
        'clinical_csv': Path("data/annotations/175_clinical_5_variables.csv"),
        'patient_id_column': "NO",
        'label_column': "CL_F2",
        'n_folds': 5,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_min_delta': 0.001,
        'device': device,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'clinical_features': ["AST", "ALT", "PLT", "APRI", "FIB_4"],
        'seed': args.seed,
        'no_cv': args.no_cv,
        'val_split': args.val_split,
        'weight_decay': args.weight_decay,
        'log_dir': Path(args.log_dir),
    }
    
    return config


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
            tn, fp, fn, tp = conf.ravel()
            
            sensitivity = recall
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            ppv = precision
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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, Optional[float], Optional[BestThresholdMetrics]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_labels = []
    
    for features, labels, _ in dataloader:
        features = features.to(device)
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
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
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels, _ in dataloader:
            features = features.to(device)
            labels = labels.to(device).long()
            
            logits = model(features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = correct / total if total > 0 else 0.0
    
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    probabilities = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()[:, 1]
    
    try:
        auc = roc_auc_score(all_labels, probabilities)
    except ValueError:
        auc = 0.5
        
    best_threshold_metrics = find_best_threshold(probabilities, all_labels)
    confusion = None
    if return_confusion and best_threshold_metrics is not None:
        confusion = best_threshold_metrics['confusion_matrix']
    
    return avg_loss, avg_acc, auc, confusion, best_threshold_metrics


def train(
    fold: int,
    train_ids: List[str],
    val_ids: List[str],
    df: pd.DataFrame,
    config: dict,
    writer: Optional[SummaryWriter] = None
) -> float:
    """Train model for one fold."""
    logger.info(f"\n{'='*30} Fold {fold + 1} {'='*30}")
    
    # Preprocessor
    preprocessor = ClinicalPreprocessor(
        numeric_features=config['clinical_features'],
        categorical_features=[],
    )
    # Fit on training data ONLY
    train_df = df.loc[train_ids]
    preprocessor.fit(train_df)
    
    # Datasets
    train_dataset = ClinicalDataset(train_ids, df, preprocessor, label_column=config['label_column'])
    val_dataset = ClinicalDataset(val_ids, df, preprocessor, label_column=config['label_column'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Model
    device = torch.device(config['device'])
    input_dim = len(preprocessor.feature_names)
    logger.info(f"Clinical input dimension: {input_dim}")
    
    model = ClinicalModel(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(device)
    
    # loss weighting
    train_labels = [int(df.loc[pid, config['label_column']]) for pid in train_ids]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    if pos_count > 0:
        class_weights = torch.tensor([1.0, neg_count / pos_count], dtype=torch.float32).to(device)
    else:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    
    logger.info(f"Class weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Loop
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    best_epoch = -1
    
    for epoch in range(config['num_epochs']):
        train_loss, train_acc, train_auc, train_threshold_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_acc, val_auc, _, val_threshold_metrics = evaluate(
            model, val_loader, criterion, device
        )
        
        train_auc_str = f"{train_auc:.4f}" if train_auc is not None else "nan"
        logger.info(
            f"Epoch {epoch+1:03d} | "
            f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, AUC={train_auc_str} | "
            f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}"
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
        
        if writer:
            step = epoch + 1 + (fold * config['num_epochs'])
            writer.add_scalar(f'Fold{fold}/Loss/train', train_loss, step)
            writer.add_scalar(f'Fold{fold}/Loss/val', val_loss, step)
            writer.add_scalar(f'Fold{fold}/AUC/val', val_auc, step)
            if train_auc is not None:
                writer.add_scalar(f'Fold{fold}/AUC/train', train_auc, step)
            if train_threshold_metrics is not None:
                log_best_threshold_scalars(writer, train_threshold_metrics, step, prefix=f'Fold{fold}/train')
            if val_threshold_metrics is not None:
                log_best_threshold_scalars(writer, val_threshold_metrics, step, prefix=f'Fold{fold}/val')
        
        # Early stopping
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
            
    # Load best model and re-evaluate to get final metrics
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with validation AUC: {best_val_auc:.4f}")
        
        # Re-evaluate best model
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
        
        # Log final metrics
        if train_threshold_metrics is not None:
            log_best_threshold_metrics(train_threshold_metrics, prefix="Train")
        if val_threshold_metrics is not None:
            log_best_threshold_metrics(val_threshold_metrics, prefix="Val")
            
    return best_val_auc


def main():
    args = parse_args()
    set_global_seed(args.seed)
    config = get_config(args)
    
    logger.info(f"Config: {config}")
    
    # Load Data
    clinical_config = ClinicalConfig(
        csv_path=config['clinical_csv'],
        feature_columns=config['clinical_features'],
        patient_id_column=config['patient_id_column'],
        label_column=config['label_column'],
        fibrosis_stage_column=None  # Not present in this CSV
    )
    
    # Load all clinical data
    df = load_clinical_table(clinical_config)
    logger.info(f"Loaded clinical data: {len(df)} patients")
    
    patient_ids = df.index.tolist()
    labels = df[config['label_column']].values
    
    if config['no_cv']:
        # Single split
        train_ids, val_ids = train_test_split(
            patient_ids, 
            test_size=config['val_split'], 
            stratify=labels,
            random_state=args.seed
        )
        logger.info(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
        auc = train(0, train_ids, val_ids, df, config)
        logger.info(f"Final Val AUC: {auc:.4f}")
        
    else:
        # 5-Fold CV
        skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
        fold_aucs = []
        
        writer = SummaryWriter(log_dir=str(config['log_dir']))
        
        # patient_ids is a list, we need numpy array for indexing
        patient_ids_arr = np.array(patient_ids)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids_arr, labels)):
            train_ids = patient_ids_arr[train_idx].tolist()
            val_ids = patient_ids_arr[val_idx].tolist()
            
            auc = train(fold, train_ids, val_ids, df, config, writer) # Added writer
            fold_aucs.append(auc)
            logger.info(f"Fold {fold+1} Best Val AUC: {auc:.4f}")
            
        logger.info(f"\nMean AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
        writer.close()


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split # Import here to avoid top-level dependency if possible, but it's fine
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise
