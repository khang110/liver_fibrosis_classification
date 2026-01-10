#!/usr/bin/env python3
"""Phân tích kết quả từ nhiều lần chạy với seed khác nhau."""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def extract_metrics_from_log(log_file: Path) -> Dict[str, any]:
    """Trích xuất metrics từ file log.
    
    Args:
        log_file: Đường dẫn đến file log
        
    Returns:
        Dictionary chứa các metrics
    """
    # Regex patterns để trích xuất thông tin
    seed_pattern = r"seed: (\d+)"
    # Pattern để lấy cả list các fold AUCs từ Val AUCs: ['0.7727', '0.5720', ...]
    fold_aucs_pattern = r"Val AUCs: \[([\d\.\,\s\']+)\]"
    final_mean_pattern = r"Mean Val AUC: ([\d.]+)"
    final_std_pattern = r"± ([\d.]+)"
    
    metrics = {
        'seed': None,
        'fold_aucs': [],
        'mean_auc': None,
        'std_auc': None,
    }
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Tìm seed
            seed_match = re.search(seed_pattern, content)
            if seed_match:
                metrics['seed'] = int(seed_match.group(1))
            
            # Tìm AUC của từng fold từ list Val AUCs: ['0.7727', '0.5720', ...]
            fold_aucs_match = re.search(fold_aucs_pattern, content)
            if fold_aucs_match:
                # Parse list of AUCs
                aucs_str = fold_aucs_match.group(1)
                # Remove quotes and split by comma
                aucs_list = [float(auc.strip().strip("'")) for auc in aucs_str.split(',')]
                metrics['fold_aucs'] = aucs_list
            
            # Tìm mean và std AUC
            mean_match = re.search(final_mean_pattern, content)
            if mean_match:
                metrics['mean_auc'] = float(mean_match.group(1))
            
            std_match = re.search(final_std_pattern, content)
            if std_match:
                metrics['std_auc'] = float(std_match.group(1))
    
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return metrics


def analyze_results(results_dir: Path) -> pd.DataFrame:
    """Phân tích kết quả từ tất cả các file log.
    
    Args:
        results_dir: Thư mục chứa các file log
        
    Returns:
        DataFrame chứa kết quả phân tích
    """
    log_files = sorted(results_dir.glob("seed_*.log"))
    
    if not log_files:
        print(f"No log files found in {results_dir}")
        return pd.DataFrame()
    
    results = []
    
    for log_file in log_files:
        print(f"Analyzing: {log_file.name}")
        metrics = extract_metrics_from_log(log_file)
        
        if metrics['seed'] is not None:
            result = {
                'Seed': metrics['seed'],
                'Mean AUC': metrics['mean_auc'],
                'Std AUC': metrics['std_auc'],
                'Log File': log_file.name,
            }
            
            # Thêm AUC của từng fold
            for i, auc in enumerate(metrics['fold_aucs'], 1):
                result[f'Fold {i} AUC'] = auc
            
            results.append(result)
    
    if not results:
        print("No valid results found")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df = df.sort_values('Mean AUC', ascending=False)
    
    return df


def print_summary(df: pd.DataFrame) -> None:
    """In ra tóm tắt kết quả.
    
    Args:
        df: DataFrame chứa kết quả
    """
    if df.empty:
        return
    
    print("\n" + "="*80)
    print("SUMMARY: Training Results with Different Seeds")
    print("="*80)
    print()
    
    # In ra bảng kết quả
    print(df.to_string(index=False))
    print()
    
    # Check if we have valid data
    if df['Mean AUC'].isna().all():
        print("No valid metrics found in log files.")
        return
    
    # Thống kê tổng quan
    print("="*80)
    print("Overall Statistics:")
    print("="*80)
    print(f"Number of seeds tested: {len(df)}")
    print(f"Best Mean AUC: {df['Mean AUC'].max():.4f} (Seed: {df.loc[df['Mean AUC'].idxmax(), 'Seed']:.0f})")
    print(f"Worst Mean AUC: {df['Mean AUC'].min():.4f} (Seed: {df.loc[df['Mean AUC'].idxmin(), 'Seed']:.0f})")
    print(f"Average Mean AUC: {df['Mean AUC'].mean():.4f}")
    print(f"Std of Mean AUC: {df['Mean AUC'].std():.4f}")
    print()
    
    print(f"Lowest Std AUC: {df['Std AUC'].min():.4f} (Seed: {df.loc[df['Std AUC'].idxmin(), 'Seed']:.0f})")
    print(f"Highest Std AUC: {df['Std AUC'].max():.4f} (Seed: {df.loc[df['Std AUC'].idxmax(), 'Seed']:.0f})")
    print(f"Average Std AUC: {df['Std AUC'].mean():.4f}")
    print()
    
    # Tìm seed tốt nhất (cân bằng giữa mean cao và std thấp)
    df['Score'] = df['Mean AUC'] - 0.5 * df['Std AUC']  # Penalty cho variance cao
    best_idx = df['Score'].idxmax()
    
    print("="*80)
    print("Recommended Seed (Best Mean AUC with Low Std):")
    print("="*80)
    print(f"Seed: {df.loc[best_idx, 'Seed']:.0f}")
    print(f"Mean AUC: {df.loc[best_idx, 'Mean AUC']:.4f}")
    print(f"Std AUC: {df.loc[best_idx, 'Std AUC']:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training results from multiple seeds"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results_multiple_seeds',
        help='Directory containing log files'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Output CSV file path (optional)'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return
    
    # Phân tích kết quả
    df = analyze_results(results_dir)
    
    if df.empty:
        print("No results to analyze")
        return
    
    # In summary
    print_summary(df)
    
    # Lưu ra CSV nếu được chỉ định
    if args.output_csv:
        output_path = Path(args.output_csv)
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
