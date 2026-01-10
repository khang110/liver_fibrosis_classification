#!/bin/bash

# Script để chạy training với nhiều seed khác nhau
# Usage: bash run_multiple_seeds.sh

# Danh sách các seed cần thử
SEEDS=(456 789 2024 314159 2026)

# Tham số cố định từ train_clinical.sh
POOLING="mean"
BACKBONE="efficientnetv2_b0"
LEARNING_RATE="1e-5"
NUM_EPOCHS="40"
EARLY_STOPPING_PATIENCE="10"
DEVICE="cuda"
NUM_WORKERS="4"
CLINICAL_DIM="32"
FUSION_HIDDEN="128"
DROPOUT="0.4"
WEIGHT_DECAY="1e-4"
LABEL_SMOOTHING="0.1"

# Tạo thư mục để lưu kết quả
RESULTS_DIR="results_multiple_seeds"
mkdir -p $RESULTS_DIR

echo "========================================="
echo "Running training with multiple seeds"
echo "Seeds: ${SEEDS[@]}"
echo "========================================="
echo ""

# Chạy training cho từng seed
for seed in "${SEEDS[@]}"
do
    echo "========================================="
    echo "Training with seed: $seed"
    echo "========================================="
    
    # Tạo log file riêng cho từng seed
    LOG_FILE="$RESULTS_DIR/seed_${seed}_$(date +%Y%m%d_%H%M%S).log"
    
    # Chạy training
    python train_clinical.py \
        --pooling $POOLING \
        --backbone $BACKBONE \
        --learning_rate $LEARNING_RATE \
        --num_epochs $NUM_EPOCHS \
        --early_stopping_patience $EARLY_STOPPING_PATIENCE \
        --device $DEVICE \
        --num_workers $NUM_WORKERS \
        --clinical_dim $CLINICAL_DIM \
        --fusion_hidden $FUSION_HIDDEN \
        --dropout $DROPOUT \
        --weight_decay $WEIGHT_DECAY \
        --label_smoothing $LABEL_SMOOTHING \
        --seed $seed 2>&1 | tee $LOG_FILE
    
    echo ""
    echo "Finished training with seed $seed"
    echo "Log saved to: $LOG_FILE"
    echo ""
    sleep 2
done

echo "========================================="
echo "All training runs completed!"
echo "Results saved in: $RESULTS_DIR"
echo "========================================="

# Tạo summary từ các log files
echo ""
echo "Creating summary of results..."
python analyze_seed_results.py --results_dir $RESULTS_DIR
