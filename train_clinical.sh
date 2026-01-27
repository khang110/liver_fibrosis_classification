python train_clinical.py \
    --pooling mean \
    --backbone efficientnetv2_b2 \
    --learning_rate 1e-5 \
    --num_epochs 50 \
    --early_stopping_patience 15 \
    --device cuda \
    --num_workers 4 \
    --clinical_dim 128 \
    --fusion_hidden 128 \
    --dropout 0.5 \
    --weight_decay 1e-4 \
    --label_smoothing 0.1 \
    --seed 42

    # For Repeated Stratified K-Fold CV (recommended for robustness):
    # --n_repeats 3  # This will run 5 folds x 3 repeats = 15 total folds
    
    # --use_scheduler \
    # --lr_factor 0.1 \
    # --lr_patience 5 \

    