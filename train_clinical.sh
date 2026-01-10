python train_clinical.py \
    --pooling mean \
    --backbone resnet18 \
    --learning_rate 1e-5 \
    --num_epochs 40 \
    --early_stopping_patience 10 \
    --device cuda \
    --num_workers 4 \
    --clinical_dim 32 \
    --fusion_hidden 128 \
    --dropout 0.4 \
    --weight_decay 1e-4 \
    --label_smoothing 0.1 \
    # --no_cv