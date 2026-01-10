# Activate ENV
```bash
conda activate qus
```
# Train A1: Mean pooling model
```bash
python train.py --model_type mean --backbone efficientnetv2_b2 --no_cv --learning_rate 1e-3
```
# Train A2: Attention pooling model
```bash
python train.py --model_type attention --backbone efficientnetv2_b2 --no_cv --learning_rate 4e-4
```
# Train C1: Mean pooling
```bash
python train_clinical.py --pooling mean --backbone efficientnetv2_b2 --no_cv --learning_rate 1e-3
```
# Train C2: Attention pooling
```bash
python train_clinical.py --pooling attention --backbone efficientnetv2_b2 --no_cv --learning_rate 2e-4
```
# Train: dual stream
```bash
python train_dual_stream.py \
    --model_type mean \
    --backbone_bmode efficientnetv2_b0 \
    --backbone_nakagami efficientnetv2_b0 \
    --learning_rate 1e-5 \
    --no_cv
```
# Train: dual stream + clinical 

```bash
python train_dual_stream_clinical.py \
    --pooling mean \
    --backbone_bmode efficientnetv2_b0 \
    --backbone_nakagami efficientnetv2_b0 \
    --learning_rate 1e-5 \
    --no_cv
```

# Cosine scheduler (mặc định)
```bash
python train_clinical_scheduler.py --pooling mean --scheduler cosine
```

# Plateau scheduler
```bash
python train_clinical_scheduler.py --pooling mean --scheduler plateau --scheduler_patience 5
```

# Step scheduler
```bash
python train_clinical_scheduler.py --pooling mean --scheduler step --scheduler_step_size 10
```

# Không dùng scheduler
```bash
python train_clinical_scheduler.py --pooling mean --scheduler none
```