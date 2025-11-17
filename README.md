# Train A1: Mean pooling model
python train_bmode_a1.py --model_type mean

# Train A2: Attention pooling model
python train_bmode_a1.py --model_type attention

# Train A2 with custom attention hidden dimension
python train_bmode_a1.py --model_type attention --attention_hidden 256

# Train A1 with ResNet34 backbone
python train_bmode_a1.py --model_type mean --backbone resnet34

# Train A2 with custom batch size and learning rate
python train_bmode_a1.py --model_type attention --batch_size 16 --learning_rate 5e-5

# Train C1: Mean pooling
python train_bmode_clinical.py --pooling mean --backbone resnet18

# Train C2: Attention pooling
python train_bmode_clinical.py --pooling attention --backbone resnet18

# Train C1 with custom hyperparameters
python train_bmode_clinical.py \
    --pooling mean \
    --backbone resnet34 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --clinical_dim 64 \
    --fusion_hidden 256