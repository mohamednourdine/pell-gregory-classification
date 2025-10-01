#!/bin/bash

# Unified Pell-Gregory Model Training Script
# Trains a single model for both left (37-38) and right (47-48) landmarks

model_name='UnifiedPellGregory'
mode='unified'
ensemble_size=1

echo "Starting Unified Pell-Gregory Model Training..."
echo "Model: $model_name"
echo "Mode: $mode" 
echo "Ensemble size: $ensemble_size"
echo "Training unified model for both left and right landmarks (10 total landmarks)"

for I in $(seq 1 $ensemble_size)
do
    echo "Training ensemble member $I/$ensemble_size..."
    python train.py \
        --MODEL_PATH trained/$mode/$model_name \
        --MODEL_NAME $I \
        --EXPERIMENT_NAME unified_experiment_$I \
        --MODEL unet \
        --BATCH_SIZE 8 \
        --IMAGE_SIZE 256 \
        --GAUSS_SIGMA 5.0 \
        --GAUSS_AMPLITUDE 1000.0 \
        --LEARN_RATE 1e-3 \
        --WEIGHT_DECAY 0.0 \
        --EPOCHS 200 \
        --VALID_RATIO 0.15 \
        --DOWN_DROP '0.4,0.4,0.4,0.4' \
        --UP_DROP '0.4,0.4,0.4,0.4' \
        --USE_ELASTIC_TRANS True \
        --USE_AFFINE_TRANS True \
        --USE_HORIZONTAL_FLIP False \
        --OPTIM_PATIENCE 15
        
    echo "Completed training ensemble member $I"
done

echo "Unified model training completed!"
echo "Models saved in: trained/$mode/$model_name/"