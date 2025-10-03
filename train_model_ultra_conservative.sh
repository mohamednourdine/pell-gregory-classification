#!/bin/bash

# Ultra-Conservative Unified Pell-Gregory Model Training Script
# Optimized for very low-memory environments with frequent checkpointing

model_name='UnifiedPellGregory'
mode='unified'
ensemble_size=1

echo "Starting Ultra-Conservative Unified Pell-Gregory Model Training..."
echo "Model: $model_name"
echo "Mode: $mode" 
echo "Ensemble size: $ensemble_size"
echo "Training unified model for both left and right landmarks (10 total landmarks)"
echo "Using ultra-conservative memory settings with frequent checkpointing..."

for I in $(seq 1 $ensemble_size)
do
    echo "Training ensemble member $I/$ensemble_size..."
    python train.py \
        --MODEL_PATH trained/$mode/$model_name \
        --MODEL_NAME $I \
        --EXPERIMENT_NAME unified_experiment_$I \
        --MODEL unet \
        --BATCH_SIZE 1 \
        --IMAGE_SIZE 256 \
        --GAUSS_SIGMA 5.0 \
        --GAUSS_AMPLITUDE 1000.0 \
        --LEARN_RATE 5e-4 \
        --WEIGHT_DECAY 0.0 \
        --EPOCHS 25 \
        --VALID_RATIO 0.15 \
        --DOWN_DROP '0.2,0.2,0.2,0.2' \
        --UP_DROP '0.2,0.2,0.2,0.2' \
        --USE_ELASTIC_TRANS False \
        --USE_AFFINE_TRANS False \
        --USE_HORIZONTAL_FLIP False \
        --OPTIM_PATIENCE 8 \
        --SAVE_EPOCHS '5,10,15,20,25'
        
    echo "Completed training ensemble member $I"
done

echo "Ultra-conservative unified model training completed!"
echo "Models saved in: trained/$mode/$model_name/"