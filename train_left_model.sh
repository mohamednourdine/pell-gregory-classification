#!/bin/bash

# Separate Left Side (37-38-PELLGREGORY) Model Training Script
# Optimized for CPU training with proven results

# Activate conda environment and set CPU optimization
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maht-net
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

model_name='LeftSidePellGregory'
mode='left'
ensemble_size=1

echo "Starting Left Side Pell-Gregory Model Training..."
echo "Model: $model_name"
echo "Mode: $mode" 
echo "Ensemble size: $ensemble_size"
echo "Training for left side landmarks (37-38-PELLGREGORY) only"

for I in $(seq 1 $ensemble_size)
do
    echo "Training ensemble member $I/$ensemble_size..."
    python train_separate.py \
        --MODEL_PATH trained/$mode/$model_name \
        --MODEL_NAME $I \
        --EXPERIMENT_NAME left_side_experiment_$I \
        --MODEL unet \
        --SIDE left \
        --BATCH_SIZE 16 \
        --IMAGE_SIZE 256 \
        --GAUSS_SIGMA 5.0 \
        --GAUSS_AMPLITUDE 1000.0 \
        --LEARN_RATE 1e-3 \
        --WEIGHT_DECAY 1e-4 \
        --EPOCHS 60 \
        --VALID_RATIO 0.15 \
        --DOWN_DROP '0.4,0.4,0.4,0.4' \
        --UP_DROP '0.4,0.4,0.4,0.4' \
        --USE_ELASTIC_TRANS False \
        --USE_AFFINE_TRANS False \
        --USE_HORIZONTAL_FLIP False \
        --OPTIM_PATIENCE 15
        
    echo "Completed training left side ensemble member $I"
done

echo "Left side model training completed!"
echo "Models saved in: trained/$mode/$model_name/"