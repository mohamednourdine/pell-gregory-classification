#!/bin/bash

# Improved Left Side (37-38-PELLGREGORY) Model Training Script
# Fixed parameters for better convergence

# Activate conda environment and set CPU optimization
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maht-net
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

model_name='LeftSidePellGregory'
mode='left'
ensemble_size=1

echo "Starting IMPROVED Left Side Pell-Gregory Model Training..."
echo "Model: $model_name"
echo "Mode: $mode" 
echo "Ensemble size: $ensemble_size"
echo "Training for left side landmarks (37-38-PELLGREGORY) only"
echo "FIXED PARAMETERS for better convergence"

for I in $(seq 1 $ensemble_size)
do
    echo "Training ensemble member $I/$ensemble_size..."
    python train_separate.py \
        --MODEL_PATH trained/$mode/$model_name \
        --MODEL_NAME $I \
        --EXPERIMENT_NAME left_side_experiment_improved_$I \
        --MODEL unet \
        --SIDE left \
        --BATCH_SIZE 16 \
        --IMAGE_SIZE 256 \
        --GAUSS_SIGMA 2.0 \
        --GAUSS_AMPLITUDE 100.0 \
        --LEARN_RATE 1e-4 \
        --WEIGHT_DECAY 1e-5 \
        --EPOCHS 30 \
        --VALID_RATIO 0.15 \
        --DOWN_DROP '0.2,0.2,0.2,0.2' \
        --UP_DROP '0.2,0.2,0.2,0.2' \
        --USE_ELASTIC_TRANS False \
        --USE_AFFINE_TRANS False \
        --USE_HORIZONTAL_FLIP False \
        --OPTIM_PATIENCE 10
        
    echo "Completed training left side ensemble member $I"
done

echo "Left side model training completed!"
echo "Models saved in: trained/$mode/$model_name/"