#!/bin/bash

# CPU-Optimized Unified Pell-Gregory Model Training Script
# Optimized for i3.xlarge (4 vCPUs, 30GB RAM)

model_name='UnifiedPellGregory'
mode='unified'
ensemble_size=1

echo "Starting CPU-Optimized Training on i3.xlarge..."
echo "Model: $model_name"
echo "Mode: $mode" 
echo "Ensemble size: $ensemble_size"
echo "Optimizing for 4 vCPUs with maximum CPU utilization..."

# Set CPU optimization environment variables
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "CPU Thread Settings:"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS" 
echo "  TORCH_NUM_THREADS: $TORCH_NUM_THREADS"

for I in $(seq 1 $ensemble_size)
do
    echo "Training ensemble member $I/$ensemble_size..."
    python train.py \
        --MODEL_PATH trained/$mode/$model_name \
        --MODEL_NAME ${I}_cpu_optimized \
        --EXPERIMENT_NAME cpu_optimized_experiment_$I \
        --MODEL unet \
        --BATCH_SIZE 8 \
        --IMAGE_SIZE 256 \
        --GAUSS_SIGMA 5.0 \
        --GAUSS_AMPLITUDE 1000.0 \
        --LEARN_RATE 5e-4 \
        --WEIGHT_DECAY 1e-5 \
        --EPOCHS 100 \
        --VALID_RATIO 0.15 \
        --DOWN_DROP '0.3,0.3,0.3,0.3' \
        --UP_DROP '0.3,0.3,0.3,0.3' \
        --USE_ELASTIC_TRANS False \
        --USE_AFFINE_TRANS False \
        --USE_HORIZONTAL_FLIP False \
        --OPTIM_PATIENCE 10
        
    echo "Completed training ensemble member $I"
done

echo "CPU-optimized training completed!"
echo "Models saved in: trained/$mode/$model_name/"