#!/bin/bash

# Test script for unified model training
echo "Testing Unified Pell-Gregory Model..."

python train.py \
    --DATA_PATH ./data \
    --MODEL_PATH ./trained \
    --MODEL_NAME unified_test_model \
    --EXPERIMENT_NAME unified_test \
    --MODEL unet \
    --BATCH_SIZE 4 \
    --IMAGE_SIZE 256 \
    --GAUSS_SIGMA 5.0 \
    --GAUSS_AMPLITUDE 1000.0 \
    --LEARN_RATE 1e-3 \
    --EPOCHS 2 \
    --VALID_RATIO 0.15 \
    --USE_ELASTIC_TRANS False \
    --USE_AFFINE_TRANS False \
    --USE_HORIZONTAL_FLIP False

echo "Training test completed!"