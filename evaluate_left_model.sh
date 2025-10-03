#!/bin/bash

# Left Side Model Evaluation Script
# Evaluates only the left side (37-38-PELLGREGORY) model

source ~/miniconda3/etc/profile.d/conda.sh
conda activate maht-net

echo "=========================================="
echo "Left Side Pell-Gregory Model Evaluation"
echo "=========================================="

# Default parameters
MODEL_PATH="trained/left/LeftSidePellGregory/1.pth"
DATA_SPLIT="test"
SAMPLES=15
LOG_PATH="logs/left"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data-split)
            DATA_SPLIT="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --log-path)
            LOG_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model-path PATH    Path to trained left model (default: $MODEL_PATH)"
            echo "  --data-split SPLIT   Data split to evaluate on: train/test (default: $DATA_SPLIT)"
            echo "  --samples NUM        Number of MC samples (default: $SAMPLES)"
            echo "  --log-path PATH      Path to save logs (default: $LOG_PATH)"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

MODEL_NAME=$(basename "$MODEL_PATH" .pth)

echo "Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Model Name: $MODEL_NAME"
echo "  Data Split: $DATA_SPLIT"
echo "  Samples: $SAMPLES"
echo "  Log Path: $LOG_PATH"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    echo "Please ensure the left model has been trained first."
    exit 1
fi

# Step 1: Generate predictions
echo "Step 1: Generating predictions for left model..."
echo "----------------------------------------"

python generate_predictions_separate.py \
    --MODEL_PATH "$MODEL_PATH" \
    --SIDE left \
    --DATA_SPLIT "$DATA_SPLIT" \
    --LOG_PATH "$LOG_PATH" \
    --SAMPLES "$SAMPLES" \
    --BATCH_SIZE 32

if [ $? -ne 0 ]; then
    echo "ERROR: Prediction generation failed!"
    exit 1
fi

echo ""
echo "✅ Predictions generated successfully!"
echo ""

# Step 2: Evaluate model
echo "Step 2: Evaluating left model performance..."
echo "----------------------------------------"

python evaluate_separate.py \
    --SIDE left \
    --DATA_SPLIT "$DATA_SPLIT" \
    --LOG_PATH "$LOG_PATH" \
    --SAMPLES "$SAMPLES" \
    --MODEL_NAME "$MODEL_NAME"

if [ $? -ne 0 ]; then
    echo "ERROR: Model evaluation failed!"
    exit 1
fi

echo ""
echo "✅ Left model evaluation completed successfully!"
echo ""
echo "Results Summary:"
echo "  - Predictions saved in: $LOG_PATH/"
echo "  - Evaluated left side: 5 landmarks (37-38-PELLGREGORY)"
echo "  - Used $SAMPLES Monte Carlo samples for uncertainty estimation"
echo ""
echo "=========================================="