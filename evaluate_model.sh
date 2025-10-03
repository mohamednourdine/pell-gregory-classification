#!/bin/bash

# Unified Model Evaluation Pipeline
# Generates predictions and evaluates unified Pell-Gregory model

echo "=========================================="
echo "Unified Pell-Gregory Model Evaluation"
echo "=========================================="

# Default parameters
MODEL_PATH="trained/unified/UnifiedPellGregory/1.pth"
DATA_SPLIT="test"
SAMPLES=15
LOG_PATH="logs"

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
            echo "  --model-path PATH    Path to trained unified model (default: $MODEL_PATH)"
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

# Extract model name from path
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
    echo "Please ensure the unified model has been trained first."
    exit 1
fi

# Step 1: Generate predictions
echo "Step 1: Generating predictions for unified model..."
echo "----------------------------------------"

python generate_predictions.py \
    --MODE unified \
    --MODEL_PATH "$MODEL_PATH" \
    --DATA_SPLIT "$DATA_SPLIT" \
    --LOG_PATH "$LOG_PATH" \
    --SAMPLES "$SAMPLES" \
    --LEFT_IMAGES_PATH "./data/dataset/resized/37-38-PELLGREGORY" \
    --RIGHT_IMAGES_PATH "./data/dataset/resized/47-48-PELLGREGORY" \
    --LEFT_ANNOT_PATH "./data/dataset/resized/annotations/37-38-PELLGREGORY" \
    --RIGHT_ANNOT_PATH "./data/dataset/resized/annotations/47-48-PELLGREGORY" \
    --unified \
    --BATCH_SIZE 16

if [ $? -ne 0 ]; then
    echo "ERROR: Prediction generation failed!"
    exit 1
fi

echo ""
echo "✅ Predictions generated successfully!"
echo ""

# Step 2: Evaluate model
echo "Step 2: Evaluating unified model performance..."
echo "----------------------------------------"

python evaluate.py \
    --MODE unified \
    --DATA_SPLIT "$DATA_SPLIT" \
    --LOG_PATH "$LOG_PATH" \
    --SAMPLES "$SAMPLES" \
    --MODEL_NAME "$MODEL_NAME" \
    --LEFT_ANNOT_PATH "data/dataset/resized/annotations/37-38-PELLGREGORY/$DATA_SPLIT" \
    --RIGHT_ANNOT_PATH "data/dataset/resized/annotations/47-48-PELLGREGORY/$DATA_SPLIT" \
    --LEFT_IMAGES_PATH "data/dataset/resized/37-38-PELLGREGORY" \
    --RIGHT_IMAGES_PATH "data/dataset/resized/47-48-PELLGREGORY" \
    --unified

if [ $? -ne 0 ]; then
    echo "ERROR: Model evaluation failed!"
    exit 1
fi

echo ""
echo "✅ Unified model evaluation completed successfully!"
echo ""
echo "Results Summary:"
echo "  - Predictions saved in: $LOG_PATH/$DATA_SPLIT/unified/$MODEL_NAME/predictions/"
echo "  - Evaluated both left (37-38-PELLGREGORY) and right (47-48-PELLGREGORY) landmarks"
echo "  - Used $SAMPLES Monte Carlo samples for uncertainty estimation"
echo "  - Model predicts all 10 landmarks (5 left + 5 right) simultaneously"
echo ""
echo "=========================================="