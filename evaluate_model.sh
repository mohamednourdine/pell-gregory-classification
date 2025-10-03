#!/bin/bash

# Separate Model Evaluation Pipeline
# Generates predictions and evaluates separate left/right Pell-Gregory models

echo "=========================================="
echo "Separate Pell-Gregory Model Evaluation"
echo "=========================================="

# Default parameters
LEFT_MODEL_PATH="trained/left/LeftSidePellGregory/1.pth"
RIGHT_MODEL_PATH="trained/right/RightSidePellGregory/1.pth"
DATA_SPLIT="test"
SAMPLES=15
LOG_PATH="logs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --left-model)
            LEFT_MODEL_PATH="$2"
            shift 2
            ;;
        --right-model)
            RIGHT_MODEL_PATH="$2"
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
            echo "  --left-model PATH    Path to trained left model (default: $LEFT_MODEL_PATH)"
            echo "  --right-model PATH   Path to trained right model (default: $RIGHT_MODEL_PATH)"
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

# Extract model names from paths
LEFT_MODEL_NAME=$(basename "$LEFT_MODEL_PATH" .pth)
RIGHT_MODEL_NAME=$(basename "$RIGHT_MODEL_PATH" .pth)

echo "Configuration:"
echo "  Left Model Path: $LEFT_MODEL_PATH"
echo "  Right Model Path: $RIGHT_MODEL_PATH"
echo "  Left Model Name: $LEFT_MODEL_NAME"
echo "  Right Model Name: $RIGHT_MODEL_NAME"
echo "  Data Split: $DATA_SPLIT"
echo "  Samples: $SAMPLES"
echo "  Log Path: $LOG_PATH"
echo ""

# Check if models exist
if [ ! -f "$LEFT_MODEL_PATH" ]; then
    echo "ERROR: Left model file not found: $LEFT_MODEL_PATH"
    echo "Please ensure the left model has been trained first."
    exit 1
fi

if [ ! -f "$RIGHT_MODEL_PATH" ]; then
    echo "ERROR: Right model file not found: $RIGHT_MODEL_PATH"
    echo "Please ensure the right model has been trained first."
    exit 1
fi

# Step 1: Generate predictions for left model
echo "Step 1: Generating predictions for left model..."
echo "----------------------------------------"

python generate_predictions_separate.py \
    --MODEL_PATH "$LEFT_MODEL_PATH" \
    --SIDE left \
    --DATA_SPLIT "$DATA_SPLIT" \
    --LOG_PATH "$LOG_PATH/left" \
    --SAMPLES "$SAMPLES" \
    --BATCH_SIZE 16

if [ $? -ne 0 ]; then
    echo "ERROR: Left model prediction generation failed!"
    exit 1
fi

echo ""
echo "✅ Left model predictions generated successfully!"
echo ""

# Step 2: Generate predictions for right model
echo "Step 2: Generating predictions for right model..."
echo "----------------------------------------"

python generate_predictions_separate.py \
    --MODEL_PATH "$RIGHT_MODEL_PATH" \
    --SIDE right \
    --DATA_SPLIT "$DATA_SPLIT" \
    --LOG_PATH "$LOG_PATH/right" \
    --SAMPLES "$SAMPLES" \
    --BATCH_SIZE 16

if [ $? -ne 0 ]; then
    echo "ERROR: Right model prediction generation failed!"
    exit 1
fi

echo ""
echo "✅ Right model predictions generated successfully!"
echo ""

# Step 3: Evaluate left model
echo "Step 3: Evaluating left model performance..."
echo "----------------------------------------"

python evaluate_separate.py \
    --SIDE left \
    --DATA_SPLIT "$DATA_SPLIT" \
    --LOG_PATH "$LOG_PATH/left" \
    --SAMPLES "$SAMPLES" \
    --MODEL_NAME "$LEFT_MODEL_NAME"

if [ $? -ne 0 ]; then
    echo "ERROR: Left model evaluation failed!"
    exit 1
fi

echo ""
echo "✅ Left model evaluation completed!"
echo ""

# Step 4: Evaluate right model
echo "Step 4: Evaluating right model performance..."
echo "----------------------------------------"

python evaluate_separate.py \
    --SIDE right \
    --DATA_SPLIT "$DATA_SPLIT" \
    --LOG_PATH "$LOG_PATH/right" \
    --SAMPLES "$SAMPLES" \
    --MODEL_NAME "$RIGHT_MODEL_NAME"

if [ $? -ne 0 ]; then
    echo "ERROR: Right model evaluation failed!"
    exit 1
fi

echo ""
echo "✅ Separate model evaluation completed successfully!"
echo ""
echo "Results Summary:"
echo "  - Left model predictions saved in: $LOG_PATH/left/"
echo "  - Right model predictions saved in: $LOG_PATH/right/" 
echo "  - Left side: 5 landmarks (37-38-PELLGREGORY)"
echo "  - Right side: 5 landmarks (47-48-PELLGREGORY)"
echo "  - Used $SAMPLES Monte Carlo samples for uncertainty estimation"
echo "  - Models trained separately for optimal performance"
echo ""
echo "=========================================="