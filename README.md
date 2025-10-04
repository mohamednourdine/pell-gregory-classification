# Pell-Gregory Landmark Detection

A deep learning system for automated landmark detection in medical/anatomical images using U-Net architecture for separate left and right side classification.

## Overview

This project implements a robust landmark detection system that predicts 5 anatomical landmarks per side (left/right) using separate U-Net models. The system achieves sub-millimeter accuracy through optimized separate training, significantly outperforming unified training approaches.

## Key Features

- **Separate Model Training**: Independent U-Net models for left (37-38-PELLGREGORY) and right (47-48-PELLGREGORY) side landmarks
- **High Accuracy**: Achieves sub-1mm Mean Radial Error (MRE) with proven separate training approach
- **CPU Optimized**: Efficient training with optimized threading (4 threads) and batch processing
- **Production Ready**: Clean, validated codebase with comprehensive error handling and evaluation metrics
- **Original Format Compatibility**: Generates predictions in the original CSV format for seamless integration

## Architecture

- **Model**: U-Net with 5 output channels (one per landmark)
- **Input**: 256x256 RGB images (converted from grayscale)
- **Output**: 5 heatmaps representing landmark probability distributions
- **Training**: Separate models for optimal convergence and accuracy

## Performance

| Approach | Mean Radial Error | Status |
|----------|------------------|---------|
| Separate Training | **< 1mm** | ✅ Production |
| Unified Training | 80+ mm | ❌ Deprecated |

## Quick Start

### Training

```bash
# Train left side model (37-38-PELLGREGORY)
./train_left_model.sh

# Train right side model (47-48-PELLGREGORY)  
./train_right_model.sh

# Or train manually
python train_separate.py --SIDE left --EPOCHS 20 --BATCH_SIZE 16
python train_separate.py --SIDE right --EPOCHS 20 --BATCH_SIZE 16
```

### Prediction Generation

```bash
# Generate predictions for both sides
python generate_predictions_separate.py \
    --SIDE left \
    --MODEL_PATH trained/left/LeftSidePellGregory.pth \
    --DATA_PATH data/dataset/resized/37-38-PELLGREGORY/test

python generate_predictions_separate.py \
    --SIDE right \
    --MODEL_PATH trained/right/RightSidePellGregory.pth \
    --DATA_PATH data/dataset/resized/47-48-PELLGREGORY/test
```

### Evaluation

```bash
# Evaluate models
./evaluate_left_model.sh
./evaluate_right_model.sh

# Or evaluate manually
python evaluate_separate.py --SIDE left --PREDICTIONS_CSV LEFT-PG-Predictions.csv
python evaluate_separate.py --SIDE right --PREDICTIONS_CSV RIGHT-PG-Predictions.csv
```

## Project Structure

```
pell-gregory-classification/
├── train_separate.py              # Main training script
├── generate_predictions_separate.py # Prediction generation
├── evaluate_separate.py           # Model evaluation
├── train_left_model.sh            # Left side training script
├── train_right_model.sh           # Right side training script
├── evaluate_left_model.sh         # Left side evaluation script
├── evaluate_right_model.sh        # Right side evaluation script
├── model/
│   ├── __init__.py
│   └── unet_model.py              # U-Net architecture
├── utilities/
│   ├── common_utils.py            # Constants and utilities
│   ├── landmark_utils.py          # Dataset and landmark functions
│   ├── eval_utils.py              # Evaluation metrics
│   └── plotting.py                # Visualization utilities
└── data/dataset/resized/
    ├── 37-38-PELLGREGORY/         # Left side dataset
    ├── 47-48-PELLGREGORY/         # Right side dataset
    └── annotations/               # Landmark annotations
```

## Configuration

Key parameters in `utilities/common_utils.py`:

```python
N_LANDMARKS = 5          # Landmarks per side
ORIG_IMAGE_X = 1100      # Original image width
ORIG_IMAGE_Y = 600       # Original image height  
PIXELS_PER_MM = 6        # Pixel to millimeter conversion
```

## Dataset Format

- **Images**: PNG format, 256x256 pixels
- **Annotations**: Text files with landmark coordinates (x,y) per line
- **Structure**: Separate train/test splits for each side

### Example Annotation File
```
123.45,234.56
145.67,245.78
167.89,256.90
189.01,267.12
201.23,278.34
```

## Evaluation Metrics

The system reports comprehensive accuracy metrics:

- **MRE**: Mean Radial Error (mm)
- **STD**: Standard deviation of errors
- **SDR_X**: Success Detection Rate within X mm (2, 2.5, 3, 4 mm)

## CPU Optimization

Training is optimized for CPU with:
- `OMP_NUM_THREADS=4`
- `MKL_NUM_THREADS=4` 
- `TORCH_NUM_THREADS=4`
- Batch size: 16

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Pandas
- PIL/Pillow
- scikit-image
- pathlib

## Installation

```bash
# Create conda environment
conda create -n pell-gregory python=3.8
conda activate pell-gregory

# Install dependencies
pip install torch torchvision numpy matplotlib pandas pillow scikit-image
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{pell-gregory-landmark-detection,
  title={Pell-Gregory Landmark Detection using Separate U-Net Models},
  author={Your Name},
  year={2025},
  url={https://github.com/mohamednourdine/pell-gregory-classification}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
