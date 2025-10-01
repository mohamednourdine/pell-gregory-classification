# Notebooks Directory

This directory contains essential Jupyter notebooks for the Pell-Gregory classification project.

## Core Notebooks

### 📊 Results and Visualization
- **`PG Plotting - 37-38-PG-Results.ipynb`** - Visualization and analysis of results for 37-38 PELLGREGORY dataset
- **`PG Plotting - 47-48-PG-Results.ipynb`** - Visualization and analysis of results for 47-48 PELLGREGORY dataset

### 🔮 Model Inference
- **`Generate Model Predictions.ipynb`** - Generate predictions from trained models on test datasets

### 🛠️ Utilities
- **`Calculate the Number of Pixels per MM.ipynb`** - Calculate pixel-to-millimeter conversion for accurate measurements
- **`Resize images from a source directory.ipynb`** - Utility for resizing images to standard dimensions

## Usage Guidelines

### For Model Training and Evaluation:
1. Use the main Python scripts (`train.py`, `evaluate.py`) for training and evaluation
2. Use notebooks for result visualization and analysis

### For Data Preprocessing:
1. Use `Resize images from a source directory.ipynb` for new image datasets
2. Use `Calculate the Number of Pixels per MM.ipynb` to verify scaling

### For Results Analysis:
1. Use `Generate Model Predictions.ipynb` to generate predictions
2. Use the `PG Plotting` notebooks to visualize and analyze results

## Dependencies

All notebooks require the same dependencies as the main project:
- PyTorch
- NumPy
- Matplotlib
- OpenCV
- Pandas
- Scikit-image

## Notes

- Notebooks are kept for research, analysis, and experimentation
- Production training should use the main Python scripts
- Some notebooks may require path adjustments based on your data location