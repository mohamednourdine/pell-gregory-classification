import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd
from scipy import ndimage
import numpy as np
import PIL
import argparse
import shutil
import random
import io
import sys
from pathlib import Path

from utilities.common_utils import *
from utilities.landmark_utils import *
from utilities.plotting import *
from model import UnifiedUNet

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

device = 'cpu'
parser = argparse.ArgumentParser('Unified Pell-Gregory Model Prediction Generator')
parser.add_argument('--MODEL_PATH', required=True, type=str, help='Path to the unified model.')
parser.add_argument('--DATA_SPLIT', type=str, choices=['train', 'test'], default='test', help='Which data split to evaluate on.')
parser.add_argument('--LOG_PATH', type=str, default='logs', help='Path to save logs.')
parser.add_argument('--SAMPLES', type=int, default=15, help='Number of MC samples to use for prediction.')
parser.add_argument('--LEFT_IMAGES_PATH', type=str, default='./data/dataset/resized/37-38-PELLGREGORY', help='Path to left side image data.')
parser.add_argument('--RIGHT_IMAGES_PATH', type=str, default='./data/dataset/resized/47-48-PELLGREGORY', help='Path to right side image data.')
parser.add_argument('--LEFT_ANNOT_PATH', type=str, default='./data/dataset/resized/annotations/37-38-PELLGREGORY', help='Path to left side annotation data.')
parser.add_argument('--RIGHT_ANNOT_PATH', type=str, default='./data/dataset/resized/annotations/47-48-PELLGREGORY', help='Path to right side annotation data.')
parser.add_argument('--IMAGE_SIZE', type=int, default=256, help='Size the test images will be rescaled to before being passed to the model.')
parser.add_argument('--GAUSS_SIGMA', type=float, default=5, help='Sigma of the Gaussian kernel used to generate ground truth heatmaps for the landmarks.')
parser.add_argument('--GAUSS_AMPLITUDE', type=float, default=1000.0)
parser.add_argument('--BATCH_SIZE', type=int, default=30)
args = parser.parse_args()


def get_predicted_landmarks_unified(pred_heatmaps, gauss_sigma):
    """Extract landmarks from unified model predictions (10 landmarks total)"""
    n_landmarks = pred_heatmaps.shape[0]  # Should be 10 for unified model
    heatmap_y, heatmap_x = pred_heatmaps.shape[1:]
    pred_landmarks = np.zeros((n_landmarks, 2))
    max_activations = np.zeros(n_landmarks)
    
    for i in range(n_landmarks):
        max_activation, pred_yx = get_max_heatmap_activation(pred_heatmaps[i], gauss_sigma)
        max_activations[i] = max_activation
        # Rescale to original resolution and convert to mm
        rescale = np.array([ORIG_IMAGE_Y, ORIG_IMAGE_X]) / np.array([heatmap_y, heatmap_x])
        pred_landmarks[i] = np.around(pred_yx * rescale) / PIXELS_PER_MM
    
    return pred_landmarks, max_activations


def get_predicted_landmarks(pred_heatmaps, gauss_sigma):
    """Legacy function for single-side models (5 landmarks)"""
    n_landmarks = pred_heatmaps.shape[0]
    heatmap_y, heatmap_x = pred_heatmaps.shape[1:]
    pred_landmarks = np.zeros((n_landmarks, 2))
    max_activations = np.zeros(n_landmarks)
    
    for i in range(n_landmarks):
        max_activation, pred_yx = get_max_heatmap_activation(pred_heatmaps[i], gauss_sigma)
        max_activations[i] = max_activation
        # Rescale to original resolution and convert to mm
        rescale = np.array([ORIG_IMAGE_Y, ORIG_IMAGE_X]) / np.array([heatmap_y, heatmap_x])
        pred_landmarks[i] = np.around(pred_yx * rescale) / PIXELS_PER_MM
    
    return pred_landmarks, max_activations


# Using Unified Model for Both Left and Right Landmarks
print("Using Unified Model for Both Left and Right Landmarks")

# Combine test files from both sides
left_test_dir = Path(args.LEFT_IMAGES_PATH) / args.DATA_SPLIT
right_test_dir = Path(args.RIGHT_IMAGES_PATH) / args.DATA_SPLIT
left_test_files = list_files(left_test_dir)
right_test_files = list_files(right_test_dir)

print(f'Left test files: {len(left_test_files)}, Right test files: {len(right_test_files)}')

# Create unified dataset for testing
left_annot_path = Path(args.LEFT_ANNOT_PATH) / args.DATA_SPLIT
right_annot_path = Path(args.RIGHT_ANNOT_PATH) / args.DATA_SPLIT

test_ds = UnifiedLandmarkDataset(
    left_test_files, right_test_files,
    left_annot_path, right_annot_path,
    args.GAUSS_SIGMA, args.GAUSS_AMPLITUDE
)

test_dl = DataLoader(test_ds, args.BATCH_SIZE, shuffle=False, num_workers=0)

# Load unified model
model_path = Path(args.MODEL_PATH)
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

print(f'Loaded unified model from {model_path}')
print(f'Model expects {N_LANDMARKS} landmarks ({N_LANDMARKS_PER_SIDE} per side)')

# Generate predictions
log_path = Path(args.LOG_PATH) / args.DATA_SPLIT / 'unified' / model_path.stem / 'predictions'
log_path.mkdir(parents=True, exist_ok=True)

all_predictions = []

with torch.no_grad():
    for batch_idx, (imgs, true_heatmaps, img_paths) in enumerate(test_dl):
        print(f'Processing batch {batch_idx + 1}/{len(test_dl)}')
        
        imgs = imgs.to(device)
        
        # Generate multiple samples for uncertainty estimation
        for sample_idx in range(args.SAMPLES):
            model.train()  # Enable dropout for MC sampling
            pred_heatmaps = model(imgs)
            model.eval()
            
            # Process each image in the batch
            for img_idx in range(imgs.shape[0]):
                img_name = Path(img_paths[img_idx]).name
                
                # Extract landmarks from unified prediction (10 landmarks)
                pred_landmarks, max_activations = get_predicted_landmarks_unified(
                    pred_heatmaps[img_idx].cpu().numpy(), args.GAUSS_SIGMA
                )
                
                # Separate left and right landmarks
                left_landmarks = pred_landmarks[:N_LANDMARKS_PER_SIDE]  # 0-4
                right_landmarks = pred_landmarks[N_LANDMARKS_PER_SIDE:]  # 5-9
                
                left_activations = max_activations[:N_LANDMARKS_PER_SIDE]
                right_activations = max_activations[N_LANDMARKS_PER_SIDE:]
                
                # Determine which side this image belongs to
                if 'PELLGREGORY' in str(img_paths[img_idx]):
                    if '37-38-PELLGREGORY' in str(img_paths[img_idx]):
                        # Left side image
                        landmarks_to_save = left_landmarks
                        activations_to_save = left_activations
                        side = 'left'
                    else:
                        # Right side image  
                        landmarks_to_save = right_landmarks
                        activations_to_save = right_activations
                        side = 'right'
                else:
                    # Default to left side if uncertain
                    landmarks_to_save = left_landmarks
                    activations_to_save = left_activations
                    side = 'left'
                
                # Save prediction in same format as original
                for lm_idx in range(N_LANDMARKS_PER_SIDE):
                    prediction_data = {
                        'image_name': img_name,
                        'sample': sample_idx,
                        'landmark': lm_idx,
                        'x': landmarks_to_save[lm_idx, 0],
                        'y': landmarks_to_save[lm_idx, 1],
                        'activation': activations_to_save[lm_idx],
                        'side': side
                    }
                    all_predictions.append(prediction_data)

# Save predictions to CSV
predictions_df = pd.DataFrame(all_predictions)
csv_path = log_path / f'predictions_unified.csv'
predictions_df.to_csv(csv_path, index=False)

print(f'✅ Unified model predictions saved to {csv_path}')
print(f'Total predictions: {len(all_predictions)}')
print(f'Unique images: {predictions_df["image_name"].nunique()}')
print(f'Samples per image: {args.SAMPLES}')
print(f'Landmarks per side: {N_LANDMARKS_PER_SIDE}')