#!/usr/bin/env python3
"""
Separate Model Prediction Generator
Generates predictions for either left (37-38-PELLGREGORY) or right (47-48-PELLGREGORY) landmarks
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import time

from utilities.common_utils import *
from utilities.landmark_utils import LandmarkDataset, get_max_heatmap_activation

def get_predicted_landmarks_single(heatmap, gauss_sigma):
    """Extract predicted landmarks from single side heatmap using old working method"""
    n_landmarks = heatmap.shape[0]
    heatmap_y, heatmap_x = heatmap.shape[1:]
    pred_landmarks = np.zeros((n_landmarks, 2))
    max_activations = np.zeros(n_landmarks)
    
    for i in range(n_landmarks):
        # Use old working method with Gaussian filtering
        max_activation, pred_yx = get_max_heatmap_activation(heatmap[i], gauss_sigma)
        # Apply proper coordinate rescaling like old code
        rescale = np.array([ORIG_IMAGE_Y, ORIG_IMAGE_X]) / np.array([heatmap_y, heatmap_x])
        pred_yx = np.around(pred_yx * rescale)
        pred_landmarks[i] = pred_yx
        max_activations[i] = max_activation
    
    return pred_landmarks, max_activations

def main():
    parser = argparse.ArgumentParser(description='Separate Model Prediction Generator')
    parser.add_argument('--MODEL_PATH', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--SIDE', type=str, required=True, choices=['left', 'right'], 
                        help='Which side to predict: left (37-38) or right (47-48)')
    parser.add_argument('--DATA_SPLIT', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--LOG_PATH', type=str, default='logs', help='Directory to save predictions')
    parser.add_argument('--SAMPLES', type=int, default=5, help='Number of MC dropout samples')
    parser.add_argument('--BATCH_SIZE', type=int, default=30, help='Batch size for prediction')
    parser.add_argument('--GAUSS_SIGMA', type=float, default=5.0)
    parser.add_argument('--GAUSS_AMPLITUDE', type=float, default=1000.0)

    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set paths based on side
    if args.SIDE == 'left':
        side_name = '37-38-PELLGREGORY'
        print(f'Generating predictions for LEFT side ({side_name})')
    else:
        side_name = '47-48-PELLGREGORY'
        print(f'Generating predictions for RIGHT side ({side_name})')
    
    # Setup paths
    data_path = Path('data/dataset/resized')
    images_path = data_path / side_name / args.DATA_SPLIT
    annotations_path = data_path / 'annotations' / side_name / args.DATA_SPLIT
    
    # Get test files
    test_files = list_files(images_path)
    print(f'Found {len(test_files)} test images')
    
    # Create dataset
    test_ds = LandmarkDataset(
        test_files, annotations_path,
        args.GAUSS_SIGMA, args.GAUSS_AMPLITUDE
    )
    
    test_dl = DataLoader(test_ds, args.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load model
    model_path = Path(args.MODEL_PATH)
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    print(f'Loaded {args.SIDE} side model from {model_path}')
    print(f'Model expects {N_LANDMARKS} landmarks')
    
    # Generate predictions
    log_path = Path(args.LOG_PATH) / args.DATA_SPLIT / args.SIDE / model_path.stem / 'predictions'
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Store predictions in the original format (one row per image per sample)
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
                    
                    # Extract landmarks from prediction
                    pred_landmarks, max_activations = get_predicted_landmarks_single(
                        pred_heatmaps[img_idx].cpu().numpy(), args.GAUSS_SIGMA
                    )
                    
                    # Create row data in original format
                    row_data = {'file': img_name}
                    
                    # Add activation columns
                    for lm_idx in range(N_LANDMARKS):
                        row_data[f'{lm_idx}_act'] = max_activations[lm_idx]
                    
                    # Add Y coordinate columns
                    for lm_idx in range(N_LANDMARKS):
                        row_data[f'{lm_idx}_y'] = pred_landmarks[lm_idx, 1]  # Y coordinate
                    
                    # Add X coordinate columns
                    for lm_idx in range(N_LANDMARKS):
                        row_data[f'{lm_idx}_x'] = pred_landmarks[lm_idx, 0]  # X coordinate
                    
                    all_predictions.append(row_data)
    
    # Save predictions to CSV in original format
    predictions_df = pd.DataFrame(all_predictions)
    csv_path = log_path / f'{args.SIDE.upper()}-PG-Predictions.csv'
    predictions_df.to_csv(csv_path, index=True)  # Include index like original
    
    print(f'✅ {args.SIDE.title()} side predictions saved to {csv_path}')
    print(f'Total predictions: {len(all_predictions)}')
    print(f'Unique images: {predictions_df["file"].nunique()}')
    print(f'Samples per image: {args.SAMPLES}')
    print(f'Landmarks per side: {N_LANDMARKS}')

if __name__ == '__main__':
    main()