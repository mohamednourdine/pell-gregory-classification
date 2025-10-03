#!/usr/bin/env python3
"""
Separate Model Evaluation Script
Evaluates either left (37-38-PELLGREGORY) or right (47-48-PELLGREGORY) landmarks
"""

import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from collections import OrderedDict

from utilities.common_utils import *
from utilities.eval_utils import *

def get_predictions_for_image_from_df(df, image_name, n_samples):
    """Helper function to extract predictions from DataFrame"""
    image_df = df[df['image_name'] == image_name]
    
    if len(image_df) == 0:
        return None, None
    
    # Group by sample and landmark
    landmark_samples = []
    activation_samples = []
    
    for sample_idx in range(n_samples):
        sample_df = image_df[image_df['sample'] == sample_idx]
        if len(sample_df) == 0:
            continue
            
        # Sort by landmark index
        sample_df = sample_df.sort_values('landmark')
        
        landmarks = sample_df[['x', 'y']].values
        activations = sample_df['activation'].values
        
        landmark_samples.append(landmarks)
        activation_samples.append(activations)
    
    return landmark_samples, activation_samples

def main():
    parser = argparse.ArgumentParser(description='Separate Model Evaluation')
    parser.add_argument('--MODEL_NAME', type=str, required=True, help='Name of the model to evaluate')
    parser.add_argument('--SIDE', type=str, required=True, choices=['left', 'right'], 
                        help='Which side to evaluate: left (37-38) or right (47-48)')
    parser.add_argument('--DATA_SPLIT', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--LOG_PATH', type=str, default='logs', help='Directory containing predictions')
    parser.add_argument('--SAMPLES', type=int, default=10, help='Number of MC dropout samples')

    args = parser.parse_args()
    
    # Set paths based on side
    if args.SIDE == 'left':
        side_name = '37-38-PELLGREGORY'
        print(f'Evaluating LEFT side model ({side_name}) performance for model {args.MODEL_NAME}')
    else:
        side_name = '47-48-PELLGREGORY'
        print(f'Evaluating RIGHT side model ({side_name}) performance for model {args.MODEL_NAME}')
    
    print(f'Split: {args.DATA_SPLIT}')
    
    # Setup paths
    data_path = Path('data/dataset/resized')
    images_path = data_path / side_name / args.DATA_SPLIT
    annotations_path = data_path / 'annotations' / side_name / args.DATA_SPLIT
    
    # Get test files
    test_files = list_files(images_path)
    print(f'Test images: {len(test_files)}')
    
    # Compute true landmarks
    true_landmarks_dict = OrderedDict()
    for img_path in test_files:
        true_landmarks_dict[img_path.name] = get_true_landmarks(annotations_path, img_path)
    
    # Load pre-generated model predictions
    model_log_dir = Path(args.LOG_PATH) / args.DATA_SPLIT / args.SIDE / args.MODEL_NAME / 'predictions'
    predictions_file = model_log_dir / f'{args.SIDE.upper()}-PG-Predictions.csv'
    
    if not predictions_file.exists():
        print(f"❌ Predictions file not found: {predictions_file}")
        print(f"Please run generate_predictions_separate.py first with --SIDE {args.SIDE}")
        sys.exit(1)
    
    predictions_df = pd.read_csv(predictions_file)
    print(f"Loaded {len(predictions_df)} prediction records")
    
    # Compute metrics
    radial_errors = np.zeros((len(test_files), N_LANDMARKS))
    
    print(f"\\n=== EVALUATING {args.SIDE.upper()} SIDE ({side_name}) ===")
    for i, image_file in enumerate(test_files):
        # Get predictions for this image (original format uses 'file' column)
        image_predictions = predictions_df[predictions_df['file'] == image_file.name]
        
        if len(image_predictions) == 0:
            print(f"WARNING: No predictions found for {image_file.name}")
            continue
            
        # Extract landmarks from original format (direct columns instead of samples)
        predicted_landmarks = np.zeros((len(image_predictions), N_LANDMARKS, 2))
        activations = np.zeros((len(image_predictions), N_LANDMARKS))
        
        for idx, row in image_predictions.iterrows():
            for lm_idx in range(N_LANDMARKS):
                predicted_landmarks[idx, lm_idx, 0] = row[f'{lm_idx}_x']  # X
                predicted_landmarks[idx, lm_idx, 1] = row[f'{lm_idx}_y']  # Y
                activations[idx, lm_idx] = row[f'{lm_idx}_act']
        
        # Compute mean across samples
        predicted_landmarks_mean = np.mean(predicted_landmarks, axis=0)
        predicted_landmarks_var = np.var(predicted_landmarks, axis=0)
        
        # Compute radial errors
        true_landmarks = true_landmarks_dict[image_file.name]
        radial_errors[i] = get_radial_errors_mm_for_image(true_landmarks, predicted_landmarks_mean)
        
        if i < 5:  # Print first few for debugging
            print(f"Image: {image_file.name}")
            print(f"  True landmarks shape: {true_landmarks.shape}")
            print(f"  Predicted landmarks shape: {predicted_landmarks_mean.shape}")
            print(f"  MRE: {np.mean(radial_errors[i]):.2f}mm")
    
    # Compute and print metrics
    print("\\n" + "="*60)
    print(f"{args.SIDE.upper()} SIDE MODEL EVALUATION RESULTS")
    print("="*60)
    
    metrics = get_accuracy_metrics(radial_errors)
    print_accuracy_metrics(metrics)
    
    print("\\n" + "="*60)
    print(f'SUMMARY for {args.SIDE} side model: {args.MODEL_NAME}')
    print(f'Test split: {args.DATA_SPLIT}, Samples: {args.SAMPLES}')
    print(f'Images: {len(test_files)}')
    print(f'MRE: {metrics["mre"]:.2f}mm')
    print(f'STD: {metrics["std"]:.2f}mm')
    print("="*60)

if __name__ == '__main__':
    main()