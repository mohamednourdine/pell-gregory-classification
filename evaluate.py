import pandas as pd
import numpy as np
import PIL
import argparse
import shutil
import random
import io
import sys
from pathlib import Path
from collections import defaultdict, OrderedDict

from utilities.common_utils import *
from utilities.landmark_utils import *
from utilities.plotting import *
from utilities.eval_utils import *
from model import UnifiedUNet

parser = argparse.ArgumentParser('Unified Pell-Gregory Model Evaluation')
parser.add_argument('--DATA_SPLIT', type=str, default='test', choices=['train', 'test'], help='Which data split to evaluate on.')
parser.add_argument('--LOG_PATH', type=str, default='logs', help='Path to model logs.')
parser.add_argument('--SAMPLES', type=int, default=15, help='Number of MC samples to use for prediction.')
parser.add_argument('--MODEL_NAME', type=str, required=True, help='Name of the evaluated unified model.')
parser.add_argument('--LEFT_ANNOT_PATH', type=str, default='data/dataset/resized/annotations/37-38-PELLGREGORY/test', help='Path to left side annotation data.')
parser.add_argument('--RIGHT_ANNOT_PATH', type=str, default='data/dataset/resized/annotations/47-48-PELLGREGORY/test', help='Path to right side annotation data.')
parser.add_argument('--LEFT_IMAGES_PATH', type=str, default='data/dataset/resized/37-38-PELLGREGORY', help='Path to left side image data.')
parser.add_argument('--RIGHT_IMAGES_PATH', type=str, default='data/dataset/resized/47-48-PELLGREGORY', help='Path to right side image data.')
parser.add_argument('--IMAGE_SIZE', type=int, default=256, help='Size the test images will be rescaled to before being passed to the model.')
args = parser.parse_args()

args.LOG_PATH = Path(args.LOG_PATH)
args.LEFT_ANNOT_PATH = Path(args.LEFT_ANNOT_PATH)
args.RIGHT_ANNOT_PATH = Path(args.RIGHT_ANNOT_PATH)

print(f'Evaluating UNIFIED model performance metrics for model {args.MODEL_NAME}')
print(f'Split: {args.DATA_SPLIT}')

# Get test files from both sides
left_test_dir = Path(args.LEFT_IMAGES_PATH) / f'{args.DATA_SPLIT}'
right_test_dir = Path(args.RIGHT_IMAGES_PATH) / f'{args.DATA_SPLIT}'

left_test_files = list_files(left_test_dir)
right_test_files = list_files(right_test_dir)
    
print(f'Left test images: {len(left_test_files)}, Right test images: {len(right_test_files)}')
print(f'Total test images: {len(left_test_files) + len(right_test_files)}')
# Compute true landmarks for both sides
left_true_landmarks_dict = OrderedDict()
right_true_landmarks_dict = OrderedDict()

for img_path in left_test_files:
    left_true_landmarks_dict[img_path.name] = get_true_landmarks(args.LEFT_ANNOT_PATH, img_path)

for img_path in right_test_files:
    right_true_landmarks_dict[img_path.name] = get_true_landmarks(args.RIGHT_ANNOT_PATH, img_path)

# Load pre-generated model predictions from unified model
model_log_dir = args.LOG_PATH / f'{args.DATA_SPLIT}/unified/{args.MODEL_NAME}/predictions'
predictions_file = model_log_dir / 'predictions_unified.csv'

if not predictions_file.exists():
    print(f"❌ Predictions file not found: {predictions_file}")
    print("Please run generate_predictions.py first to generate predictions.")
    sys.exit(1)

predictions_df = pd.read_csv(predictions_file)
print(f"Loaded {len(predictions_df)} prediction records")

# Compute metrics for both sides separately and combined
left_radial_errors = np.zeros((len(left_test_files), N_LANDMARKS_PER_SIDE))
right_radial_errors = np.zeros((len(right_test_files), N_LANDMARKS_PER_SIDE))

print("\\n=== EVALUATING LEFT SIDE (37-38-PELLGREGORY) ===")
for i, image_file in enumerate(left_test_files):
    # Get predictions for this image (filter by side if available)
    image_predictions = predictions_df[
        (predictions_df['image_name'] == image_file.name) & 
        (predictions_df.get('side', 'left') == 'left')
    ]
    
    if len(image_predictions) == 0:
        print(f"WARNING: No left side predictions found for {image_file.name}")
        continue
        
    # Compute the statistics across all samples for the image
    landmark_samples, activation_samples = get_predictions_for_image_from_df(
        image_predictions, image_file.name, args.SAMPLES
    )
    predicted_landmarks_mean, predicted_landmarks_var = get_predicted_landmarks_for_image(landmark_samples)
    
    # Compute radial errors
    true_landmarks = left_true_landmarks_dict[image_file.name]
    left_radial_errors[i] = get_radial_errors_mm_for_image(true_landmarks, predicted_landmarks_mean)
    
    if i < 5:  # Print first few for debugging
        print(f"Left Image: {image_file.name}")
        print(f"  True landmarks shape: {true_landmarks.shape}")
        print(f"  Predicted landmarks shape: {predicted_landmarks_mean.shape}")
        print(f"  MRE: {np.mean(left_radial_errors[i]):.2f}mm")

print("\\n=== EVALUATING RIGHT SIDE (47-48-PELLGREGORY) ===")
for i, image_file in enumerate(right_test_files):
    # Get predictions for this image (filter by side if available)
    image_predictions = predictions_df[
        (predictions_df['image_name'] == image_file.name) & 
        (predictions_df.get('side', 'right') == 'right')
    ]
    
    if len(image_predictions) == 0:
        print(f"WARNING: No right side predictions found for {image_file.name}")
        continue
        
    # Compute the statistics across all samples for the image
    landmark_samples, activation_samples = get_predictions_for_image_from_df(
        image_predictions, image_file.name, args.SAMPLES
    )
    predicted_landmarks_mean, predicted_landmarks_var = get_predicted_landmarks_for_image(landmark_samples)
    
    # Compute radial errors
    true_landmarks = right_true_landmarks_dict[image_file.name]
    right_radial_errors[i] = get_radial_errors_mm_for_image(true_landmarks, predicted_landmarks_mean)
    
    if i < 5:  # Print first few for debugging
        print(f"Right Image: {image_file.name}")
        print(f"  True landmarks shape: {true_landmarks.shape}")
        print(f"  Predicted landmarks shape: {predicted_landmarks_mean.shape}")
        print(f"  MRE: {np.mean(right_radial_errors[i]):.2f}mm")

# Compute separate metrics for each side
print("\\n" + "="*60)
print("UNIFIED MODEL EVALUATION RESULTS")
print("="*60)

print("\\n--- LEFT SIDE (37-38-PELLGREGORY) METRICS ---")
left_metrics = get_accuracy_metrics(left_radial_errors)
print_accuracy_metrics(left_metrics)

print("\\n--- RIGHT SIDE (47-48-PELLGREGORY) METRICS ---")
right_metrics = get_accuracy_metrics(right_radial_errors)
print_accuracy_metrics(right_metrics)

# Compute combined metrics
print("\\n--- COMBINED (BOTH SIDES) METRICS ---")
combined_radial_errors = np.vstack([left_radial_errors, right_radial_errors])
combined_metrics = get_accuracy_metrics(combined_radial_errors)
print_accuracy_metrics(combined_metrics)

print("\\n" + "="*60)
print(f'SUMMARY for unified model: {args.MODEL_NAME}')
print(f'Test split: {args.DATA_SPLIT}, Samples: {args.SAMPLES}')
print(f'Left images: {len(left_test_files)}, Right images: {len(right_test_files)}')
print(f'Total images: {len(left_test_files) + len(right_test_files)}')
print(f'Combined MRE: {combined_metrics["mre"]:.2f}mm')
print(f'Combined STD: {combined_metrics["std"]:.2f}mm')
print("="*60)

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