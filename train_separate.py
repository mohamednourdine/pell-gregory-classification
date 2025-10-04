import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Optimize CPU threading for maximum performance
import os
if 'OMP_NUM_THREADS' in os.environ:
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    print(f"PyTorch using {torch.get_num_threads()} threads for CPU training")

from scipy import ndimage
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn  # for heatmaps
import numpy as np
import PIL
import argparse
import shutil
import random
import time
from pathlib import Path
import gc  # For garbage collection

from utilities.common_utils import *
from utilities.plotting import *
from utilities.eval_utils import get_accuracy_metrics, get_radial_errors_mm_for_image
from model.unet_model import UNet
from utilities.landmark_utils import LandmarkDataset, get_max_heatmap_activation, radial_errors_batch

ORIG_IMAGE_SIZE = np.array([ORIG_IMAGE_X, ORIG_IMAGE_Y])  # WxH
random_id = int(random.uniform(0, 99999999))

def get_predicted_landmarks(pred_heatmaps, gauss_sigma):
    """Extract predicted landmarks from heatmaps"""
    n_landmarks = pred_heatmaps.shape[0]
    heatmap_y, heatmap_x = pred_heatmaps.shape[1:]
    pred_landmarks = np.zeros((n_landmarks, 2))
    max_activations = np.zeros(n_landmarks)
    
    for i in range(n_landmarks):
        max_activation, pred_yx = get_max_heatmap_activation(pred_heatmaps[i], gauss_sigma)
        rescale = np.array([ORIG_IMAGE_Y, ORIG_IMAGE_X]) / np.array([heatmap_y, heatmap_x])
        pred_yx = np.around(pred_yx * rescale)
        pred_landmarks[i] = pred_yx[::-1]  # Convert y,x to x,y
        max_activations[i] = max_activation
    
    return pred_landmarks, max_activations

def train_separate_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_PATH', type=str, default='data/dataset/resized/256x256', help='Define the root path to the dataset.')
    parser.add_argument('--MODEL_PATH', type=str, default='trained', help='Path where the model checkpoints will be saved after it has been trained.')
    parser.add_argument('--MODEL_NAME', type=str, default='1')
    parser.add_argument('--EXPERIMENT_NAME', type=str, default='experiment')
    parser.add_argument('--SIDE', type=str, required=True, choices=['left', 'right'], help='Which side to train: left (37-38) or right (47-48)')
    parser.add_argument('--MODEL', type=str, default='unet')
    parser.add_argument('--FILTERS', type=int, default=32)
    parser.add_argument('--DOWN_DROP', type=str, default='0.4,0.4,0.4,0.4')
    parser.add_argument('--UP_DROP', type=str, default='0.4,0.4,0.4,0.4')
    parser.add_argument('--BATCH_SIZE', type=int, default=8)
    parser.add_argument('--IMAGE_SIZE', type=int, default=256)
    parser.add_argument('--GAUSS_SIGMA', type=float, default=5.0)
    parser.add_argument('--GAUSS_AMPLITUDE', type=float, default=1000.0)
    parser.add_argument('--USE_ELASTIC_TRANS', type=bool, default=False)
    parser.add_argument('--USE_AFFINE_TRANS', type=bool, default=False)
    parser.add_argument('--USE_HORIZONTAL_FLIP', type=bool, default=False)
    parser.add_argument('--ELASTIC_SIGMA', type=float, default=7.0)
    parser.add_argument('--ELASTIC_ALPHA', type=float, default=1.0)
    parser.add_argument('--LEARN_RATE', type=float, default=1e-3)
    parser.add_argument('--WEIGHT_DECAY', type=float, default=0.0)
    parser.add_argument('--OPTIM_PATIENCE', type=int, default=15)
    parser.add_argument('--EPOCHS', type=int, default=60)
    parser.add_argument('--VALID_RATIO', type=float, default=0.15)
    parser.add_argument('--SAVE_EPOCHS', type=str, default='')
    parser.add_argument('--VAL_MRE_STOP', type=float, default=0.5, help='The system stops training if validation MRE drops below the specified value.')

    args = parser.parse_args()
    
    # Convert string lists to actual lists
    args.DOWN_DROP = [float(x) for x in args.DOWN_DROP.split(',')]
    args.UP_DROP = [float(x) for x in args.UP_DROP.split(',')]
    if args.SAVE_EPOCHS:
        args.SAVE_EPOCHS = [int(x) for x in args.SAVE_EPOCHS.split(',')]
    else:
        args.SAVE_EPOCHS = []

    # Set paths based on side
    if args.SIDE == 'left':
        side_name = '37-38-PELLGREGORY'
        print(f'Training {args.SIDE} side model for {side_name}')
    else:
        side_name = '47-48-PELLGREGORY'
        print(f'Training {args.SIDE} side model for {side_name}')
    
    # Setup paths
    data_path = Path('data/dataset/resized')
    images_path = data_path / side_name
    train_annotations_path = data_path / 'annotations' / side_name / 'train'
    
    # Get training and validation files
    train_images_path = images_path / 'train'
    test_images_path = images_path / 'test'
    
    train_fnames = list_files(train_images_path)
    test_fnames = list_files(test_images_path)
    
    # Split training into train/validation
    num_train = len(train_fnames)
    num_valid = int(num_train * args.VALID_RATIO)
    
    indices = list(range(num_train))
    random.shuffle(indices)
    
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]
    
    train_fnames_subset = [train_fnames[i] for i in train_indices]
    valid_fnames_subset = [train_fnames[i] for i in valid_indices]
    
    print(f'Number of train images: {len(train_fnames_subset)}, Number of validation images: {len(valid_fnames_subset)}')
    
    # Setup transformations
    num_workers = 0
    elastic_trans = None
    affine_trans = None
    
    if args.USE_ELASTIC_TRANS:
        elastic_trans = ElasticTransform(args.ELASTIC_SIGMA, args.ELASTIC_ALPHA)
    
    if args.USE_AFFINE_TRANS:
        angle = 0.05
        scales = [0.95, 1.05]
        tx, ty = 0.03, 0.03
        affine_trans = AffineTransform(angle, scales, tx, ty)

    # Create datasets
    train_ds = LandmarkDataset(
        train_fnames_subset, train_annotations_path,
        args.GAUSS_SIGMA, args.GAUSS_AMPLITUDE,
        elastic_trans=elastic_trans,
        affine_trans=affine_trans, 
        horizontal_flip=args.USE_HORIZONTAL_FLIP
    )

    valid_ds = LandmarkDataset(
        valid_fnames_subset, train_annotations_path,
        args.GAUSS_SIGMA, args.GAUSS_AMPLITUDE
    )

    # Create data loaders
    train_dl = DataLoader(train_ds, args.BATCH_SIZE, shuffle=True, num_workers=num_workers)
    valid_dl = DataLoader(valid_ds, args.BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # Setup device and model
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Graphic Cart Used for the experiment: {device}')

    if args.MODEL == 'unet':
        net = UNet(in_ch=3, out_ch=N_LANDMARKS, down_drop=args.DOWN_DROP, up_drop=args.UP_DROP)
        print(f'Using U-Net model with {N_LANDMARKS} landmarks for {args.SIDE} side')
    else:
        raise ValueError(f"Unknown model type: {args.MODEL}")
       
    net.to(device)

    # Setup optimizer and loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.LEARN_RATE, weight_decay=args.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.OPTIM_PATIENCE)

    # Setup model saving path
    model_save_path = Path(args.MODEL_PATH)
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    print(f'Training started at: {time.ctime()}')
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.EPOCHS + 1):
        epoch_start_time = time.time()
        
        # Training phase
        net.train()
        trained_examples = 0
        train_loss, train_mre, train_sdr_2mm, train_sdr_2_5mm, train_sdr_3mm, train_sdr_4mm = 0, 0, 0, 0, 0, 0
        
        print(f'Starting training loop with {len(train_dl)} batches...')
        
        for batch_idx, (images, true_heatmaps, _) in enumerate(train_dl):
            if (batch_idx + 1) % 10 == 1:
                print(f'Processing batch {batch_idx + 1}/{len(train_dl)}')
            
            images = images.to(device)
            true_heatmaps = true_heatmaps.to(device)
            
            optimizer.zero_grad()
            pred_heatmaps = net(images)
            loss = criterion(pred_heatmaps, true_heatmaps)
            loss.backward()
            optimizer.step()
            
            # Metrics (using historical calculation method)
            actual_bs = images.shape[0]
            train_loss += loss * actual_bs  # Weighted by batch size
            trained_examples += actual_bs

            radial_errors = radial_errors_batch(pred_heatmaps, true_heatmaps, args.GAUSS_SIGMA)
            mre = np.mean(radial_errors)
            train_mre += mre * actual_bs
            train_sdr_2mm += np.sum(radial_errors < 2)
            train_sdr_2_5mm += np.sum(radial_errors < 2.5)
            train_sdr_3mm += np.sum(radial_errors < 3)
            train_sdr_4mm += np.sum(radial_errors < 4)
            
            # Clear memory
            del images, true_heatmaps, pred_heatmaps, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        train_loss_avg = train_loss / trained_examples
        train_mre_avg = train_mre / trained_examples
        train_sdr_2mm_avg = train_sdr_2mm / (trained_examples * N_LANDMARKS)
        train_sdr_2_5mm_avg = train_sdr_2_5mm / (trained_examples * N_LANDMARKS)
        train_sdr_3mm_avg = train_sdr_3mm / (trained_examples * N_LANDMARKS)
        train_sdr_4mm_avg = train_sdr_4mm / (trained_examples * N_LANDMARKS)
        
        # Validation phase
        net.eval()
        val_loss, val_mre, val_sdr_2mm, val_sdr_2_5mm, val_sdr_3mm, val_sdr_4mm = 0, 0, 0, 0, 0, 0
        val_examples = 0
        
        with torch.no_grad():
            for images, true_heatmaps, _ in valid_dl:
                images = images.to(device)
                true_heatmaps = true_heatmaps.to(device)
                
                pred_heatmaps = net(images)
                loss = criterion(pred_heatmaps, true_heatmaps)

                actual_bs = images.shape[0]
                val_loss += loss * actual_bs
                val_examples += actual_bs

                radial_errors = radial_errors_batch(pred_heatmaps, true_heatmaps, args.GAUSS_SIGMA)
                mre = np.mean(radial_errors)
                val_mre += mre * actual_bs
                val_sdr_2mm += np.sum(radial_errors < 2)
                val_sdr_2_5mm += np.sum(radial_errors < 2.5)
                val_sdr_3mm += np.sum(radial_errors < 3)
                val_sdr_4mm += np.sum(radial_errors < 4)
        
        val_loss_avg = val_loss / val_examples
        val_mre_avg = val_mre / val_examples
        val_sdr_2mm_avg = val_sdr_2mm / (val_examples * N_LANDMARKS)
        val_sdr_2_5mm_avg = val_sdr_2_5mm / (val_examples * N_LANDMARKS)
        val_sdr_3mm_avg = val_sdr_3mm / (val_examples * N_LANDMARKS)
        val_sdr_4mm_avg = val_sdr_4mm / (val_examples * N_LANDMARKS)
        
        epoch_duration = time.time() - epoch_start_time
        
        # Print results (matching old format exactly)
        print(f"Epoch: {epoch}, train_loss: {train_loss_avg:.5f}, train_MRE: {train_mre_avg:.2f}, "
              f"train_SDR_2mm: {train_sdr_2mm_avg:.5f}, train_SDR_2_5mm: {train_sdr_2_5mm_avg:.5f}, "
              f"train_SDR_3mm: {train_sdr_3mm_avg:.5f}, train_SDR_4mm: {train_sdr_4mm_avg:.5f}, ")
        print(f"Duration: {epoch_duration:.0f} seconds")
        print(f"val_loss: {val_loss_avg:.5f}, val_MRE: {val_mre_avg:.2f}, "
              f"val_SDR_2mm: {val_sdr_2mm_avg:.5f} val_SDR_2_5mm: {val_sdr_2_5mm_avg:.5f} "
              f"val_SDR_3mm: {val_sdr_3mm_avg:.5f} val_SDR_4mm: {val_sdr_4mm_avg:.5f}")
        print("_" * 76)
        
        # Save model if validation improved or at specified epochs  
        if val_loss_avg < best_val_loss * 0.9999 or epoch in args.SAVE_EPOCHS:
            best_val_loss = min(best_val_loss, val_loss_avg)
            model_path = model_save_path / f'{args.MODEL_NAME}.pth'
            # Ensure the directory exists
            model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f'Saving model checkpoint to {model_path}.')
            torch.save(net, model_path)
        
        # Update learning rate
        scheduler.step(val_loss_avg)
        
        # Early stopping check
        if args.VAL_MRE_STOP is not None and val_mre_avg < args.VAL_MRE_STOP:
            print(f'Stopping experiment due to validation MRE below {args.VAL_MRE_STOP}.')
            break
    
    print(f'Training started at: {time.ctime()}')
    print(f'Training ended at: {time.ctime()}')

if __name__ == '__main__':
    train_separate_model()