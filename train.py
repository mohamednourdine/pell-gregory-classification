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
import io
import sys
import time
import gc  # For garbage collection
from pathlib import Path

from utilities.common_utils import *
from utilities.plotting import *
from model.unet_model import UNet
from utilities.landmark_utils import LandmarkDataset

ORIG_IMAGE_SIZE = np.array([ORIG_IMAGE_X, ORIG_IMAGE_Y])  # WxH
random_id = int(random.uniform(0, 99999999))

parser = argparse.ArgumentParser()
parser.add_argument('--DATA_PATH', type=str, default='./data', help='Define the root path to the dataset.')
parser.add_argument('--MODEL_PATH', type=str, default='./trained', help='Path where the model checkpoints will be saved after it has been trained.')

parser.add_argument('--MODEL_NAME', type=str, default=f'model_{random_id}')
parser.add_argument('--EXPERIMENT_NAME', type=str, default=f'exp_{random_id}')
parser.add_argument('--MODEL', type=str, default='unet')
parser.add_argument('--FILTERS', type=lambda layers: [int(layer) for layer in layers.split(',')], default='64,128,256,512,1024')
parser.add_argument('--DOWN_DROP', type=lambda layers: [float(layer) for layer in layers.split(',')], default='0.4,0.4,0.4,0.4')
parser.add_argument('--UP_DROP', type=lambda layers: [float(layer) for layer in layers.split(',')], default='0.4,0.4,0.4,0.4')
parser.add_argument('--BATCH_SIZE', type=int, default=64)
parser.add_argument('--IMAGE_SIZE', type=int, default=256)
parser.add_argument('--GAUSS_SIGMA', type=float, default=5.0)
parser.add_argument('--GAUSS_AMPLITUDE', type=float, default=1000.0)
parser.add_argument('--USE_ELASTIC_TRANS', type=bool, default=False)
parser.add_argument('--USE_AFFINE_TRANS', type=bool, default=False)
parser.add_argument('--USE_HORIZONTAL_FLIP', type=bool, default=False) #Laterally Rotated Out-of-distribution Data
parser.add_argument('--ELASTIC_SIGMA', type=float, default=10.0)  #Elastically Distorted Out-of-distribution Data
parser.add_argument('--ELASTIC_ALPHA', type=float, default=15.0)
parser.add_argument('--LEARN_RATE', type=float, default=1e-3)
parser.add_argument('--WEIGHT_DECAY', type=float, default=0.0)
parser.add_argument('--OPTIM_PATIENCE', type=float, default=15)
parser.add_argument('--EPOCHS', type=int, default=200)
parser.add_argument('--VALID_RATIO', type=float, default=0.15) #Validation split using a ratio of 85:15
parser.add_argument('--SAVE_EPOCHS', type=lambda epochs: [float(epoch) for epoch in epochs.split(',')], default=None)
parser.add_argument('--VAL_MRE_STOP', type=float, default=None, help='The system stops training if validation MRE drops below the specified value.')
args = parser.parse_args()

print(f'Training unified model {args.MODEL_NAME}')

# Data paths for both left and right landmarks
path = Path(args.DATA_PATH)
model_path = Path(args.MODEL_PATH) if args.MODEL_PATH is not None else path / 'models'
model_path.mkdir(parents=True, exist_ok=True)

# Paths for left side (37-38-PELLGREGORY)
left_annotations_path = path / f'dataset/resized/annotations/37-38-PELLGREGORY/train'
left_train_path = path / f'dataset/resized/37-38-PELLGREGORY/train'

# Paths for right side (47-48-PELLGREGORY)
right_annotations_path = path / f'dataset/resized/annotations/47-48-PELLGREGORY/train'
right_train_path = path / f'dataset/resized/47-48-PELLGREGORY/train'

# Get file lists for both sides
left_fnames = list_files(left_train_path)
right_fnames = list_files(right_train_path)

print(f'Left side images: {len(left_fnames)}, Right side images: {len(right_fnames)}')
print(f'Total combined images: {len(left_fnames) + len(right_fnames)}')

# Calculate validation split for combined dataset
total_files = len(left_fnames) + len(right_fnames)
n_valid = int(args.VALID_RATIO * total_files)

# Split each side proportionally for validation
left_n_valid = max(1, int(args.VALID_RATIO * len(left_fnames)))  # Ensure at least 1
right_n_valid = max(1, int(args.VALID_RATIO * len(right_fnames)))  # Ensure at least 1

left_train_fnames = left_fnames[:-left_n_valid] if left_n_valid < len(left_fnames) else left_fnames[:-1]
left_valid_fnames = left_fnames[-left_n_valid:] if left_n_valid < len(left_fnames) else left_fnames[-1:]

right_train_fnames = right_fnames[:-right_n_valid] if right_n_valid < len(right_fnames) else right_fnames[:-1]
right_valid_fnames = right_fnames[-right_n_valid:] if right_n_valid < len(right_fnames) else right_fnames[-1:]

print(f'Training: Left={len(left_train_fnames)}, Right={len(right_train_fnames)}, Total={len(left_train_fnames) + len(right_train_fnames)}')
print(f'Validation: Left={len(left_valid_fnames)}, Right={len(right_valid_fnames)}, Total={len(left_valid_fnames) + len(right_valid_fnames)}')

num_workers = 0
elastic_trans = None
affine_trans = None
if args.USE_ELASTIC_TRANS:
    elastic_trans = ElasticTransform(sigma=args.ELASTIC_SIGMA, alpha=args.ELASTIC_ALPHA)
if args.USE_AFFINE_TRANS:
    angle = 5
    scales = [0.95, 1.05]
    tx, ty = 0.03, 0.03
    affine_trans = AffineTransform(angle, scales, tx, ty)

# Create unified datasets
train_ds = UnifiedLandmarkDataset(
    left_train_fnames, right_train_fnames,
    left_annotations_path, right_annotations_path,
    args.GAUSS_SIGMA, args.GAUSS_AMPLITUDE,
    elastic_trans=elastic_trans,
    affine_trans=affine_trans, 
    horizontal_flip=args.USE_HORIZONTAL_FLIP
)

valid_ds = UnifiedLandmarkDataset(
    left_valid_fnames, right_valid_fnames,
    left_annotations_path, right_annotations_path,
    args.GAUSS_SIGMA, args.GAUSS_AMPLITUDE
)

# Create data loaders
train_dl = DataLoader(train_ds, args.BATCH_SIZE, shuffle=True, num_workers=num_workers)
valid_dl = DataLoader(valid_ds, args.BATCH_SIZE, shuffle=False, num_workers=num_workers)


"""
CUDA is a parallel computing platform and programming model that makes using a GPU 
for general purpose computing simple and elegant. 

"""

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'Graphic Cart Used for the experiment: {device}')

if args.MODEL == 'unet':
    net = UNet(in_ch=3, out_ch=N_LANDMARKS_PER_SIDE, down_drop=args.DOWN_DROP, up_drop=args.UP_DROP)
    print(f'Using U-Net model with {N_LANDMARKS_PER_SIDE} landmarks')
else:
    raise ValueError(f"Unknown model type: {args.MODEL}")
   
net.to(device)
# count_parameters(net)
# Optimizer + loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=args.LEARN_RATE, weight_decay=args.WEIGHT_DECAY)
"""
Reduce learning rate when a metric has stopped improving. Models often benefit from reducing the 
learning rate by a factor of 2-10 once learning stagnates. This callback monitors a quantity and if 
no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.
"""
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.OPTIM_PATIENCE)


start_time = time.time()

trained_losses = []
trained_mre = []
#Success Detection Rate
trained_sdr_4mm = [] 

def train():
    trained_examples = 0
    train_loss, train_mre, train_sdr_2mm,train_sdr_2_5mm,train_sdr_3mm,train_sdr_4mm = 0, 0, 0, 0, 0, 0
    
  #  print(train_dl)    
    '''We train the model and continue with the extraction of necessary information'''
    net.train() 
    
    print(f"Starting training loop with {len(train_dl)} batches...")
    batch_count = 0

    for imgs, true_points, _ in train_dl:
        batch_count += 1
        if batch_count % 10 == 1:  # Print progress every 10 batches
            print(f"Processing batch {batch_count}/{len(train_dl)}")
        
        imgs = imgs.to(device)
        true_points = true_points.to(device)
       
        optimizer.zero_grad()
        pred_heatmaps = net(imgs)
        loss = criterion(pred_heatmaps, true_points)  # The loss of the predicted heatmaps, obtained as the MSELoss of pred_h and true_points. 
        loss.backward()
        optimizer.step()
        
        # Metrics
        actual_bs = imgs.shape[0]
        train_loss += loss * actual_bs  # Weighted by batch size
        trained_examples += actual_bs

        radial_errors = radial_errors_batch(pred_heatmaps, true_points, args.GAUSS_SIGMA)
        mre = np.mean(radial_errors)
        train_mre += mre * actual_bs
        train_sdr_2mm += np.sum(radial_errors < 2)
        train_sdr_2_5mm += np.sum(radial_errors < 2.5)
        train_sdr_3mm += np.sum(radial_errors < 3)
        train_sdr_4mm += np.sum(radial_errors < 4)
        
        # Clean up references to free memory
        del radial_errors

    trained_losses.append(f'{train_loss / trained_examples:{4}.{4}}')
    trained_mre.append(f'{train_mre / trained_examples:{4}.{4}}')
    trained_sdr_4mm.append(f'{train_sdr_4mm / (trained_examples * N_LANDMARKS):{4}.{4}}')
    
    print(f'Epoch: {e}, train_loss: {train_loss / trained_examples:{4}.{4}}, '
          f'train_MRE: {train_mre / trained_examples:{4}.{4}}, '
          f'train_SDR_2mm: {train_sdr_2mm / (trained_examples * N_LANDMARKS):{4}.{4}}, '
          f'train_SDR_2_5mm: {train_sdr_2_5mm / (trained_examples * N_LANDMARKS):{4}.{4}}, '
          f'train_SDR_3mm: {train_sdr_3mm / (trained_examples * N_LANDMARKS):{4}.{4}}, '
          f'train_SDR_4mm: {train_sdr_4mm / (trained_examples * N_LANDMARKS):{4}.{4}}, ', end='')
    print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed 
      
    return train_loss / trained_examples


def validate():

    val_loss, val_mre, val_sdr_2mm, val_sdr_2_5mm, val_sdr_3mm, val_sdr_4mm = 0, 0, 0, 0, 0, 0
    val_examples = 0
    net.eval()
    with torch.no_grad():
        for imgs, true_points, _ in valid_dl:
            imgs = imgs.to(device)
            true_points = true_points.to(device)

            pred_heatmaps = net(imgs)
            loss = criterion(pred_heatmaps, true_points)

            actual_bs = imgs.shape[0]
            val_loss += loss * actual_bs
            val_examples += actual_bs

            radial_errors = radial_errors_batch(pred_heatmaps, true_points, args.GAUSS_SIGMA)
            mre = np.mean(radial_errors)
            val_mre += mre * actual_bs
            val_sdr_2mm += np.sum(radial_errors < 2)
            val_sdr_2_5mm += np.sum(radial_errors < 2.5)
            val_sdr_3mm += np.sum(radial_errors < 3)
            val_sdr_4mm += np.sum(radial_errors < 4)
            
            # Clean up references to free memory
            del radial_errors

            # plot_imgs(imgs)

    print(f'val_loss: {val_loss / val_examples:{4}.{4}}, '
          f'val_MRE: {val_mre / val_examples:{4}.{4}}, '
          f'val_SDR_2mm: {val_sdr_2mm / (val_examples * N_LANDMARKS):{4}.{4}}',
          f'val_SDR_2_5mm: {val_sdr_2_5mm / (val_examples * N_LANDMARKS):{4}.{4}}',
          f'val_SDR_3mm: {val_sdr_3mm / (val_examples * N_LANDMARKS):{4}.{4}}',
          f'val_SDR_4mm: {val_sdr_4mm / (val_examples * N_LANDMARKS):{4}.{4}}')

    print("____________________________________________________________________________")

    return val_loss / val_examples, val_sdr_2mm / (val_examples * N_LANDMARKS) ,val_sdr_2_5mm / (val_examples * N_LANDMARKS) ,val_sdr_3mm / (val_examples * N_LANDMARKS), val_sdr_4mm / (val_examples * N_LANDMARKS), val_mre / val_examples


# Loop over epochs
best_val_loss = None
num_bad_epochs = 0
try:

    # Record the start time
    start_time = time.time()

    print(f"Training started at: {time.ctime(start_time)}")

    for e in range(1, args.EPOCHS + 1):
        train_loss = train()
        val_loss, val_sdr_2mm, val_sdr_2_5mm, val_sdr_3mm, val_sdr_4mm, val_mre = validate()
        scheduler.step(val_loss)
        
        # Force garbage collection after each epoch to free memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if args.SAVE_EPOCHS is not None and e in args.SAVE_EPOCHS:
            print(f'Saving model checkpoint (one of {args.SAVE_EPOCHS} requested).')
            with open(model_path / f'{args.MODEL_NAME}_e_{e}.pth', 'wb') as f:
                torch.save(net, f)

        if not best_val_loss or val_loss < best_val_loss * 0.9999:
            num_bad_epochs = 0
            best_val_loss = val_loss
            checkpoint_path = model_path / f'{args.MODEL_NAME}.pth'
            print(f'Saving model checkpoint to {checkpoint_path}.')
            with open(checkpoint_path, 'wb') as f:
                torch.save(net, f)

        if args.VAL_MRE_STOP is not None and val_mre < args.VAL_MRE_STOP:
            print(
                f'Stopping experiment since val_mre is {val_mre} which is below {args.VAL_MRE_STOP} as specified by program arguments.')
            break

        if val_loss > best_val_loss:
            num_bad_epochs += 1

        if num_bad_epochs > 10:
            print('Stopping experiment due to plateauing.')
            break

    # Record the end time
    end_time = time.time()

    # Calculate the total training time
    training_time = end_time - start_time

    # Print the start and end times, as well as the total training time
    print(f"Training started at: {time.ctime(start_time)}")
    print(f"Training ended at: {time.ctime(end_time)}")
    print(f"Total training time: {training_time:.2f} seconds")

    # plot_training_result(trained_losses, trained_mre, trained_sdr_4mm)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')