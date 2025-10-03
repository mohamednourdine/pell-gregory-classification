import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from scipy import ndimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import PIL
import shutil
from pathlib import Path
import time
import random

from .common_utils import *


def get_true_landmarks(annotations_path, image_path):
    ''' 
    Returns an array of true landmarks for an image (5 landmarks per side)
    '''
    image_id = image_path.stem     #The stem of the filename identified by the path (i.e. the filename without the final extension).
    annots = (annotations_path / f'{image_id}.txt').read_text()
    annots = annots.split('\n')[:N_LANDMARKS_PER_SIDE]  # 5 landmarks per side
    annots = [l.split(',') for l in annots]
    true_landmarks = [np.array([float(l[1]), float(l[0])]) for l in annots]  # Swap XY to YX order
    return np.array(true_landmarks)


FILE_COL = 'file'


def read_prediction_files_as_df(prediction_files):
    ''' 
    Reads individual prediction files as dataframes and then concatenates them into a single dataframe 
    with all predictions for all images and samples.
    '''
    dataframes = []
    for f in prediction_files:
        dataframes.append(pd.read_csv(f))
    df = pd.concat(dataframes, ignore_index=True)
    df.sort_values(FILE_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_predictions_for_image(df, image_file, n_samples):
    ''' 
    Extracts all of the landmark position and activations samples for the given image from the dataframe
    and the returns them as numpy arrays.
    '''
    image_df = df.loc[df[FILE_COL] == image_file]
    image_df.reset_index(drop=True, inplace=True)
    computed_samples = image_df.shape[0]

    n_samples = min(n_samples, computed_samples)
    activation_samples = np.zeros((n_samples, N_LANDMARKS))
    landmark_samples = np.zeros((n_samples, N_LANDMARKS, 2))

    for i, row in image_df.iterrows():
        if i < n_samples:
            for lm in range(N_LANDMARKS):
                activation_samples[i, lm] = row[f'{lm}_act']
                landmark_samples[i, lm] = np.array([row[f'{lm}_y'], row[f'{lm}_x']])
    return landmark_samples, activation_samples


def get_landmark_prediction_variance(lm_samples):
    n_samples, n_landmarks, _ = lm_samples.shape
    landmark_mean = np.mean(lm_samples, axis=0)
    distances = np.zeros((n_samples, n_landmarks))
    for s in range(n_samples):
        for lm in range(n_landmarks):
            dist = np.linalg.norm(lm_samples[s, lm] - landmark_mean[lm])
            distances[s, lm] = dist
    return np.mean(distances, axis=0)


def get_predicted_landmarks_for_image(landmark_samples):
    ''' 
        Returns the average predicted landmark positions in term of samples and their variance computed
        as the mean distance between landmarks and the mean landmark.
        Takes an array of dimension (n_samples, n_landmarks, 2).
    '''
    landmark_mean = np.mean(landmark_samples, axis=0)
    landmark_var = get_landmark_prediction_variance(landmark_samples)
    return landmark_mean, landmark_var / PIXELS_PER_MM


def get_predicted_activations_for_image(activation_samples):
    ''' Returns the average activation landmark positions and their variance in term of samples.
        Takes an array of dimension (n_samples, n_landmarks).
    '''
    activation_mean = np.mean(activation_samples, axis=0)
    activation_var = np.var(activation_samples, axis=0)
    return activation_mean, activation_var


def radial_error_mm(true, pred):
    ''' 
    Returns the radial error in mms for a single landmark.
    The radial error is the distance between the desired point of impact and actual point of impact, 
    both points projected and measured on an imaginary plane drawn perpendicular to the flight path of the munition.
    '''
    return np.linalg.norm(pred / PIXELS_PER_MM - true / PIXELS_PER_MM)


def get_radial_errors_mm_for_image(true_landmarks, predicted_landmarks):
    ''' 
        Returns an array containing the radial error for each landmark for the image.
        Works with any number of landmarks (5 for single side, 10 for unified model).
    '''
    n_landmarks = len(true_landmarks)
    radial_errors = np.zeros(n_landmarks)
    for lm in range(n_landmarks):
        radial_errors[lm] = radial_error_mm(true_landmarks[lm], predicted_landmarks[lm])
    return radial_errors

def get_radial_errors_mm_for_individual_landmarks(radial_errors):
    '''
        Returns an array containing the radial error for each landmark for the image.
    '''

    print('STATISCAL VALUES ON TEST1')

    for lm in range(N_LANDMARKS):
        sdr_lm = np.array([], dtype = np.float32)

        for errors in np.array(radial_errors):
            sdr_lm = np.append(sdr_lm, errors[lm])

        # for result in np.array(lm):

        print('------------------------------------------------------------------------')

        print(f"Results of L{lm + 1}")
        metric = get_accuracy_metrics(sdr_lm)
        print (metric)
        print_accuracy_metrics(metric)

        # print (sdr_lm)

    return



def get_accuracy_metrics(radial_errors_mm_all):
    '''
    This function Computes the accuracy metrics from radial errors by getting the mean and standard deviation of the
    results obtaine. This results then compaire to the actial values and printed out.
    '''

    # print(f"Count of the frame { len(radial_errors_mm_all)}" )
    mre = radial_errors_mm_all.mean()
    std = radial_errors_mm_all.std()

    sdr_2 = (radial_errors_mm_all < 2.0).mean()
    sdr_2_5 = (radial_errors_mm_all < 2.5).mean()
    sdr_3 = (radial_errors_mm_all < 3.0).mean()
    sdr_4 = (radial_errors_mm_all < 4.0).mean()

    return {'mre': mre, 'std': std,
            'sdr_2': sdr_2, 'sdr_2_5': sdr_2_5,
            'sdr_3': sdr_3, 'sdr_4': sdr_4}


def print_accuracy_metrics(result):
    '''
    Success Detection Rate gives the percentage of predictions within that radius of the ground truth.
    '''
    print(f"Mean Root Error (MRE): {result['mre']:{4}.{4}} mm, Standard Deviation (STD): {result['std']:{4}.{4}} mm\
           \nSuccess Detection Rate\
           \nSDR 2mm: {result['sdr_2']:{4}.{4}}\
           \nSDR 2.5mm: {result['sdr_2_5']:{4}.{4}}\
           \nSDR 3mm: {result['sdr_3']:{4}.{4}}\
           \nSDR 4mm: {result['sdr_4']:{4}.{4}}")


def log_metrics(metrics, metrics_dir, n_samples):
    ''' 
    Logs the computed metrics to a csv file in the metrics subdirectory of the log directory for the model.
    '''
    metrics['samples'] = n_samples
    metrics_df = pd.DataFrame(data=metrics, index=[0])
    metrics_df.to_csv(metrics_dir / f'{n_samples}.csv')


def get_test_predictions_df(model_log_dir):
    """ 
    Load computed model predictions, this function simple reads all the prediction files and return a dataframe of all
    the predictions files.
    """
    prediction_dir = model_log_dir / 'predictions'
    prediction_files = list_files(prediction_dir)
    predictions_df = read_prediction_files_as_df(prediction_files)
    return predictions_df


def get_predictions_for_image_from_unified_df(df, image_name, n_samples):
    """
    Extract prediction samples for a specific image from unified model predictions DataFrame.
    Works with the new format: image_name,sample,landmark,x,y,activation,side
    """
    # Filter predictions for this specific image
    image_predictions = df[df['image_name'] == image_name]
    
    if len(image_predictions) == 0:
        raise ValueError(f"No predictions found for image: {image_name}")
    
    # Determine the side
    if len(image_predictions) > 0:
        side = image_predictions.iloc[0]['side']
    
    # Group by sample to reconstruct landmark arrays
    samples = image_predictions['sample'].unique()[:n_samples]
    n_available_samples = len(samples)
    
    landmark_samples = np.zeros((n_available_samples, N_LANDMARKS_PER_SIDE, 2))
    activation_samples = np.zeros((n_available_samples, N_LANDMARKS_PER_SIDE))
    
    for sample_idx, sample_id in enumerate(samples):
        sample_data = image_predictions[image_predictions['sample'] == sample_id]
        
        for _, row in sample_data.iterrows():
            landmark_idx = row['landmark']
            # Map from unified landmark index to side-specific index
            if side == 'left':
                if landmark_idx >= N_LANDMARKS_PER_SIDE:
                    continue  # Skip right landmarks for left side
                side_landmark_idx = landmark_idx
            else:  # right side
                if landmark_idx < N_LANDMARKS_PER_SIDE:
                    continue  # Skip left landmarks for right side
                side_landmark_idx = landmark_idx - N_LANDMARKS_PER_SIDE
            
            landmark_samples[sample_idx, side_landmark_idx] = [row['y'], row['x']]  # Y, X order
            activation_samples[sample_idx, side_landmark_idx] = row['activation']
    
    return landmark_samples, activation_samples


def get_predictions_for_image_from_df(df, image_name, n_samples):
    """
    Extract prediction samples for a specific image from unified model predictions DataFrame.
    Works with both legacy format and new unified format.
    """
    # Check if this is the new unified format
    if 'sample' in df.columns and 'landmark' in df.columns:
        return get_predictions_for_image_from_unified_df(df, image_name, n_samples)
    
    # Filter predictions for this specific image
    image_predictions = df[df['image_name'] == image_name]
    
    if len(image_predictions) == 0:
        raise ValueError(f"No predictions found for image: {image_name}")
    
    # Determine the side and corresponding landmark indices
    if len(image_predictions) > 0:
        side = image_predictions.iloc[0].get('side', 'unknown')
        if side == 'left':
            landmark_indices = list(range(0, N_LANDMARKS_PER_SIDE))  # 0-4
        elif side == 'right':
            landmark_indices = list(range(N_LANDMARKS_PER_SIDE, N_LANDMARKS))  # 5-9
        else:
            # Fallback: assume all landmarks are available (legacy format)
            landmark_indices = list(range(N_LANDMARKS_PER_SIDE))
    
    # Extract landmark coordinates and activations
    n_available_samples = min(len(image_predictions), n_samples)
    landmark_samples = np.zeros((n_available_samples, N_LANDMARKS_PER_SIDE, 2))
    activation_samples = np.zeros((n_available_samples, N_LANDMARKS_PER_SIDE))
    
    for sample_idx in range(n_available_samples):
        prediction = image_predictions.iloc[sample_idx]
        
        for lm_idx, unified_idx in enumerate(landmark_indices):
            # Extract coordinates (Y, X order)
            y = prediction[f'{unified_idx}_y']
            x = prediction[f'{unified_idx}_x']
            activation = prediction[f'{unified_idx}_act']
            
            landmark_samples[sample_idx, lm_idx] = [y, x]
            activation_samples[sample_idx, lm_idx] = activation
    
    return landmark_samples, activation_samples