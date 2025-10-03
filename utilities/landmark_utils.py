import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import PIL
import shutil
from pathlib import Path
import time
import random

from .common_utils import *

# Parameter for gaussian filter
GAUSSIAN_TRUNCATE = 1.0


def get_annots_for_image(annotations_path, image_path, rescaled_image_size=None, orig_image_size=np.array([ORIG_IMAGE_X, ORIG_IMAGE_Y])):
    '''
    Gets all the annations of an image and return in a simple array format of [[x1,y1], [x2,y2], ...] 
    '''
    # Convert to Path object if it's a string
    if isinstance(image_path, str):
        image_path = Path(image_path)
    
    image_id = image_path.stem
    annots = (annotations_path / f'{image_id}.txt').read_text()
    annots = annots.split('\n')[:N_LANDMARKS_PER_SIDE]  # Each side has 5 landmarks
    annots = [l.split(',') for l in annots]
    annots = [(float(l[0]), float(l[1])) for l in annots];
    annots = np.array(annots)

    '''
    Rescale the images annotations if the images are rescaled before returning the array. 
    '''
    if rescaled_image_size is not None:
        scale = np.array([rescaled_image_size, rescaled_image_size], dtype=float) / orig_image_size  # WxH
        annots = np.around(annots * scale).astype('int32')
    return annots  


def create_true_heatmaps(annots, image_size, amplitude):
    heatmaps = np.zeros((annots.shape[0], image_size, image_size))
    for i, landmark_pos in enumerate(annots):
        try:
            x, y = landmark_pos
            heatmaps[i, y, x] = amplitude  # Swap WxH to HxW
        except:
            pass
    return heatmaps


def reset_heatmap_maximum(heatmap, amplitude):
    '''
    Heatmap maximum value is not equal to the amplitude after the transformation.
    We zero the heatmap and set it to the amplitude at the new maximum position.
    '''
    ind = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    heatmap[:] = 0
    heatmap[ind] = amplitude
    return heatmap


class ArrayToTensor(object):
    def __call__(self, np_array):
        return torch.from_numpy(np_array).float()


class UnifiedLandmarkDataset(Dataset):
    """Unified dataset for both left and right landmarks"""
    
    def __init__(self, image_fnames_left, image_fnames_right, 
                 annotations_path_left, annotations_path_right,
                 gauss_sigma, gauss_amplitude,
                 elastic_trans=None, affine_trans=None, horizontal_flip=False):
        
        # Combine both datasets
        self.combined_data = []
        
        # Add left-side images with side indicator
        for fname in image_fnames_left:
            self.combined_data.append({
                'image_path': fname,
                'annotations_path': annotations_path_left,
                'side': 'left',
                'landmark_offset': 0  # Left landmarks use indices 0-4
            })
        
        # Add right-side images with side indicator  
        for fname in image_fnames_right:
            self.combined_data.append({
                'image_path': fname,
                'annotations_path': annotations_path_right,
                'side': 'right',
                'landmark_offset': 5  # Right landmarks use indices 5-9
            })
        
        self.gauss_sigma = gauss_sigma
        self.gauss_amplitude = gauss_amplitude
        self.elastic_trans = elastic_trans
        self.affine_trans = affine_trans
        self.horizontal_flip = horizontal_flip

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx):
        data_item = self.combined_data[idx]
        
        # Set random seed for reproducible transforms
        seed = int(random.random() * 10000000)
        np.random.seed(seed)

        # Load image
        x = PIL.Image.open(data_item['image_path']).convert('L')
        image_size = x.size[0]
        x = np.array(x)
        
        # Transform flags
        do_affine = False
        do_elastic = False
        do_flip = False
        
        if self.affine_trans is not None:
            do_affine = np.random.uniform() > 0.02
            if do_affine:
                affine_matrix = self.affine_trans.get_matrix(x)
                affine_matrix = np.linalg.inv(affine_matrix)

        if self.elastic_trans is not None:
            do_elastic = True
            if do_elastic:
                x_coords, y_coords, _, _ = self.elastic_trans.get_coordinates(x)
                elastic_trans_coordinates = (y_coords, x_coords)

        if self.horizontal_flip:
            do_flip = np.random.uniform() > 0.5

        # Apply transforms to image
        if self.elastic_trans is not None and do_elastic:
            x = ndimage.interpolation.map_coordinates(x, elastic_trans_coordinates, order=1).reshape(x.shape)
        if self.affine_trans is not None and do_affine:
            x = ndimage.affine_transform(x, affine_matrix, offset=0, order=1)
        if self.horizontal_flip and do_flip:
            x = np.ascontiguousarray(np.flip(x, axis=1))
            
        # Convert to tensor format
        x = np.expand_dims(x, 2)
        x = transforms.ToTensor()(x)
        x = transforms.Lambda(lambda x: x.repeat(3, 1, 1))(x)
        x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)

        # Create unified heatmap with 10 channels (5 left + 5 right)
        y = np.zeros((N_LANDMARKS, image_size, image_size))
        
        if data_item['annotations_path'] is not None:
            # Load annotations for this side (5 landmarks)
            annots = get_annots_for_image(data_item['annotations_path'], data_item['image_path'], 
                                        rescaled_image_size=image_size)
            
            if self.horizontal_flip and do_flip:
                for i in range(annots.shape[0]):
                    annots[i][0] += 2 * (image_size / 2.0 - annots[i][0])
            
            # Create heatmaps for this side's landmarks
            side_heatmaps = create_true_heatmaps(annots, image_size, amplitude=self.gauss_amplitude)
            
            # Apply transforms to heatmaps
            if self.elastic_trans is not None and do_elastic:
                for i in range(side_heatmaps.shape[0]):
                    side_heatmaps[i] = ndimage.interpolation.map_coordinates(
                        side_heatmaps[i], elastic_trans_coordinates, order=1).reshape(side_heatmaps[i].shape)
                    side_heatmaps[i] = reset_heatmap_maximum(side_heatmaps[i], self.gauss_amplitude)

            if self.affine_trans is not None and do_affine:
                for i in range(side_heatmaps.shape[0]):
                    side_heatmaps[i] = ndimage.affine_transform(side_heatmaps[i], affine_matrix, offset=0, order=1)
                    side_heatmaps[i] = reset_heatmap_maximum(side_heatmaps[i], self.gauss_amplitude)

            # Apply gaussian filter
            for i in range(side_heatmaps.shape[0]):
                side_heatmaps[i] = ndimage.gaussian_filter(side_heatmaps[i], sigma=self.gauss_sigma, 
                                                         truncate=GAUSSIAN_TRUNCATE)
            
            # Place in correct positions in unified heatmap
            landmark_start = data_item['landmark_offset']
            landmark_end = landmark_start + N_LANDMARKS_PER_SIDE
            y[landmark_start:landmark_end] = side_heatmaps

        y = torch.from_numpy(y).float()
        return x, y, str(data_item['image_path'])


class LandmarkDataset(Dataset):
    def __init__(self, image_fnames, annotations_path, gauss_sigma, gauss_amplitude,
                 elastic_trans=None, affine_trans=None, horizontal_flip=False):
        self.image_fnames = image_fnames
        if annotations_path == '': annotations_path = None
        self.annotations_path = annotations_path
        self.gauss_sigma = gauss_sigma
        self.gauss_amplitude = gauss_amplitude
        self.elastic_trans = elastic_trans
        self.affine_trans = affine_trans
        self.horizontal_flip = horizontal_flip

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        # Here we get the items individually so that each thread has a different transform when multiprocessing
        
        # With the seed reset (every time), the same set of numbers will appear every time.
        seed = int(random.random() * 10000000)
        np.random.seed(seed)

        # Image, we basicall convert the images to Gray scaled images.
        x = PIL.Image.open(self.image_fnames[idx]).convert('L')
        image_size = x.size[0]
        x = np.array(x)
        
        #Since we have converted the image to one color frame, we do not need to move any axis.
        # x = np.moveaxis(x, -1, 0)

        if self.affine_trans is not None:
            do_affine = np.random.uniform() > 0.02
            if do_affine:
                affine_matrix = self.affine_trans.get_matrix(x)
                affine_matrix = np.linalg.inv(affine_matrix)

        # Elastic transform
        if self.elastic_trans is not None:
            # do_elastic = np.random.uniform() > 0.1
            do_elastic = True
            if do_elastic:
                x_coords, y_coords, _, _ = self.elastic_trans.get_coordinates(x)
                elastic_trans_coordinates = (y_coords, x_coords)

        if self.horizontal_flip:
            do_flip = np.random.uniform() > 0.5

        # x transforms
        if self.elastic_trans is not None and do_elastic:
            x = ndimage.interpolation.map_coordinates(x, elastic_trans_coordinates, order=1).reshape(x.shape)
        if self.affine_trans is not None and do_affine:
            x = ndimage.affine_transform(x, affine_matrix, offset=0, order=1)
        if self.horizontal_flip and do_flip:
            x = np.ascontiguousarray(np.flip(x, axis=1))
        x = np.expand_dims(x, 2)
        x = transforms.ToTensor()(x)
        x = transforms.Lambda(lambda x: x.repeat(3, 1, 1))(x)
        x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)

        # Labels
        # Annotations
        if self.annotations_path is not None:
            annots = get_annots_for_image(self.annotations_path, self.image_fnames[idx], rescaled_image_size=image_size)
            if self.horizontal_flip and do_flip:
                for i in range(annots.shape[0]):
                    annots[i][0] += 2 * (image_size / 2.0 - annots[i][0])
            # Create unfiltered heatmaps
            y = create_true_heatmaps(annots, image_size, amplitude=self.gauss_amplitude)

            if self.elastic_trans is not None and do_elastic:
                for i in range(y.shape[0]):
                    # Multi-dimensional image processing (scipy.ndimage)
                    y[i] = ndimage.interpolation.map_coordinates(y[i], elastic_trans_coordinates, order=1).reshape(y[i].shape)
                    y[i] = reset_heatmap_maximum(y[i], self.gauss_amplitude)

            if self.affine_trans is not None and do_affine:
                for i in range(y.shape[0]):
                    y[i] = ndimage.affine_transform(y[i], affine_matrix, offset=0, order=1)
                    y[i] = reset_heatmap_maximum(y[i], self.gauss_amplitude)

            # Apply gaussian filter
            for i in range(y.shape[0]):
                y[i] = ndimage.gaussian_filter(y[i], sigma=self.gauss_sigma, truncate=GAUSSIAN_TRUNCATE)
        else:
            y = np.array([0])

        y = torch.from_numpy(y).float()
        return x, y, str(self.image_fnames[idx])


'''
def get_max_yx(tensor):
    max_y, argmax_y = tensor.max(dim=0)
    _, argmax_x = max_y.max(dim=0)
    max_yx = (argmax_y[argmax_x.item()].item(), argmax_x.item())
    return np.array(max_yx)
'''


def np_max_yx(arr):
    argmax_0 = np.argmax(arr, axis=0)
    max_0 = arr[argmax_0, np.arange(arr.shape[1])]
    argmax_1 = np.argmax(max_0)
    max_yx_pos = np.array([argmax_0[argmax_1], argmax_1])
    max_val = arr[max_yx_pos[0], max_yx_pos[1]]
    return max_val, max_yx_pos


def get_max_heatmap_activation(tensor, gauss_sigma):
    # Handle both tensor and numpy array inputs
    if torch.is_tensor(tensor):
        array = tensor.cpu().detach().numpy()
    else:
        array = tensor
    activations = ndimage.gaussian_filter(array, sigma=gauss_sigma, truncate=GAUSSIAN_TRUNCATE)
    max_val, max_pos = np_max_yx(activations)
    return max_val, max_pos


def radial_errors_calcalation(pred, targ, gauss_sigma, orig_image_x=ORIG_IMAGE_X, orig_image_y=ORIG_IMAGE_Y):
    example_radial_errors = np.zeros(N_LANDMARKS)
    heatmap_y, heatmap_x = pred.shape[1:]
    for i in range(N_LANDMARKS):
        max_pred_act, pred_yx = get_max_heatmap_activation(pred[i], gauss_sigma)
        _, true_yx = get_max_heatmap_activation(targ[i], gauss_sigma)

        # Rescale to original resolution
        rescale = np.array([ORIG_IMAGE_Y, ORIG_IMAGE_X]) / np.array([heatmap_y, heatmap_x])
        pred_yx = np.around(pred_yx * rescale) / PIXELS_PER_MM
        true_yx = np.around(true_yx * rescale) / PIXELS_PER_MM
        example_radial_errors[i] = np.linalg.norm(pred_yx - true_yx)
    return example_radial_errors


def radial_errors_batch(preds, targs, gauss_sigma):
    assert (preds.shape[0] == targs.shape[0])
    batch_size = preds.shape[0]
    batch_radial_errors = np.zeros((batch_size, N_LANDMARKS))
    
    # Convert to numpy if needed, ensuring no gradients are attached
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(targs):
        targs = targs.detach().cpu().numpy()
    
    for i in range(batch_size):
        batch_radial_errors[i] = radial_errors_calcalation(preds[i], targs[i], gauss_sigma)
    return batch_radial_errors

def aug_and_save(img, img_name, label, aug_list, base_path):
    kp = [list_to_kp(label)]
    img = shrink.augment_image(img)
    kp = shrink.augment_keypoints(kp)
    img_save_name = base_path + "/" + img_name + "_aug{}".format(0)
    io.imsave(img_save_name + ".bmp", img)
    with open(img_save_name + ".txt", "w") as lf:
            stringified = [str(tup) for tup in kp_to_list(kp[0].keypoints)]
            stringified = [s.replace("(", "").replace(")","") for s in stringified]
            lf.write("\n".join(stringified))
    for i, aug in enumerate(aug_list):
        img_aug = aug.augment_image(img)
        kp_aug = aug.augment_keypoints(kp)
        # save img:
        img_save_name = base_path + "/" + img_name + "_aug{}".format(i+1)
        io.imsave(img_save_name + ".bmp", img_aug)
        # save labelfile:
        print(img_save_name)
        with open(img_save_name + ".txt", "w") as lf:
            stringified = [str(tup) for tup in kp_to_list(kp_aug[0].keypoints)]
            stringified = [s.replace("(", "").replace(")","") for s in stringified]
            lf.write("\n".join(stringified))