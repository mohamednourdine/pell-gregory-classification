import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from skimage import io, transform, img_as_float
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms, models  # add models to the list
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import make_grid
import time
import random
import csv

# ignore harmless warnings
import warnings

warnings.filterwarnings("ignore")

from utilities.plotting import *

# with open(TXT_PATH) as csvfile:
#      print("\n".join([x.split(",")[1] for x in file1.read().split("\n")]))
def merge(list1, list2):
    merged_list = [tuple([int(float(list1[i])), int(float(list2[i]))]) for i in range(0, len(list1))]
    return merged_list


# import sample coordinates from text as tuples
def extract_labels_from_txt(path):
    with open(path, "r") as f:
        # only first 19 are actual coords in dataset label files
        coords_raw = f.readlines()[:19]
        coords_raw = [tuple([int(float(s)) for s in t.split(",")]) for t in coords_raw]
        return coords_raw


def extract_cordinate_from_cvs(path):
    with open(path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        x_cordinates = []
        y_cordinates = []
        index = 0
        for row in readCSV:
            if (index == 1):
                y_cords = row[21:40]
                x_cords = row[40:]
                #                 print(merge(x_cords, y_cords))
                cordinates = merge(x_cords, y_cords)
                break
            index += 1
    return cordinates


def print_image(img, labels):
    print(img.shape)
    plt.rcParams["figure.figsize"] = [32, 18]
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 1, 1)
    ax1.imshow(img, cmap="gray")
    # also plot resized image for later
    orig_y, orig_x = img.shape[:2]
    SCALE = 15

    # for rescale, use same target for both x&y axis
    rescaled_img = transform.resize(img, (orig_y / SCALE, orig_y / SCALE))
    ax2.imshow(rescaled_img, cmap="gray")

    for c in coords_raw:
        # add patches to original image
        # could also just plt.scatter() but less control then
        ax1.add_patch(plt.Circle(c, 5, color='r'))
        # and rescaled marks to resized images
        x, y = c
        x = int(x * (orig_y * 1.0 / orig_x) / SCALE)
        y = int(y / SCALE)
        ax2.add_patch(plt.Circle((x, y), 1, color='g'))

    plt.show()


def display_image_and_cord(image_number, img_path, cord_path):
    data = []
    target = []
    for i, fi in enumerate(os.listdir(img_path)):
        if i < image_number:
            loop_img = io.imread(img_path + fi, as_gray=True)
            lf = fi[:-4] + ".txt"
            loop_labels = extract_labels_from_txt(cord_path + lf)

            loop_labels = (np.array(loop_labels))
            print(loop_img)
            print_image(loop_img, loop_labels)



# Take a look at one of the image samples and labels

# NOTE: THE IMAGE FOLDERS HAS BEEN MODIFIED AND SEPERATED INTO TRAIN AND TEST FOLDERS SETS
SAMPLE_PATH = "data/images/128/test1/151.bmp"
TXT_PATH = "logs/test1/ensemble/Ensemble/predictions/1.csv"
# import sample image
img = io.imread(SAMPLE_PATH, as_gray=True)
print(img.shape)


image_label = extract_cordinate_from_cvs(TXT_PATH)
# plot_imgs(img)
