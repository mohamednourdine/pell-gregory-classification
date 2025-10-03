import PIL.Image
import numpy as np
import os
from torchvision import transforms

# Test loading an actual image from the dataset
image_path = "/home/ubuntu/pell-gregory-classification/data/dataset/resized/37-38-PELLGREGORY/train/son-1.png"

if os.path.exists(image_path):
    print(f"Loading image: {image_path}")
    
    # Mimic what the dataset does
    x = PIL.Image.open(image_path).convert('L')
    print(f"Original PIL image size: {x.size}")
    
    x = np.array(x)
    print(f"Numpy array shape: {x.shape}")
    
    # Add channel dimension
    x = np.expand_dims(x, 2)
    print(f"After expand_dims: {x.shape}")
    
    # Convert to tensor
    x = transforms.ToTensor()(x)
    print(f"After ToTensor: {x.shape}")
    
    # Repeat to 3 channels
    x = transforms.Lambda(lambda x: x.repeat(3, 1, 1))(x)
    print(f"After repeat to 3 channels: {x.shape}")
    
    # Add batch dimension
    x = x.unsqueeze(0)
    print(f"After adding batch dimension: {x.shape}")
    
else:
    print(f"Image not found: {image_path}")
    print("Let's check what files exist:")
    base_dir = "/home/ubuntu/pell-gregory-classification/data/dataset/resized/37-38-PELLGREGORY"
    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir):
            print(f"Directory: {root}")
            print(f"Files: {files[:5]}...")  # Show first 5 files
            break