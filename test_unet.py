import torch
import numpy as np
from model.unet_model import UNet

print("Testing UNet with actual dataset configuration...")

# Test UNet with proper dimensions - matching train_separate.py exactly
net = UNet(in_ch=3, out_ch=5, down_drop=[0.0, 0.0, 0.0, 0.0], up_drop=[0.0, 0.0, 0.0, 0.0])

# Test with a sample input that matches what the dataset produces
# The dataset produces 3-channel images of size 256x256
batch_size = 16  # Same as training
test_input = torch.randn(batch_size, 3, 256, 256)

print(f"Input shape: {test_input.shape}")

try:
    output = net(test_input)
    print(f"Output shape: {output.shape}")
    print("UNet test successful!")
    print(f"Expected output shape: [{batch_size}, 5, 256, 256]")
    
    # Test forward pass statistics
    print(f"Output min: {output.min():.4f}, max: {output.max():.4f}, mean: {output.mean():.4f}")
    
except Exception as e:
    print(f"UNet test failed: {e}")
    import traceback
    traceback.print_exc()