import torch
from torch import nn, optim
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),  #kernel size is 3
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0):
        super().__init__()
        self.mp = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(p=drop) if drop != 0 else None
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.mp(x)
        if self.drop != None:
            x = self.drop(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2) #Here to kernel size of 2
        self.drop = nn.Dropout2d(p=drop) if drop != 0 else None
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, x_stored):
        x = self.up(x)
        x = torch.cat([x, x_stored], dim=1)
        if self.drop != None:
            x = self.drop(x)
        x = self.conv(x)
        return x


class AnatomicalConsistencyModule(nn.Module):
    """Module to encourage anatomical consistency between left and right landmarks"""
    def __init__(self, channels=10):
        super().__init__()
        self.channels = channels
        
    def forward(self, landmarks):
        # Split into left (0-4) and right (5-9) landmarks
        left_landmarks = landmarks[:, :5, :, :]
        right_landmarks = landmarks[:, 5:, :, :]
        
        # Apply bilateral consistency (optional processing)
        # For now, just return the landmarks as-is
        # In future, could add cross-attention or other consistency mechanisms
        
        return landmarks


class UNet(nn.Module):

    def __init__(self, in_ch, out_ch, down_drop, up_drop, predict_gaussian=False):
        super().__init__()
        assert len(down_drop) == 4
        assert len(up_drop) == 4
        self.inconv = DoubleConv(in_ch, 64)
        self.down1 = DownBlock(64, 128, down_drop[0])
        self.down2 = DownBlock(128, 256, down_drop[1])
        self.down3 = DownBlock(256, 512, down_drop[2])
        self.down4 = DownBlock(512, 1024, down_drop[3])
        self.up1 = UpBlock(1024, 512, up_drop[0])
        self.up2 = UpBlock(512, 256, up_drop[1])
        self.up3 = UpBlock(256, 128, up_drop[2])
        self.up4 = UpBlock(128, 64, up_drop[3])
        self.outconv = nn.Conv2d(64, out_ch, 1)
        self.predict_gaussian = predict_gaussian


    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = x5
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)

        x_final = x
        if self.predict_gaussian:
            n_landmarks = x.shape[1] // 2
            x_final = x.clone()
            """
            For numerical stability the implementation reverts to the linear function when input×β>threshold
            """
            x_final[:, n_landmarks:, :, :] = F.softplus(x[:, n_landmarks:, :, :]) + 1e-6
        return x_final


class UnifiedUNet(UNet):
    """Unified U-Net for predicting both left and right landmarks simultaneously"""
    
    def __init__(self, in_ch=3, down_drop=[0.4, 0.4, 0.4, 0.4], up_drop=[0.4, 0.4, 0.4, 0.4], predict_gaussian=False):
        # Initialize with 10 output channels (5 left + 5 right landmarks)
        super().__init__(in_ch, 10, down_drop, up_drop, predict_gaussian)
        
        # Add anatomical consistency module
        self.consistency_module = AnatomicalConsistencyModule(channels=10)
        
    def forward(self, x):
        # Get landmark heatmaps from base U-Net
        landmarks = super().forward(x)
        
        # Apply anatomical consistency
        landmarks = self.consistency_module(landmarks)
        
        return landmarks