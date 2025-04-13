#model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CornealTopographyCNN(nn.Module):
    def __init__(self):
        super(CornealTopographyCNN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 256x256 -> 256x256
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 256x256 -> 128x128

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 128x128 -> 128x128
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 128x128 -> 64x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 64x64 -> 64x64
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 64x64 -> 32x32
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # 32x32 -> 64x64
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 64x64 -> 128x128
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),   # 128x128 -> 256x256
            nn.Sigmoid()  # Normalize output
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded