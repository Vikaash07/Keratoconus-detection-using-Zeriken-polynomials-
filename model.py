#model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet34Autoencoder(nn.Module):
    def __init__(self, in_channels=1):
        super(ResNet34Autoencoder, self).__init__()
        self.in_channels = 64

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 256 -> 128
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 128 -> 64

        self.layer1 = self._make_layer(BasicBlock, 64, 3)   # -> 64x64
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)  # -> 32x32
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)  # -> 16x16
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)  # -> 8x8

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),     # 128 -> 256
            nn.Sigmoid()  # Output normalized map
        )

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # 256 -> 128
        x = self.maxpool(x)                     # 128 -> 64
        x = self.layer1(x)                      # -> 64
        x = self.layer2(x)                      # -> 32
        x = self.layer3(x)                      # -> 16
        x = self.layer4(x)                      # -> 8

        x = self.decoder(x)                     # -> 256
        return x