import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class TCIRIntensityCNN(nn.Module):
    """
    Multi-task CNN for TCIR satellite imagery.

    Input:
        [B, C, 128, 128]

    Outputs:
        wind_kt
        pressure_hpa
        size_nmi
    """

    def __init__(self, in_channels=4):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.shared = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Dropout(0.25),
        )

        # Wind intensity
        self.wind_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Minimum sea-level pressure
        self.pressure_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Cyclone size
        self.size_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        features = self.shared(x)

        wind = self.wind_head(features)
        pressure = self.pressure_head(features)
        size = self.size_head(features)

        return {
            "wind": wind.squeeze(1),
            "pressure": pressure.squeeze(1),
            "size": size.squeeze(1),
        }