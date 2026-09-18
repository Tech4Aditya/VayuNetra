import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


class TCIRTemporalModel(nn.Module):

    def __init__(
        self,
        in_channels=2,
        feature_dim=128,
        hidden_dim=128,
        num_layers=1
    ):
        super().__init__()

        # Shared CNN encoder
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32),    # 128 -> 64
            ConvBlock(32, 64),             # 64 -> 32
            ConvBlock(64, 128),            # 32 -> 16
            ConvBlock(128, 256),            # 16 -> 8

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        # Temporal model
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Multi-task prediction heads
        self.wind_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.pressure_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        x shape:
            [batch, time, channels, height, width]

        Example:
            [16, 4, 2, 128, 128]
        """

        batch_size, time_steps, channels, height, width = x.shape

        # Merge batch + time so CNN processes
        # every frame using the SAME encoder.
        x = x.reshape(
            batch_size * time_steps,
            channels,
            height,
            width
        )

        # CNN features
        x = self.encoder(x)

        x = self.feature_head(x)

        # Restore temporal dimension
        x = x.reshape(
            batch_size,
            time_steps,
            -1
        )

        # GRU
        sequence_output, hidden = self.gru(x)

        # Last timestep represents the complete
        # temporal history.
        temporal_feature = sequence_output[:, -1, :]

        wind = self.wind_head(temporal_feature).squeeze(1)
        pressure = self.pressure_head(temporal_feature).squeeze(1)
        size = self.size_head(temporal_feature).squeeze(1)

        return {
            "wind": wind,
            "pressure": pressure,
            "size": size
        }