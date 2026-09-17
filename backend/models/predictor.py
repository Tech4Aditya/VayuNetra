"""
predictor.py

Task 3 (Prediction) — SCOPED HONESTLY: this predicts short-horizon TREND
(intensifying / steady / weakening) and next-step track displacement from a
sequence of recent frames. It does NOT attempt full numerical-weather-style
landfall/rainfall forecasting — say this explicitly in the pitch.

Architecture: CNN feature extractor (shared style with classifier) -> LSTM
over the time dimension -> two heads (trend classification, track regression).
"""
import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, out_dim, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        # x: (B, 1, H, W) -> (B, out_dim)
        return self.net(x).flatten(1)


class CycloneTrendLSTM(nn.Module):
    def __init__(self, feat_dim=64, hidden_dim=64, num_trend_classes=3):
        super().__init__()
        self.encoder = FrameEncoder(feat_dim)
        self.lstm = nn.LSTM(feat_dim, hidden_dim, batch_first=True)
        self.trend_head = nn.Linear(hidden_dim, num_trend_classes)  # weaken/steady/intensify
        self.track_head = nn.Linear(hidden_dim, 2)                  # next-step (dx, dy)

    def forward(self, frame_seq):
        # frame_seq: (B, T, 1, H, W)
        b, t, c, h, w = frame_seq.shape
        feats = self.encoder(frame_seq.view(b * t, c, h, w)).view(b, t, -1)
        lstm_out, _ = self.lstm(feats)
        last = lstm_out[:, -1, :]
        trend_logits = self.trend_head(last)
        track_delta = self.track_head(last)
        return trend_logits, track_delta


def loss_fn(trend_logits, track_delta, trend_target, track_target):
    trend_loss = nn.functional.cross_entropy(trend_logits, trend_target)
    track_loss = nn.functional.mse_loss(track_delta, track_target)
    return trend_loss + track_loss
