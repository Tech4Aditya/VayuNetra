"""
classifier.py

Cyclone identification + intensity estimation + category classification.

Current input:
    (B, 1, H, W)

Outputs:
    presence_logit      -> cyclone / no cyclone
    category_logits     -> 6 intensity categories
    intensity           -> continuous intensity estimate

Architecture:
    CNN backbone
        |
        +--> presence head
        +--> category head
        +--> intensity regression head

The architecture is designed so the input channel count can later be
changed from 1 (IR only) to multiple satellite channels such as:

    IR + Water Vapour + Visible + derived channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CATEGORIES = 6

CATEGORIES = [
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Super Cyclonic Storm",
]


class ConvBlock(nn.Module):
    """
    Convolutional feature extraction block.

    Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool
    """

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

            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class CycloneCNN(nn.Module):

    def __init__(
        self,
        num_categories=NUM_CATEGORIES,
        in_channels=1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_categories = num_categories

        # ---------------------------------------------------------
        # CNN BACKBONE
        # ---------------------------------------------------------

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),     # 64 -> 32
            ConvBlock(32, 64),              # 32 -> 16
            ConvBlock(64, 128),             # 16 -> 8
            ConvBlock(128, 256),            # 8 -> 4
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # ---------------------------------------------------------
        # SHARED FEATURE HEAD
        # ---------------------------------------------------------

        self.feature_head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Dropout(0.25),
        )

        # ---------------------------------------------------------
        # TASK 1 — CYCLONE IDENTIFICATION
        # ---------------------------------------------------------

        self.presence_head = nn.Linear(128, 1)

        # ---------------------------------------------------------
        # TASK 2 — CATEGORY CLASSIFICATION
        # ---------------------------------------------------------

        self.category_head = nn.Linear(
            128,
            num_categories,
        )

        # ---------------------------------------------------------
        # TASK 2B — CONTINUOUS INTENSITY ESTIMATION
        #
        # Current synthetic target is normalized intensity [0, 1].
        # Later this can become physical wind speed / pressure.
        # ---------------------------------------------------------

        self.intensity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x:
            (B, C, H, W)

        Returns:
            presence_logit:
                (B,)

            category_logits:
                (B, NUM_CATEGORIES)

            intensity:
                (B,)
        """

        feats = self.features(x)

        feats = self.pool(feats)

        feats = self.feature_head(feats)

        presence_logit = self.presence_head(feats).squeeze(-1)

        category_logits = self.category_head(feats)

        intensity = self.intensity_head(feats).squeeze(-1)

        return (
            presence_logit,
            category_logits,
            intensity,
        )


def loss_fn(
    presence_logit,
    category_logits,
    intensity_pred,
    presence_target,
    category_target,
    intensity_target,
    category_weight=1.0,
    intensity_weight=0.5,
):
    """
    Multi-task loss.

    Identification:
        BCEWithLogits

    Classification:
        CrossEntropy

        IMPORTANT:
        Negative/no-cyclone samples have category_target = -1.
        Those samples are excluded from category loss.

    Intensity:
        Smooth L1 regression

        Also excluded for negative samples.
    """

    # ---------------------------------------------------------
    # TASK 1 — PRESENCE
    # ---------------------------------------------------------

    presence_target = presence_target.float()

    presence_loss = F.binary_cross_entropy_with_logits(
        presence_logit,
        presence_target,
    )

    # ---------------------------------------------------------
    # TASK 2 — CATEGORY
    # ---------------------------------------------------------

    positive_mask = presence_target > 0.5

    if positive_mask.any():

        category_loss = F.cross_entropy(
            category_logits[positive_mask],
            category_target[positive_mask],
        )

    else:
        # Keeps the graph valid if a batch happens to contain
        # only negative examples.
        category_loss = torch.zeros(
            (),
            device=presence_logit.device,
        )

    # ---------------------------------------------------------
    # TASK 2B — INTENSITY
    # ---------------------------------------------------------

    if positive_mask.any():

        intensity_loss = F.smooth_l1_loss(
            intensity_pred[positive_mask],
            intensity_target[positive_mask],
        )

    else:
        intensity_loss = torch.zeros(
            (),
            device=presence_logit.device,
        )

    # ---------------------------------------------------------
    # TOTAL
    # ---------------------------------------------------------

    total_loss = (
        presence_loss
        + category_weight * category_loss
        + intensity_weight * intensity_loss
    )

    return total_loss