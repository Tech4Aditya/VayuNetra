"""
VayuNetra INSAT V3
-------------------

Storm-level split + canonical 7-class labels + ordinal intensity prediction
+ wind regression + pressure regression.

Dataset:
    backend/data/labels/*_insat_manifest.csv

Calibrated samples:
    backend/data/processed/insat_<storm>_calibrated/

Input:
    2 channels
        C0 = TIR1
        C1 = Water Vapour

Model outputs:
    1. Ordinal intensity thresholds (6)
    2. Wind regression
    3. Pressure regression

Canonical classes:
    0 Depression
    1 Deep Depression
    2 Cyclonic Storm
    3 Severe Cyclonic Storm
    4 Very Severe Cyclonic Storm
    5 Extremely Severe Cyclonic Storm
    6 Super Cyclonic Storm

IMPORTANT:
    Entire storms are kept together in train/validation/test.
    No image-level random split is used.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

# This script is:
# backend/ml/scripts/train_insat_v3_ordinal.py
#
# parents[2] -> backend
ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# CANONICAL CATEGORIES
# ============================================================

CATEGORIES = [
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm",
    "Super Cyclonic Storm",
]

CATEGORY_TO_ID = {
    name: i
    for i, name in enumerate(CATEGORIES)
}


# ============================================================
# RAW LABEL -> CANONICAL LABEL
# ============================================================

RAW_TO_CANONICAL = {

    # Depression
    "D": "Depression",
    "Depression": "Depression",

    # Deep Depression
    "DD": "Deep Depression",
    "Deep Depression": "Deep Depression",

    # Cyclonic Storm
    "CS": "Cyclonic Storm",
    "Cyclonic Storm": "Cyclonic Storm",

    # Severe Cyclonic Storm
    "SCS": "Severe Cyclonic Storm",
    "Severe Cyclonic Storm": "Severe Cyclonic Storm",

    # Very Severe Cyclonic Storm
    "VSCS": "Very Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm": "Very Severe Cyclonic Storm",

    # Extremely Severe Cyclonic Storm
    "ESCS": "Extremely Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm":
        "Extremely Severe Cyclonic Storm",

    # Super Cyclonic Storm
    "SuCS": "Super Cyclonic Storm",
    "Super Cyclonic Storm": "Super Cyclonic Storm",
}


# ============================================================
# STORM SPLIT
# ============================================================

# IMPORTANT:
# AMPHAN contains the only Super Cyclonic Storm samples.
# Therefore AMPHAN must remain in training if we want the
# model to see SuCS during training.

TRAIN_STORMS = [
    "FANI",
    "AMPHAN",
    "ASANI",
    "BULBUL",
    "TAUKTAE",
    "MICHAUNG",
]

VAL_STORMS = [
    "YAAS",
    "GULAAB",
]

TEST_STORMS = [
    "BIPARJOY",
    "NISARGA",
]


# ============================================================
# TRAINING CONFIG
# ============================================================

BATCH_SIZE = 8

EPOCHS = 100

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

ORDINAL_LOSS_WEIGHT = 1.0

WIND_LOSS_WEIGHT = 1.0

PRESSURE_LOSS_WEIGHT = 0.75

PATIENCE = 15

SEED = 42


# ============================================================
# SEED
# ============================================================

def seed_all():

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ============================================================
# MODEL
# ============================================================

class INSATOrdinalCNN(nn.Module):

    def __init__(self, in_channels=2):

        super().__init__()

        # ----------------------------------------------------
        # Convolutional block
        # ----------------------------------------------------

        def block(cin, cout):

            return nn.Sequential(

                nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),

                nn.BatchNorm2d(cout),

                nn.ReLU(inplace=True),

                nn.Conv2d(
                    cout,
                    cout,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),

                nn.BatchNorm2d(cout),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(2),
            )

        # ----------------------------------------------------
        # CNN backbone
        # ----------------------------------------------------

        self.features = nn.Sequential(

            block(in_channels, 32),

            block(32, 64),

            block(64, 128),

            block(128, 256),
        )

        # ----------------------------------------------------
        # Global pooling
        # ----------------------------------------------------

        self.pool = nn.AdaptiveAvgPool2d(1)

        # ----------------------------------------------------
        # Shared feature layer
        # ----------------------------------------------------

        self.shared = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.25),
        )

        # ----------------------------------------------------
        # ORDINAL HEAD
        #
        # 6 thresholds:
        #
        # category >= 1
        # category >= 2
        # category >= 3
        # category >= 4
        # category >= 5
        # category >= 6
        #
        # Therefore:
        #
        # Depression:
        # 0 0 0 0 0 0
        #
        # Deep Depression:
        # 1 0 0 0 0 0
        #
        # Cyclonic Storm:
        # 1 1 0 0 0 0
        #
        # ...
        #
        # Super Cyclonic Storm:
        # 1 1 1 1 1 1
        # ----------------------------------------------------

        self.ordinal_head = nn.Linear(
            128,
            6,
        )

        # ----------------------------------------------------
        # WIND REGRESSION
        # ----------------------------------------------------

        self.wind_head = nn.Sequential(

            nn.Linear(
                128,
                64,
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                64,
                1,
            ),
        )

        # ----------------------------------------------------
        # PRESSURE REGRESSION
        # ----------------------------------------------------

        self.pressure_head = nn.Sequential(

            nn.Linear(
                128,
                64,
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.shared(x)

        ordinal_logits = self.ordinal_head(x)

        wind = self.wind_head(x).squeeze(-1)

        pressure = self.pressure_head(x).squeeze(-1)

        return (
            ordinal_logits,
            wind,
            pressure,
        )


# ============================================================
# DATASET
# ============================================================

class INSATDataset(Dataset):

    def __init__(
        self,
        df,
        stats,
    ):

        self.df = df.reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # FIX:
        # stats contains 8 values
        # ----------------------------------------------------

        self.tir_mean = stats[0]

        self.tir_std = stats[1]

        self.wv_mean = stats[2]

        self.wv_std = stats[3]

        self.wind_mean = stats[4]

        self.wind_std = stats[5]

        self.pressure_mean = stats[6]

        self.pressure_std = stats[7]

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        sample_dir = Path(
            row["sample_dir"]
        )

        # ----------------------------------------------------
        # Load calibrated TIR1
        # ----------------------------------------------------

        tir = np.load(
            sample_dir /
            "tir1_radiance.npy"
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Load calibrated WV
        # ----------------------------------------------------

        wv = np.load(
            sample_dir /
            "wv_radiance.npy"
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Shape validation
        # ----------------------------------------------------

        if tir.shape != (128, 128):

            raise ValueError(
                f"Bad TIR shape at "
                f"{sample_dir}: {tir.shape}"
            )

        if wv.shape != (128, 128):

            raise ValueError(
                f"Bad WV shape at "
                f"{sample_dir}: {wv.shape}"
            )

        # ----------------------------------------------------
        # Normalize channels
        # ----------------------------------------------------

        tir = (
            tir - self.tir_mean
        ) / max(
            self.tir_std,
            1e-6,
        )

        wv = (
            wv - self.wv_mean
        ) / max(
            self.wv_std,
            1e-6,
        )

        # ----------------------------------------------------
        # Stack channels
        # ----------------------------------------------------

        x = np.stack(
            [
                tir,
                wv,
            ],
            axis=0,
        )

        # ----------------------------------------------------
        # Canonical class
        # ----------------------------------------------------

        category = int(
            row["category_id"]
        )

        # ----------------------------------------------------
        # Ordinal targets
        #
        # category >= threshold
        # ----------------------------------------------------

        ordinal = np.array(
            [
                1.0
                if category >= k
                else 0.0
                for k in range(1, 7)
            ],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Normalize wind
        # ----------------------------------------------------

        wind = (
            float(row["wind_kt"])
            - self.wind_mean
        ) / max(
            self.wind_std,
            1e-6,
        )

        # ----------------------------------------------------
        # Normalize pressure
        # ----------------------------------------------------

        pressure = (
            float(row["pressure_hpa"])
            - self.pressure_mean
        ) / max(
            self.pressure_std,
            1e-6,
        )

        return (

            torch.from_numpy(x),

            torch.tensor(
                category,
                dtype=torch.long,
            ),

            torch.from_numpy(
                ordinal
            ),

            torch.tensor(
                wind,
                dtype=torch.float32,
            ),

            torch.tensor(
                pressure,
                dtype=torch.float32,
            ),
        )


# ============================================================
# MANIFEST PATH
# ============================================================

def manifest(storm):

    return (
        ROOT
        / "data"
        / "labels"
        / f"{storm.lower()}_insat_manifest.csv"
    )


# ============================================================
# SAMPLE DIRECTORY
# ============================================================

def sample_dir_from_file(
    storm,
    file_value,
):

    stem = Path(
        str(file_value)
    ).stem

    return (
        ROOT
        / "data"
        / "processed"
        / f"insat_{storm.lower()}_calibrated"
        / stem
    )


# ============================================================
# LOAD ALL DATA
# ============================================================

def load_all():

    frames = []

    all_storms = (
        TRAIN_STORMS
        + VAL_STORMS
        + TEST_STORMS
    )

    for storm in all_storms:

        path = manifest(storm)

        if not path.exists():

            raise FileNotFoundError(
                f"Missing manifest: {path}"
            )

        df = pd.read_csv(path)

        if len(df) == 0:

            raise ValueError(
                f"Empty manifest: {path}"
            )

        # ----------------------------------------------------
        # Store storm name
        # ----------------------------------------------------

        df["storm"] = storm

        # ----------------------------------------------------
        # Canonicalize category
        # ----------------------------------------------------

        df[
            "canonical_category"
        ] = df[
            "category"
        ].map(
            RAW_TO_CANONICAL
        )

        # ----------------------------------------------------
        # Check unmapped labels
        # ----------------------------------------------------

        if df[
            "canonical_category"
        ].isna().any():

            bad = sorted(
                df.loc[
                    df[
                        "canonical_category"
                    ].isna(),
                    "category",
                ].unique()
            )

            raise ValueError(
                f"Unmapped categories "
                f"in {storm}: {bad}"
            )

        # ----------------------------------------------------
        # Category IDs
        # ----------------------------------------------------

        df[
            "category_id"
        ] = df[
            "canonical_category"
        ].map(
            CATEGORY_TO_ID
        )

        # ----------------------------------------------------
        # Locate calibrated sample
        # ----------------------------------------------------

        df[
            "sample_dir"
        ] = [

            str(
                sample_dir_from_file(
                    storm,
                    x,
                )
            )

            for x in df["file"]
        ]

        frames.append(df)

    # --------------------------------------------------------
    # Combine all storms
    # --------------------------------------------------------

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Validate calibrated pairs
    # --------------------------------------------------------

    missing = []

    for p in df[
        "sample_dir"
    ]:

        p = Path(p)

        tir_path = (
            p /
            "tir1_radiance.npy"
        )

        wv_path = (
            p /
            "wv_radiance.npy"
        )

        if (
            not tir_path.exists()
            or not wv_path.exists()
        ):

            missing.append(
                str(p)
            )

    if missing:

        raise FileNotFoundError(
            "Missing calibrated pair:\n"
            + missing[0]
        )

    return df


# ============================================================
# TRAIN-ONLY NORMALIZATION STATS
# ============================================================

def stats_from_train(df):

    tir_sum = 0.0

    tir_sq = 0.0

    wv_sum = 0.0

    wv_sq = 0.0

    n = 0

    for p in df[
        "sample_dir"
    ]:

        p = Path(p)

        tir = np.load(
            p /
            "tir1_radiance.npy"
        ).astype(
            np.float64
        )

        wv = np.load(
            p /
            "wv_radiance.npy"
        ).astype(
            np.float64
        )

        tir_sum += tir.sum()

        tir_sq += np.square(
            tir
        ).sum()

        wv_sum += wv.sum()

        wv_sq += np.square(
            wv
        ).sum()

        n += tir.size

    # --------------------------------------------------------
    # TIR statistics
    # --------------------------------------------------------

    tir_mean = (
        tir_sum / n
    )

    tir_var = (
        tir_sq / n
        - tir_mean ** 2
    )

    tir_std = np.sqrt(
        max(
            tir_var,
            1e-12,
        )
    )

    # --------------------------------------------------------
    # WV statistics
    # --------------------------------------------------------

    wv_mean = (
        wv_sum / n
    )

    wv_var = (
        wv_sq / n
        - wv_mean ** 2
    )

    wv_std = np.sqrt(
        max(
            wv_var,
            1e-12,
        )
    )

    # --------------------------------------------------------
    # Physical target stats
    # --------------------------------------------------------

    wind_mean = float(
        df[
            "wind_kt"
        ].mean()
    )

    wind_std = float(
        df[
            "wind_kt"
        ].std(
            ddof=0
        )
    )

    pressure_mean = float(
        df[
            "pressure_hpa"
        ].mean()
    )

    pressure_std = float(
        df[
            "pressure_hpa"
        ].std(
            ddof=0
        )
    )

    return (

        float(tir_mean),

        float(tir_std),

        float(wv_mean),

        float(wv_std),

        wind_mean,

        wind_std,

        pressure_mean,

        pressure_std,
    )


# ============================================================
# ORDINAL PREDICTION -> CLASS
# ============================================================

def ordinal_class(
    probabilities
):

    # Count how many thresholds are satisfied.
    return np.sum(
        probabilities >= 0.5,
        axis=1,
    ).astype(
        np.int64
    )


# ============================================================
# CLASS BALANCING
# ============================================================

def ordinal_pos_weights(
    train_df
):

    y = train_df[
        "category_id"
    ].to_numpy()

    positive = np.array(
        [
            (
                y >= k
            ).sum()
            for k in range(1, 7)
        ],
        dtype=np.float64,
    )

    negative = (
        len(y)
        - positive
    )

    weights = (
        negative
        /
        np.maximum(
            positive,
            1,
        )
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    device,
    stats,
    ordinal_loss_fn,
):

    model.eval()

    all_true = []

    all_pred = []

    all_wind_true = []

    all_wind_pred = []

    all_pressure_true = []

    all_pressure_pred = []

    losses = []

    with torch.no_grad():

        for (
            x,
            category,
            ordinal,
            wind,
            pressure,
        ) in loader:

            x = x.to(
                device
            )

            ordinal = ordinal.to(
                device
            )

            wind = wind.to(
                device
            )

            pressure = pressure.to(
                device
            )

            (
                ordinal_logits,
                wind_hat,
                pressure_hat,
            ) = model(x)

            # ------------------------------------------------
            # Losses
            # ------------------------------------------------

            loss_ordinal = (
                ordinal_loss_fn(
                    ordinal_logits,
                    ordinal,
                )
            )

            loss_wind = (
                nn.functional.smooth_l1_loss(
                    wind_hat,
                    wind,
                )
            )

            loss_pressure = (
                nn.functional.smooth_l1_loss(
                    pressure_hat,
                    pressure,
                )
            )

            loss = (

                ORDINAL_LOSS_WEIGHT
                * loss_ordinal

                +

                WIND_LOSS_WEIGHT
                * loss_wind

                +

                PRESSURE_LOSS_WEIGHT
                * loss_pressure
            )

            losses.append(
                float(
                    loss.item()
                )
            )

            # ------------------------------------------------
            # Category prediction
            # ------------------------------------------------

            probabilities = (
                torch.sigmoid(
                    ordinal_logits
                )
                .cpu()
                .numpy()
            )

            predictions = (
                ordinal_class(
                    probabilities
                )
            )

            all_true.extend(
                category.cpu().numpy()
            )

            all_pred.extend(
                predictions
            )

            # ------------------------------------------------
            # Wind
            # ------------------------------------------------

            all_wind_true.extend(
                wind.cpu().numpy()
            )

            all_wind_pred.extend(
                wind_hat.cpu().numpy()
            )

            # ------------------------------------------------
            # Pressure
            # ------------------------------------------------

            all_pressure_true.extend(
                pressure.cpu().numpy()
            )

            all_pressure_pred.extend(
                pressure_hat.cpu().numpy()
            )

    # ========================================================
    # Convert arrays
    # ========================================================

    all_true = np.asarray(
        all_true
    )

    all_pred = np.asarray(
        all_pred
    )

    # ========================================================
    # Convert normalized wind back to knots
    # ========================================================

    wind_true = (
        np.asarray(
            all_wind_true
        )
        * stats[5]
        + stats[4]
    )

    wind_pred = (
        np.asarray(
            all_wind_pred
        )
        * stats[5]
        + stats[4]
    )

    # ========================================================
    # Convert normalized pressure back to hPa
    # ========================================================

    pressure_true = (
        np.asarray(
            all_pressure_true
        )
        * stats[7]
        + stats[6]
    )

    pressure_pred = (
        np.asarray(
            all_pressure_pred
        )
        * stats[7]
        + stats[6]
    )

    # ========================================================
    # Metrics
    # ========================================================

    accuracy = accuracy_score(
        all_true,
        all_pred,
    )

    macro_f1 = f1_score(
        all_true,
        all_pred,
        average="macro",
        zero_division=0,
    )

    category_mae = np.mean(
        np.abs(
            all_true
            - all_pred
        )
    )

    wind_mae = mean_absolute_error(
        wind_true,
        wind_pred,
    )

    # R2 can fail if test target is constant.
    try:

        wind_r2 = r2_score(
            wind_true,
            wind_pred,
        )

    except Exception:

        wind_r2 = float("nan")

    pressure_mae = mean_absolute_error(
        pressure_true,
        pressure_pred,
    )

    try:

        pressure_r2 = r2_score(
            pressure_true,
            pressure_pred,
        )

    except Exception:

        pressure_r2 = float("nan")

    return {

        "loss": float(
            np.mean(losses)
        ),

        "accuracy": float(
            accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "category_mae": float(
            category_mae
        ),

        "wind_mae": float(
            wind_mae
        ),

        "wind_r2": float(
            wind_r2
        ),

        "pressure_mae": float(
            pressure_mae
        ),

        "pressure_r2": float(
            pressure_r2
        ),

        "true": all_true,

        "pred": all_pred,

        "wind_true": wind_true,

        "wind_pred": wind_pred,

        "pressure_true": pressure_true,

        "pressure_pred": pressure_pred,
    }


# ============================================================
# MAIN TRAINING
# ============================================================

def train():

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    seed_all()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_all()

    # ========================================================
    # SPLIT BY STORM
    # ========================================================

    train_df = df[
        df["storm"].isin(
            TRAIN_STORMS
        )
    ].copy()

    val_df = df[
        df["storm"].isin(
            VAL_STORMS
        )
    ].copy()

    test_df = df[
        df["storm"].isin(
            TEST_STORMS
        )
    ].copy()

    if len(train_df) == 0:

        raise RuntimeError(
            "Training set is empty."
        )

    if len(val_df) == 0:

        raise RuntimeError(
            "Validation set is empty."
        )

    if len(test_df) == 0:

        raise RuntimeError(
            "Test set is empty."
        )

    # ========================================================
    # PRINT SPLIT
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STORM SPLIT"
    )

    print(
        "=" * 70
    )

    print(
        "TRAIN:",
        TRAIN_STORMS,
        len(train_df),
    )

    print(
        "VAL  :",
        VAL_STORMS,
        len(val_df),
    )

    print(
        "TEST :",
        TEST_STORMS,
        len(test_df),
    )

    # ========================================================
    # CATEGORY DISTRIBUTION
    # ========================================================

    print(
        "\nTRAIN CATEGORY"
    )

    print(
        train_df[
            "canonical_category"
        ]
        .value_counts()
        .reindex(
            CATEGORIES,
            fill_value=0,
        )
    )

    print(
        "\nVAL CATEGORY"
    )

    print(
        val_df[
            "canonical_category"
        ]
        .value_counts()
        .reindex(
            CATEGORIES,
            fill_value=0,
        )
    )

    print(
        "\nTEST CATEGORY"
    )

    print(
        test_df[
            "canonical_category"
        ]
        .value_counts()
        .reindex(
            CATEGORIES,
            fill_value=0,
        )
    )

    # ========================================================
    # TRAIN-ONLY STATS
    # ========================================================

    stats = stats_from_train(
        train_df
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAIN-ONLY STATS"
    )

    print(
        "=" * 70
    )

    print(
        f"TIR mean/std: "
        f"{stats[0]:.8f} / "
        f"{stats[1]:.8f}"
    )

    print(
        f"WV  mean/std: "
        f"{stats[2]:.8f} / "
        f"{stats[3]:.8f}"
    )

    print(
        f"Wind mean/std: "
        f"{stats[4]:.4f} / "
        f"{stats[5]:.4f}"
    )

    print(
        f"Pressure mean/std: "
        f"{stats[6]:.4f} / "
        f"{stats[7]:.4f}"
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_ds = INSATDataset(
        train_df,
        stats,
    )

    val_ds = INSATDataset(
        val_df,
        stats,
    )

    test_ds = INSATDataset(
        test_df,
        stats,
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = INSATOrdinalCNN(
        in_channels=2
    ).to(device)

    # ========================================================
    # ORDINAL CLASS BALANCING
    # ========================================================

    pos_weight = (
        ordinal_pos_weights(
            train_df
        )
        .to(device)
    )

    print(
        "\nOrdinal positive weights:"
    )

    print(
        pos_weight.cpu().numpy()
    )

    ordinal_loss_fn = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=0.5,

            patience=5,
        )
    )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint_dir = (
        ROOT
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / "insat_v3_ordinal_best.pt"
    )

    # ========================================================
    # SAVE NORMALIZATION STATS
    # ========================================================

    stats_path = (

        ROOT
        / "data"
        / "processed"
        / "insat_v3_training_stats.npz"
    )

    np.savez(

        stats_path,

        tir_mean=stats[0],

        tir_std=stats[1],

        wv_mean=stats[2],

        wv_std=stats[3],

        wind_mean=stats[4],

        wind_std=stats[5],

        pressure_mean=stats[6],

        pressure_std=stats[7],
    )

    # ========================================================
    # TRAIN
    # ========================================================

    best_val_loss = float(
        "inf"
    )

    stale_epochs = 0

    print(
        "\n"
        + "=" * 70
    )

    print(
        "INSAT V3 TRAINING"
    )

    print(
        "=" * 70
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for (
            x,
            category,
            ordinal,
            wind,
            pressure,
        ) in train_loader:

            x = x.to(
                device
            )

            ordinal = ordinal.to(
                device
            )

            wind = wind.to(
                device
            )

            pressure = pressure.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            (
                ordinal_logits,
                wind_hat,
                pressure_hat,
            ) = model(x)

            # ------------------------------------------------
            # Ordinal loss
            # ------------------------------------------------

            ordinal_loss = (
                ordinal_loss_fn(
                    ordinal_logits,
                    ordinal,
                )
            )

            # ------------------------------------------------
            # Wind loss
            # ------------------------------------------------

            wind_loss = (
                nn.functional.smooth_l1_loss(
                    wind_hat,
                    wind,
                )
            )

            # ------------------------------------------------
            # Pressure loss
            # ------------------------------------------------

            pressure_loss = (
                nn.functional.smooth_l1_loss(
                    pressure_hat,
                    pressure,
                )
            )

            # ------------------------------------------------
            # Total loss
            # ------------------------------------------------

            loss = (

                ORDINAL_LOSS_WEIGHT
                * ordinal_loss

                +

                WIND_LOSS_WEIGHT
                * wind_loss

                +

                PRESSURE_LOSS_WEIGHT
                * pressure_loss
            )

            # ------------------------------------------------
            # Backprop
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            running_loss += (
                float(
                    loss.item()
                )
            )

        # ====================================================
        # VALIDATION
        # ====================================================

        val = evaluate(

            model,

            val_loader,

            device,

            stats,

            ordinal_loss_fn,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val["loss"]
        )

        # ----------------------------------------------------
        # Current learning rate
        # ----------------------------------------------------

        current_lr = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        marker = ""

        if (
            val["loss"]
            < best_val_loss
        ):

            best_val_loss = (
                val["loss"]
            )

            stale_epochs = 0

            torch.save(

                {

                    "model_state_dict":
                        model.state_dict(),

                    "categories":
                        CATEGORIES,

                    "train_storms":
                        TRAIN_STORMS,

                    "val_storms":
                        VAL_STORMS,

                    "test_storms":
                        TEST_STORMS,

                    "stats":
                        stats,

                    "epoch":
                        epoch,

                    "val_metrics": {

                        k: v

                        for k, v
                        in val.items()

                        if np.isscalar(v)
                    },
                },

                checkpoint_path,
            )

            marker = (
                " <-- BEST"
            )

        else:

            stale_epochs += 1

        # ----------------------------------------------------
        # Print epoch
        # ----------------------------------------------------

        print(

            f"epoch {epoch:03d}/{EPOCHS} "

            f"train="
            f"{running_loss / len(train_loader):.4f} "

            f"val="
            f"{val['loss']:.4f} "

            f"val_acc="
            f"{val['accuracy']:.3f} "

            f"val_f1="
            f"{val['macro_f1']:.3f} "

            f"wind_mae="
            f"{val['wind_mae']:.2f} "

            f"pressure_mae="
            f"{val['pressure_mae']:.2f} "

            f"lr="
            f"{current_lr:.2e}"

            f"{marker}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            stale_epochs
            >= PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING BEST CHECKPOINT"
    )

    print(
        "=" * 70
    )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device,

        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    test = evaluate(

        model,

        test_loader,

        device,

        stats,

        ordinal_loss_fn,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "INSAT V3 FINAL TEST"
    )

    print(
        "=" * 70
    )

    print(
        f"Accuracy       : "
        f"{test['accuracy']:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{test['macro_f1']:.4f}"
    )

    print(
        f"Category MAE   : "
        f"{test['category_mae']:.4f} classes"
    )

    print(
        f"Wind MAE       : "
        f"{test['wind_mae']:.4f} kt"
    )

    print(
        f"Wind R2        : "
        f"{test['wind_r2']:.4f}"
    )

    print(
        f"Pressure MAE   : "
        f"{test['pressure_mae']:.4f} hPa"
    )

    print(
        f"Pressure R2    : "
        f"{test['pressure_r2']:.4f}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        test["true"],

        test["pred"],

        labels=list(
            range(
                len(CATEGORIES)
            )
        ),
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        "=" * 70
    )

    print(
        cm
    )

    # ========================================================
    # CLASS SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEST CLASS SUMMARY"
    )

    print(
        "=" * 70
    )

    for i, name in enumerate(
        CATEGORIES
    ):

        true_count = int(
            (
                test["true"]
                == i
            ).sum()
        )

        predicted_count = int(
            (
                test["pred"]
                == i
            ).sum()
        )

        print(

            f"{i}: "

            f"{name:35s} "

            f"true={true_count:3d} "

            f"pred={predicted_count:3d}"
        )

    # ========================================================
    # PER-CLASS ACCURACY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PER-CLASS RECALL"
    )

    print(
        "=" * 70
    )

    for i, name in enumerate(
        CATEGORIES
    ):

        true_count = (
            cm[i].sum()
        )

        correct = cm[i, i]

        if true_count > 0:

            recall = (
                correct
                /
                true_count
            )

            print(

                f"{name:35s}: "
                f"{recall:.3f} "
                f"({correct}/{true_count})"
            )

        else:

            print(

                f"{name:35s}: "
                f"N/A "
                f"(no test samples)"
            )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report = {

        "model":
            "INSAT V3 Ordinal CNN",

        "dataset":
            "10-storm NIO INSAT",

        "total_samples":
            int(len(df)),

        "train_samples":
            int(len(train_df)),

        "validation_samples":
            int(len(val_df)),

        "test_samples":
            int(len(test_df)),

        "categories":
            CATEGORIES,

        "train_storms":
            TRAIN_STORMS,

        "validation_storms":
            VAL_STORMS,

        "test_storms":
            TEST_STORMS,

        "metrics": {

            "accuracy":
                test["accuracy"],

            "macro_f1":
                test["macro_f1"],

            "category_mae":
                test["category_mae"],

            "wind_mae_kt":
                test["wind_mae"],

            "wind_r2":
                test["wind_r2"],

            "pressure_mae_hpa":
                test["pressure_mae"],

            "pressure_r2":
                test["pressure_r2"],
        },

        "confusion_matrix":
            cm.tolist(),

        "train_category_distribution":
            train_df[
                "canonical_category"
            ]
            .value_counts()
            .reindex(
                CATEGORIES,
                fill_value=0,
            )
            .to_dict(),

        "validation_category_distribution":
            val_df[
                "canonical_category"
            ]
            .value_counts()
            .reindex(
                CATEGORIES,
                fill_value=0,
            )
            .to_dict(),

        "test_category_distribution":
            test_df[
                "canonical_category"
            ]
            .value_counts()
            .reindex(
                CATEGORIES,
                fill_value=0,
            )
            .to_dict(),
    }

    report_path = (

        ROOT
        / "data"
        / "processed"
        / "insat_v3_evaluation.json"
    )

    report_path.write_text(

        json.dumps(
            report,
            indent=2,
        ),

        encoding="utf-8",
    )

    # ========================================================
    # FINAL OUTPUT PATHS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "V3 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Checkpoint : "
        f"{checkpoint_path}"
    )

    print(
        f"Stats      : "
        f"{stats_path}"
    )

    print(
        f"Report     : "
        f"{report_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train()