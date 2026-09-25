"""
VayuNetra INSAT V3 - Storm-Level Cross-Validation

Purpose
-------
Evaluate the INSAT V3 ordinal CNN across multiple unseen-storm test folds
instead of relying on one fixed 6/2/2 storm split.

Uses the existing 202 calibrated samples and manifests.

Design
------
- 10 NIO storms
- 5 folds
- Each fold holds out 2 storms for testing.
- Remaining storms are split into train/validation at storm level.
- No image/sample leakage between train/validation/test.
- Canonical 7-class ordinal intensity prediction.
- Wind regression.
- Pressure regression.
- Train-only normalization for every fold.

IMPORTANT
---------
Because some intensity classes are extremely sparse, the script does NOT
pretend every fold can evaluate every class. It reports:
- exact accuracy
- macro F1
- ordinal/category MAE
- wind MAE/R2
- pressure MAE/R2
- test class support
and aggregate metrics across folds.

The fold assignment is deterministic and can be changed below.
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
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

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
    name: i for i, name in enumerate(CATEGORIES)
}

RAW_TO_CANONICAL = {
    "D": "Depression",
    "Depression": "Depression",
    "DD": "Deep Depression",
    "Deep Depression": "Deep Depression",
    "CS": "Cyclonic Storm",
    "Cyclonic Storm": "Cyclonic Storm",
    "SCS": "Severe Cyclonic Storm",
    "Severe Cyclonic Storm": "Severe Cyclonic Storm",
    "VSCS": "Very Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm": "Very Severe Cyclonic Storm",
    "ESCS": "Extremely Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm": "Extremely Severe Cyclonic Storm",
    "SuCS": "Super Cyclonic Storm",
    "Super Cyclonic Storm": "Super Cyclonic Storm",
}

STORMS = [
    "FANI",
    "TAUKTAE",
    "AMPHAN",
    "YAAS",
    "ASANI",
    "NISARGA",
    "BULBUL",
    "GULAAB",
    "BIPARJOY",
    "MICHAUNG",
]

# Five deterministic 2-storm test folds.
# This is a practical evaluation design for the current 10-storm dataset.
FOLDS = [
    ("FOLD_01", ["FANI", "NISARGA"]),
    ("FOLD_02", ["AMPHAN", "GULAAB"]),
    ("FOLD_03", ["BIPARJOY", "BULBUL"]),
    ("FOLD_04", ["ASANI", "YAAS"]),
    ("FOLD_05", ["TAUKTAE", "MICHAUNG"]),
]

VAL_STORMS_PER_FOLD = 2

SEED = 42
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10

ORDINAL_LOSS_WEIGHT = 1.0
WIND_LOSS_WEIGHT = 1.0
PRESSURE_LOSS_WEIGHT = 0.75


# ============================================================
# SEED
# ============================================================

def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MODEL
# ============================================================

class INSATOrdinalCNN(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

        self.ordinal_head = nn.Linear(128, 6)

        self.wind_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.pressure_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.shared(x)

        ordinal_logits = self.ordinal_head(x)
        wind = self.wind_head(x).squeeze(-1)
        pressure = self.pressure_head(x).squeeze(-1)

        return ordinal_logits, wind, pressure


# ============================================================
# DATASET
# ============================================================

class INSATDataset(Dataset):
    def __init__(self, df, stats):
        self.df = df.reset_index(drop=True)

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
        sample_dir = Path(row["sample_dir"])

        tir = np.load(
            sample_dir / "tir1_radiance.npy"
        ).astype(np.float32)

        wv = np.load(
            sample_dir / "wv_radiance.npy"
        ).astype(np.float32)

        if tir.shape != (128, 128):
            raise ValueError(
                f"Bad TIR shape: {sample_dir}: {tir.shape}"
            )

        if wv.shape != (128, 128):
            raise ValueError(
                f"Bad WV shape: {sample_dir}: {wv.shape}"
            )

        tir = (tir - self.tir_mean) / max(self.tir_std, 1e-6)
        wv = (wv - self.wv_mean) / max(self.wv_std, 1e-6)

        x = np.stack([tir, wv], axis=0)

        category = int(row["category_id"])

        ordinal = np.array(
            [
                1.0 if category >= k else 0.0
                for k in range(1, 7)
            ],
            dtype=np.float32,
        )

        wind = (
            float(row["wind_kt"]) - self.wind_mean
        ) / max(self.wind_std, 1e-6)

        pressure = (
            float(row["pressure_hpa"]) - self.pressure_mean
        ) / max(self.pressure_std, 1e-6)

        return (
            torch.from_numpy(x),
            torch.tensor(category, dtype=torch.long),
            torch.from_numpy(ordinal),
            torch.tensor(wind, dtype=torch.float32),
            torch.tensor(pressure, dtype=torch.float32),
        )


# ============================================================
# DATA LOADING
# ============================================================

def manifest(storm):
    return (
        ROOT
        / "data"
        / "labels"
        / f"{storm.lower()}_insat_manifest.csv"
    )


def sample_dir_from_file(storm, file_value):
    stem = Path(str(file_value)).stem

    return (
        ROOT
        / "data"
        / "processed"
        / f"insat_{storm.lower()}_calibrated"
        / stem
    )


def load_all():
    frames = []

    for storm in STORMS:
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

        df["storm"] = storm

        df["canonical_category"] = (
            df["category"].map(RAW_TO_CANONICAL)
        )

        if df["canonical_category"].isna().any():
            bad = sorted(
                df.loc[
                    df["canonical_category"].isna(),
                    "category",
                ].unique()
            )
            raise ValueError(
                f"Unmapped categories in {storm}: {bad}"
            )

        df["category_id"] = (
            df["canonical_category"]
            .map(CATEGORY_TO_ID)
        )

        df["sample_dir"] = [
            str(
                sample_dir_from_file(
                    storm,
                    x,
                )
            )
            for x in df["file"]
        ]

        frames.append(df)

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    missing = []

    for p in df["sample_dir"]:
        p = Path(p)

        if not (
            (p / "tir1_radiance.npy").exists()
            and
            (p / "wv_radiance.npy").exists()
        ):
            missing.append(str(p))

    if missing:
        raise FileNotFoundError(
            "Missing calibrated pair:\n"
            + missing[0]
        )

    return df


# ============================================================
# TRAIN-ONLY STATS
# ============================================================

def stats_from_train(df):
    tir_sum = 0.0
    tir_sq = 0.0
    wv_sum = 0.0
    wv_sq = 0.0
    n = 0

    for p in df["sample_dir"]:
        p = Path(p)

        tir = np.load(
            p / "tir1_radiance.npy"
        ).astype(np.float64)

        wv = np.load(
            p / "wv_radiance.npy"
        ).astype(np.float64)

        tir_sum += tir.sum()
        tir_sq += np.square(tir).sum()

        wv_sum += wv.sum()
        wv_sq += np.square(wv).sum()

        n += tir.size

    tir_mean = tir_sum / n
    tir_var = tir_sq / n - tir_mean ** 2
    tir_std = np.sqrt(max(tir_var, 1e-12))

    wv_mean = wv_sum / n
    wv_var = wv_sq / n - wv_mean ** 2
    wv_std = np.sqrt(max(wv_var, 1e-12))

    wind_mean = float(df["wind_kt"].mean())
    wind_std = float(df["wind_kt"].std(ddof=0))

    pressure_mean = float(df["pressure_hpa"].mean())
    pressure_std = float(df["pressure_hpa"].std(ddof=0))

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
# ORDINAL WEIGHTS
# ============================================================

def ordinal_pos_weights(train_df):
    y = train_df["category_id"].to_numpy()

    positive = np.array(
        [
            (y >= k).sum()
            for k in range(1, 7)
        ],
        dtype=np.float64,
    )

    negative = len(y) - positive

    weights = negative / np.maximum(positive, 1)

    # Cap extreme weights. With only 4 SuCS samples, the raw
    # final weight becomes 31.75 and destabilizes training.
    weights = np.clip(weights, 0.25, 8.0)

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# ORDINAL DECODE
# ============================================================

def ordinal_class(probabilities):
    return np.sum(
        probabilities >= 0.5,
        axis=1,
    ).astype(np.int64)


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

            x = x.to(device)
            ordinal = ordinal.to(device)
            wind = wind.to(device)
            pressure = pressure.to(device)

            (
                ordinal_logits,
                wind_hat,
                pressure_hat,
            ) = model(x)

            ordinal_loss = ordinal_loss_fn(
                ordinal_logits,
                ordinal,
            )

            wind_loss = nn.functional.smooth_l1_loss(
                wind_hat,
                wind,
            )

            pressure_loss = nn.functional.smooth_l1_loss(
                pressure_hat,
                pressure,
            )

            loss = (
                ORDINAL_LOSS_WEIGHT * ordinal_loss
                +
                WIND_LOSS_WEIGHT * wind_loss
                +
                PRESSURE_LOSS_WEIGHT * pressure_loss
            )

            losses.append(float(loss.item()))

            probabilities = (
                torch.sigmoid(
                    ordinal_logits
                )
                .cpu()
                .numpy()
            )

            predictions = ordinal_class(
                probabilities
            )

            all_true.extend(
                category.cpu().numpy()
            )

            all_pred.extend(
                predictions
            )

            all_wind_true.extend(
                wind.cpu().numpy()
            )

            all_wind_pred.extend(
                wind_hat.cpu().numpy()
            )

            all_pressure_true.extend(
                pressure.cpu().numpy()
            )

            all_pressure_pred.extend(
                pressure_hat.cpu().numpy()
            )

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)

    wind_true = (
        np.asarray(all_wind_true)
        * stats[5]
        + stats[4]
    )

    wind_pred = (
        np.asarray(all_wind_pred)
        * stats[5]
        + stats[4]
    )

    pressure_true = (
        np.asarray(all_pressure_true)
        * stats[7]
        + stats[6]
    )

    pressure_pred = (
        np.asarray(all_pressure_pred)
        * stats[7]
        + stats[6]
    )

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
        np.abs(all_true - all_pred)
    )

    wind_mae = mean_absolute_error(
        wind_true,
        wind_pred,
    )

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

    cm = confusion_matrix(
        all_true,
        all_pred,
        labels=list(range(7)),
    )

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "category_mae": float(category_mae),
        "wind_mae": float(wind_mae),
        "wind_r2": float(wind_r2),
        "pressure_mae": float(pressure_mae),
        "pressure_r2": float(pressure_r2),
        "true": all_true,
        "pred": all_pred,
        "confusion_matrix": cm,
    }


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(
    fold_name,
    test_storms,
    all_df,
    device,
):
    seed_all(SEED)

    test_storms = list(test_storms)

    remaining = [
        s for s in STORMS
        if s not in test_storms
    ]

    # Deterministic validation selection.
    # We choose storms with useful sample counts while keeping
    # the selection fixed for reproducibility.
    val_storms = sorted(
        remaining,
        key=lambda s: (
            -len(
                all_df[
                    all_df["storm"] == s
                ]
            ),
            s,
        ),
    )[:VAL_STORMS_PER_FOLD]

    train_storms = [
        s for s in remaining
        if s not in val_storms
    ]

    train_df = all_df[
        all_df["storm"].isin(train_storms)
    ].copy()

    val_df = all_df[
        all_df["storm"].isin(val_storms)
    ].copy()

    test_df = all_df[
        all_df["storm"].isin(test_storms)
    ].copy()

    print("\n" + "=" * 78)
    print(f"{fold_name}")
    print("=" * 78)

    print("TRAIN:", train_storms, len(train_df))
    print("VAL  :", val_storms, len(val_df))
    print("TEST :", test_storms, len(test_df))

    print("\nTEST CATEGORY SUPPORT")

    support = (
        test_df["canonical_category"]
        .value_counts()
        .reindex(
            CATEGORIES,
            fill_value=0,
        )
    )

    print(support)

    stats = stats_from_train(
        train_df
    )

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

    model = INSATOrdinalCNN(
        in_channels=2
    ).to(device)

    pos_weight = ordinal_pos_weights(
        train_df
    ).to(device)

    ordinal_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    checkpoint_dir = (
        ROOT
        / "checkpoints"
        / "insat_v3_cv"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / f"{fold_name.lower()}_best.pt"
    )

    best_val_loss = float("inf")
    stale = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        running_loss = 0.0

        for (
            x,
            category,
            ordinal,
            wind,
            pressure,
        ) in train_loader:

            x = x.to(device)
            ordinal = ordinal.to(device)
            wind = wind.to(device)
            pressure = pressure.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                ordinal_logits,
                wind_hat,
                pressure_hat,
            ) = model(x)

            ordinal_loss = ordinal_loss_fn(
                ordinal_logits,
                ordinal,
            )

            wind_loss = nn.functional.smooth_l1_loss(
                wind_hat,
                wind,
            )

            pressure_loss = nn.functional.smooth_l1_loss(
                pressure_hat,
                pressure,
            )

            loss = (
                ORDINAL_LOSS_WEIGHT * ordinal_loss
                +
                WIND_LOSS_WEIGHT * wind_loss
                +
                PRESSURE_LOSS_WEIGHT * pressure_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            running_loss += float(
                loss.item()
            )

        val = evaluate(
            model,
            val_loader,
            device,
            stats,
            ordinal_loss_fn,
        )

        scheduler.step(
            val["loss"]
        )

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            stale = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "categories":
                        CATEGORIES,
                    "fold":
                        fold_name,
                    "train_storms":
                        train_storms,
                    "val_storms":
                        val_storms,
                    "test_storms":
                        test_storms,
                    "stats":
                        stats,
                    "epoch":
                        epoch,
                    "val_metrics": {
                        k: v
                        for k, v in val.items()
                        if np.isscalar(v)
                    },
                },
                checkpoint_path,
            )

            marker = " <-- BEST"
        else:
            stale += 1
            marker = ""

        if (
            epoch == 1
            or epoch % 5 == 0
            or marker
        ):
            print(
                f"epoch {epoch:03d}/{EPOCHS} "
                f"train={running_loss / len(train_loader):.4f} "
                f"val={val['loss']:.4f} "
                f"val_acc={val['accuracy']:.3f} "
                f"val_f1={val['macro_f1']:.3f} "
                f"wind_mae={val['wind_mae']:.2f} "
                f"pressure_mae={val['pressure_mae']:.2f}"
                f"{marker}"
            )

        if stale >= PATIENCE:
            print("Early stopping.")
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test = evaluate(
        model,
        test_loader,
        device,
        stats,
        ordinal_loss_fn,
    )

    print("\nFINAL FOLD TEST")

    print(
        f"Accuracy     : {test['accuracy']:.4f}"
    )

    print(
        f"Macro F1     : {test['macro_f1']:.4f}"
    )

    print(
        f"Category MAE : {test['category_mae']:.4f}"
    )

    print(
        f"Wind MAE     : {test['wind_mae']:.4f} kt"
    )

    print(
        f"Wind R2      : {test['wind_r2']:.4f}"
    )

    print(
        f"Pressure MAE : {test['pressure_mae']:.4f} hPa"
    )

    print(
        f"Pressure R2  : {test['pressure_r2']:.4f}"
    )

    return {
        "fold": fold_name,
        "train_storms": train_storms,
        "val_storms": val_storms,
        "test_storms": test_storms,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "metrics": {
            "accuracy": test["accuracy"],
            "macro_f1": test["macro_f1"],
            "category_mae": test["category_mae"],
            "wind_mae_kt": test["wind_mae"],
            "wind_r2": test["wind_r2"],
            "pressure_mae_hpa": test["pressure_mae"],
            "pressure_r2": test["pressure_r2"],
        },
        "test_support": support.to_dict(),
        "confusion_matrix": test[
            "confusion_matrix"
        ].tolist(),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    seed_all()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        "\nLoading 10-storm INSAT dataset..."
    )

    df = load_all()

    print(
        f"Total samples: {len(df)}"
    )

    print(
        "\nStorm sample counts:"
    )

    print(
        df["storm"].value_counts()
        .reindex(STORMS)
    )

    print(
        "\nCanonical category counts:"
    )

    print(
        df["canonical_category"]
        .value_counts()
        .reindex(
            CATEGORIES,
            fill_value=0,
        )
    )

    results = []

    for fold_name, test_storms in FOLDS:
        result = train_fold(
            fold_name,
            test_storms,
            df,
            device,
        )

        results.append(result)

    # ========================================================
    # AGGREGATE
    # ========================================================

    metric_names = [
        "accuracy",
        "macro_f1",
        "category_mae",
        "wind_mae_kt",
        "wind_r2",
        "pressure_mae_hpa",
        "pressure_r2",
    ]

    aggregate = {}

    for metric in metric_names:
        values = np.array(
            [
                r["metrics"][metric]
                for r in results
            ],
            dtype=float,
        )

        aggregate[metric] = {
            "mean": float(
                np.nanmean(values)
            ),
            "std": float(
                np.nanstd(values)
            ),
            "values": [
                None if not np.isfinite(v)
                else float(v)
                for v in values
            ],
        }

    # ========================================================
    # MICRO AGGREGATION FOR BASIC CLASSIFICATION
    # ========================================================

    total_correct = 0
    total_samples = 0

    combined_cm = np.zeros(
        (7, 7),
        dtype=int,
    )

    for r in results:
        cm = np.asarray(
            r["confusion_matrix"],
            dtype=int,
        )

        combined_cm += cm

        total_correct += int(
            np.trace(cm)
        )

        total_samples += int(
            cm.sum()
        )

    pooled_accuracy = (
        total_correct
        /
        max(total_samples, 1)
    )

    aggregate[
        "pooled_accuracy"
    ] = float(
        pooled_accuracy
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    output = {
        "model":
            "INSAT V3 Ordinal CNN",
        "dataset":
            "NIO 10-storm INSAT",
        "total_samples":
            int(len(df)),
        "categories":
            CATEGORIES,
        "folds":
            results,
        "aggregate":
            aggregate,
        "pooled_confusion_matrix":
            combined_cm.tolist(),
        "notes": [
            "All splits are storm-level.",
            "No sample/image is shared between train, validation, and test.",
            "Metrics are reported per fold and as mean/std across folds.",
            "Some classes have very low support, especially Super Cyclonic Storm.",
            "A class with zero test support in a fold cannot have meaningful recall in that fold.",
        ],
    }

    report_path = (
        ROOT
        / "data"
        / "processed"
        / "insat_v3_cross_validation.json"
    )

    report_path.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "INSAT V3 CROSS-VALIDATION SUMMARY"
    )

    print(
        "=" * 78
    )

    for metric in metric_names:
        a = aggregate[metric]

        print(
            f"{metric:22s}: "
            f"{a['mean']:.4f} "
            f"+/- "
            f"{a['std']:.4f}"
        )

    print(
        f"{'pooled_accuracy':22s}: "
        f"{aggregate['pooled_accuracy']:.4f}"
    )

    print(
        "\nPooled confusion matrix:"
    )

    print(
        combined_cm
    )

    print(
        "\nReport:"
    )

    print(
        report_path
    )

    print(
        "\nCV COMPLETE."
    )


if __name__ == "__main__":
    main()
