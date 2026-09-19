# backend/train_insat_real.py

from pathlib import Path
import random
import csv

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, mean_absolute_error

from models.classifier import CycloneCNN


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
LABEL_DIR = DATA_DIR / "labels"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MODEL = CHECKPOINT_DIR / "classifier_insat.pt"
OUTPUT_STATS = DATA_DIR / "processed" / "insat_training_stats.npz"

SEED = 42

BATCH_SIZE = 8
EPOCHS = 50

LR = 1e-3
WEIGHT_DECAY = 1e-4

NUM_CATEGORIES = 7
IN_CHANNELS = 2

IMAGE_SIZE = 128

WIND_MIN = 20.0
WIND_MAX = 130.0


# ============================================================
# STORM SPLIT
# ============================================================

TRAIN_STORMS = [
    "AMPHAN",
    "NISARGA",
    "YAAS",
    "GULAAB",
    "ASANI",
    "BULBUL",
    "TAUKTAE",
]

VAL_STORMS = [
    "BIPARJOY",
]

TEST_STORMS = [
    "MICHAUNG",
]


# ============================================================
# CATEGORY MAPPING
# ============================================================
#
# Supports both:
#
# Full names:
# Depression
# Deep Depression
# Cyclonic Storm
# ...
#
# AND MOSDAC/TAUKTAE abbreviations:
# D
# DD
# CS
# SCS
# VSCS
# ESCS
# SuCS / SUCS
#
# ============================================================

CATEGORY_MAP = {

    # Category 0
    "D": 0,
    "Depression": 0,

    # Category 1
    "DD": 1,
    "Deep Depression": 1,

    # Category 2
    "CS": 2,
    "Cyclonic Storm": 2,

    # Category 3
    "SCS": 3,
    "Severe Cyclonic Storm": 3,

    # Category 4
    "VSCS": 4,
    "Very Severe Cyclonic Storm": 4,

    # Category 5
    "ESCS": 5,
    "Extremely Severe Cyclonic Storm": 5,

    # Category 6
    "SuCS": 6,
    "SUCS": 6,
    "Super Cyclonic Storm": 6,
}


CATEGORY_NAMES = [
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm",
    "Super Cyclonic Storm",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("INSAT REAL DATA TRAINING")
print("=" * 70)

print(f"Device: {DEVICE}")

print(
    f"Train storms: {', '.join(TRAIN_STORMS)}"
)

print(
    f"Val storms:   {', '.join(VAL_STORMS)}"
)

print(
    f"Test storms:  {', '.join(TEST_STORMS)}"
)

print("=" * 70)


# ============================================================
# MANIFEST
# ============================================================

def find_manifest(storm):

    path = (
        LABEL_DIR /
        f"{storm.lower()}_insat_manifest.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Manifest not found for {storm}:\n{path}"
        )

    return path


def load_manifest(storm):

    manifest_path = find_manifest(storm)

    rows = []

    with open(
        manifest_path,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


# ============================================================
# PROCESSED FILE PATH
# ============================================================

def processed_paths(storm, h5_path):

    """
    Manifest:

    .../3DIMG_17MAY2020_1500_L1C_ASIA_MER_V01R00.h5

    Processed:

    processed/
        insat_amphan_calibrated/
            3DIMG_17MAY2020_1500_L1C_ASIA_MER_V01R00/
                tir1_radiance.npy
                wv_radiance.npy
    """

    sample_name = Path(h5_path).stem

    storm_dir = (
        PROCESSED_DIR /
        f"insat_{storm.lower()}_calibrated"
    )

    sample_dir = (
        storm_dir /
        sample_name
    )

    tir_path = (
        sample_dir /
        "tir1_radiance.npy"
    )

    wv_path = (
        sample_dir /
        "wv_radiance.npy"
    )

    return tir_path, wv_path


# ============================================================
# COLLECT SAMPLES
# ============================================================

def collect_samples(storms):

    samples = []

    for storm in storms:

        rows = load_manifest(storm)

        print(
            f"\n{storm}: manifest rows = {len(rows)}"
        )

        valid = 0
        invalid = 0

        for row in rows:

            try:

                h5_path = row["file"]

                tir_path, wv_path = (
                    processed_paths(
                        storm,
                        h5_path
                    )
                )

                # --------------------------------------------
                # Check TIR
                # --------------------------------------------

                if not tir_path.exists():

                    invalid += 1

                    print(
                        f"  Missing TIR: {tir_path}"
                    )

                    continue

                # --------------------------------------------
                # Check WV
                # --------------------------------------------

                if not wv_path.exists():

                    invalid += 1

                    print(
                        f"  Missing WV: {wv_path}"
                    )

                    continue

                # --------------------------------------------
                # Category
                # --------------------------------------------

                category_raw = (
                    row["category"]
                    .strip()
                )

                if category_raw not in CATEGORY_MAP:

                    invalid += 1

                    print(
                        f"  Unknown category: "
                        f"{category_raw}"
                    )

                    continue

                category = CATEGORY_MAP[
                    category_raw
                ]

                # --------------------------------------------
                # Numeric labels
                # --------------------------------------------

                wind = float(
                    row["wind_kt"]
                )

                pressure = float(
                    row["pressure_hpa"]
                )

                # --------------------------------------------
                # Sample
                # --------------------------------------------

                sample = {

                    "storm": storm,

                    "tir_path": tir_path,

                    "wv_path": wv_path,

                    "wind": wind,

                    "pressure": pressure,

                    "category": category,

                    "category_name":
                        CATEGORY_NAMES[category],

                    "timestamp":
                        row["insat_timestamp"],
                }

                samples.append(sample)

                valid += 1

            except Exception as e:

                invalid += 1

                print(
                    f"  Error processing row: {e}"
                )

        print(
            f"  Valid processed samples: {valid}"
        )

        print(
            f"  Missing/invalid: {invalid}"
        )

    return samples


# ============================================================
# DATASET
# ============================================================

class INSATDataset(Dataset):

    def __init__(
        self,
        samples,
        tir_mean,
        tir_std,
        wv_mean,
        wv_std,
    ):

        self.samples = samples

        self.tir_mean = tir_mean
        self.tir_std = tir_std

        self.wv_mean = wv_mean
        self.wv_std = wv_std


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, idx):

        sample = self.samples[idx]

        # ----------------------------------------------------
        # Load calibrated channels
        # ----------------------------------------------------

        tir = np.load(
            sample["tir_path"]
        ).astype(np.float32)

        wv = np.load(
            sample["wv_path"]
        ).astype(np.float32)

        # ----------------------------------------------------
        # Remove unnecessary dimensions
        # ----------------------------------------------------

        tir = np.squeeze(tir)
        wv = np.squeeze(wv)

        # ----------------------------------------------------
        # Validate shapes
        # ----------------------------------------------------

        if tir.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE
        ):

            raise ValueError(
                f"Unexpected TIR shape "
                f"{tir.shape} at "
                f"{sample['tir_path']}"
            )

        if wv.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE
        ):

            raise ValueError(
                f"Unexpected WV shape "
                f"{wv.shape} at "
                f"{sample['wv_path']}"
            )

        # ----------------------------------------------------
        # Normalize using TRAIN statistics
        # ----------------------------------------------------

        tir = (
            tir - self.tir_mean
        ) / self.tir_std

        wv = (
            wv - self.wv_mean
        ) / self.wv_std

        # ----------------------------------------------------
        # 2-channel input
        # ----------------------------------------------------

        x = np.stack(
            [tir, wv],
            axis=0
        )

        x = torch.tensor(
            x,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Normalize wind
        # ----------------------------------------------------

        wind = sample["wind"]

        wind_norm = (
            wind - WIND_MIN
        ) / (
            WIND_MAX - WIND_MIN
        )

        wind_norm = np.clip(
            wind_norm,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return (

            x,

            torch.tensor(
                sample["category"],
                dtype=torch.long
            ),

            torch.tensor(
                wind_norm,
                dtype=torch.float32
            ),
        )


# ============================================================
# NORMALIZATION
# ============================================================

def calculate_normalization(samples):

    print(
        "\nCalculating train-only normalization..."
    )

    tir_values = []
    wv_values = []

    for i, sample in enumerate(samples):

        tir = np.load(
            sample["tir_path"]
        ).astype(np.float32)

        wv = np.load(
            sample["wv_path"]
        ).astype(np.float32)

        tir = np.squeeze(tir)
        wv = np.squeeze(wv)

        # ----------------------------------------------------
        # Sample pixels to reduce memory usage
        # ----------------------------------------------------

        max_pixels = 10000

        tir_flat = tir.reshape(-1)
        wv_flat = wv.reshape(-1)

        if tir_flat.size > max_pixels:

            indices = np.random.choice(
                tir_flat.size,
                max_pixels,
                replace=False
            )

            tir_values.append(
                tir_flat[indices]
            )

            wv_values.append(
                wv_flat[indices]
            )

        else:

            tir_values.append(
                tir_flat
            )

            wv_values.append(
                wv_flat
            )

        if (i + 1) % 20 == 0:

            print(
                f"  Processed "
                f"{i + 1}/{len(samples)}"
            )

    tir_values = np.concatenate(
        tir_values
    )

    wv_values = np.concatenate(
        wv_values
    )

    tir_mean = float(
        np.mean(tir_values)
    )

    tir_std = float(
        np.std(tir_values)
    )

    wv_mean = float(
        np.mean(wv_values)
    )

    wv_std = float(
        np.std(wv_values)
    )

    # Prevent division by zero

    tir_std = max(
        tir_std,
        1e-6
    )

    wv_std = max(
        wv_std,
        1e-6
    )

    print("\nNormalization statistics:")

    print(
        f"TIR mean = {tir_mean:.6f}"
    )

    print(
        f"TIR std  = {tir_std:.6f}"
    )

    print(
        f"WV mean  = {wv_mean:.6f}"
    )

    print(
        f"WV std   = {wv_std:.6f}"
    )

    return (
        tir_mean,
        tir_std,
        wv_mean,
        wv_std,
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_dataset_summary(
    name,
    samples
):

    print(
        f"\n{name} dataset: "
        f"{len(samples)} samples"
    )

    storm_counts = {}
    category_counts = {}

    for sample in samples:

        storm = sample["storm"]

        category = sample["category"]

        storm_counts[storm] = (
            storm_counts.get(
                storm,
                0
            ) + 1
        )

        category_counts[category] = (
            category_counts.get(
                category,
                0
            ) + 1
        )

    print("Storm distribution:")

    for storm, count in (
        storm_counts.items()
    ):

        print(
            f"  {storm:10s}: {count}"
        )

    print("Category distribution:")

    for category, count in sorted(
        category_counts.items()
    ):

        print(
            f"  "
            f"{CATEGORY_NAMES[category]:35s}: "
            f"{count}"
        )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(samples):

    counts = np.zeros(
        NUM_CATEGORIES,
        dtype=np.float32
    )

    for sample in samples:

        counts[
            sample["category"]
        ] += 1

    print(
        "\nTraining category distribution:"
    )

    for i in range(NUM_CATEGORIES):

        print(
            f"  {i}: "
            f"{CATEGORY_NAMES[i]:35s} "
            f"{int(counts[i])}"
        )

    # --------------------------------------------------------
    # Inverse frequency weights
    # --------------------------------------------------------

    weights = np.zeros_like(
        counts
    )

    total = np.sum(counts)

    for i in range(NUM_CATEGORIES):

        if counts[i] > 0:

            weights[i] = (
                total /
                (
                    NUM_CATEGORIES *
                    counts[i]
                )
            )

        else:

            weights[i] = 0.0

    print(
        "\nClass weights:"
    )

    for i in range(NUM_CATEGORIES):

        print(
            f"  "
            f"{CATEGORY_NAMES[i]:35s} "
            f"{weights[i]:.4f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    category_loss_fn,
    device,
):

    model.train()

    total_loss = 0.0

    total_category_loss = 0.0

    total_intensity_loss = 0.0

    all_preds = []

    all_targets = []

    for (
        x,
        category,
        wind
    ) in loader:

        x = x.to(device)

        category = category.to(device)

        wind = wind.to(device)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        (
            presence_logit,
            category_logits,
            intensity
        ) = model(x)

        # ----------------------------------------------------
        # Category loss
        # ----------------------------------------------------

        category_loss = category_loss_fn(
            category_logits,
            category
        )

        # ----------------------------------------------------
        # Intensity loss
        # ----------------------------------------------------

        intensity_loss = (
            nn.functional.smooth_l1_loss(
                intensity.squeeze(-1),
                wind
            )
        )

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        loss = (
            category_loss +
            intensity_loss
        )

        # ----------------------------------------------------
        # Backprop
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        total_loss += loss.item()

        total_category_loss += (
            category_loss.item()
        )

        total_intensity_loss += (
            intensity_loss.item()
        )

        preds = torch.argmax(
            category_logits,
            dim=1
        )

        all_preds.extend(
            preds.detach()
            .cpu()
            .numpy()
        )

        all_targets.extend(
            category.detach()
            .cpu()
            .numpy()
        )

    accuracy = accuracy_score(
        all_targets,
        all_preds
    )

    n = len(loader)

    return {

        "loss":
            total_loss / n,

        "category_loss":
            total_category_loss / n,

        "intensity_loss":
            total_intensity_loss / n,

        "accuracy":
            accuracy,
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    total_loss = 0.0

    total_category_loss = 0.0

    total_intensity_loss = 0.0

    all_preds = []

    all_targets = []

    all_wind_pred = []

    all_wind_true = []

    for (
        x,
        category,
        wind
    ) in loader:

        x = x.to(device)

        category = category.to(device)

        wind = wind.to(device)

        (
            presence_logit,
            category_logits,
            intensity
        ) = model(x)

        # ----------------------------------------------------
        # Category loss
        # ----------------------------------------------------

        category_loss = (
            nn.functional.cross_entropy(
                category_logits,
                category
            )
        )

        # ----------------------------------------------------
        # Intensity loss
        # ----------------------------------------------------

        intensity_loss = (
            nn.functional.smooth_l1_loss(
                intensity.squeeze(-1),
                wind
            )
        )

        loss = (
            category_loss +
            intensity_loss
        )

        total_loss += loss.item()

        total_category_loss += (
            category_loss.item()
        )

        total_intensity_loss += (
            intensity_loss.item()
        )

        # ----------------------------------------------------
        # Category predictions
        # ----------------------------------------------------

        preds = torch.argmax(
            category_logits,
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_targets.extend(
            category.cpu().numpy()
        )

        # ----------------------------------------------------
        # Wind predictions
        # ----------------------------------------------------

        all_wind_pred.extend(
            intensity.squeeze(-1)
            .cpu()
            .numpy()
        )

        all_wind_true.extend(
            wind.cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_targets,
        all_preds
    )

    # --------------------------------------------------------
    # Convert normalized wind back to knots
    # --------------------------------------------------------

    wind_pred_kt = (
        np.array(all_wind_pred)
        *
        (WIND_MAX - WIND_MIN)
        +
        WIND_MIN
    )

    wind_true_kt = (
        np.array(all_wind_true)
        *
        (WIND_MAX - WIND_MIN)
        +
        WIND_MIN
    )

    wind_mae = mean_absolute_error(
        wind_true_kt,
        wind_pred_kt
    )

    n = len(loader)

    return {

        "loss":
            total_loss / n,

        "category_loss":
            total_category_loss / n,

        "intensity_loss":
            total_intensity_loss / n,

        "accuracy":
            accuracy,

        "wind_mae":
            wind_mae,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        "\nLoading manifests..."
    )

    train_samples = collect_samples(
        TRAIN_STORMS
    )

    val_samples = collect_samples(
        VAL_STORMS
    )

    test_samples = collect_samples(
        TEST_STORMS
    )

    if len(train_samples) == 0:

        raise RuntimeError(
            "No training samples found."
        )

    if len(val_samples) == 0:

        raise RuntimeError(
            "No validation samples found."
        )

    if len(test_samples) == 0:

        raise RuntimeError(
            "No test samples found."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print_dataset_summary(
        "TRAIN",
        train_samples
    )

    print_dataset_summary(
        "VAL",
        val_samples
    )

    print_dataset_summary(
        "TEST",
        test_samples
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    (
        tir_mean,
        tir_std,
        wv_mean,
        wv_std,
    ) = calculate_normalization(
        train_samples
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = INSATDataset(
        train_samples,
        tir_mean,
        tir_std,
        wv_mean,
        wv_std,
    )

    val_dataset = INSATDataset(
        val_samples,
        tir_mean,
        tir_std,
        wv_mean,
        wv_std,
    )

    test_dataset = INSATDataset(
        test_samples,
        tir_mean,
        tir_std,
        wv_mean,
        wv_std,
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = CycloneCNN(
        num_categories=NUM_CATEGORIES,
        in_channels=IN_CHANNELS,
    ).to(DEVICE)

    print(
        "\nModel:"
    )

    print(model)

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = (
        calculate_class_weights(
            train_samples
        )
        .to(DEVICE)
    )

    category_loss_fn = (
        nn.CrossEntropyLoss(
            weight=class_weights
        )
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
    )

    # ========================================================
    # TRAINING
    # ========================================================

    best_val_loss = float("inf")

    best_epoch = 0

    history = []

    print("\n")

    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                category_loss_fn,
                DEVICE,
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = evaluate(
            model,
            val_loader,
            DEVICE,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_metrics["loss"]
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"\nEpoch "
            f"{epoch:02d}/{EPOCHS}"
        )

        print(
            f"  "
            f"train_loss="
            f"{train_metrics['loss']:.4f} "
            f"| train_cat="
            f"{train_metrics['category_loss']:.4f} "
            f"| train_int="
            f"{train_metrics['intensity_loss']:.4f} "
            f"| train_acc="
            f"{train_metrics['accuracy']:.3f}"
        )

        print(
            f"  "
            f"val_loss="
            f"{val_metrics['loss']:.4f} "
            f"| val_cat="
            f"{val_metrics['category_loss']:.4f} "
            f"| val_int="
            f"{val_metrics['intensity_loss']:.4f} "
            f"| val_acc="
            f"{val_metrics['accuracy']:.3f} "
            f"| wind_MAE="
            f"{val_metrics['wind_mae']:.2f} kt"
        )

        print(
            f"  lr={current_lr:.6g}"
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history.append([
            epoch,

            train_metrics["loss"],
            train_metrics["category_loss"],
            train_metrics["intensity_loss"],
            train_metrics["accuracy"],

            val_metrics["loss"],
            val_metrics["category_loss"],
            val_metrics["intensity_loss"],
            val_metrics["accuracy"],
            val_metrics["wind_mae"],
        ])

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if (
            val_metrics["loss"]
            <
            best_val_loss
        ):

            best_val_loss = (
                val_metrics["loss"]
            )

            best_epoch = epoch

            torch.save(
                {

                    "model_state_dict":
                        model.state_dict(),

                    "num_categories":
                        NUM_CATEGORIES,

                    "in_channels":
                        IN_CHANNELS,

                    "category_names":
                        CATEGORY_NAMES,

                    "category_map":
                        CATEGORY_MAP,

                    "tir_mean":
                        tir_mean,

                    "tir_std":
                        tir_std,

                    "wv_mean":
                        wv_mean,

                    "wv_std":
                        wv_std,

                    "wind_min":
                        WIND_MIN,

                    "wind_max":
                        WIND_MAX,

                    "epoch":
                        epoch,

                    "val_loss":
                        val_metrics["loss"],

                    "val_accuracy":
                        val_metrics["accuracy"],

                    "val_wind_mae":
                        val_metrics["wind_mae"],
                },

                OUTPUT_MODEL,
            )

            print(
                f"  ★ BEST MODEL SAVED "
                f"(epoch {epoch})"
            )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\n")

    print("=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    checkpoint = torch.load(
        OUTPUT_MODEL,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best val loss: "
        f"{checkpoint['val_loss']:.4f}"
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    final_val = evaluate(
        model,
        val_loader,
        DEVICE,
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    final_test = evaluate(
        model,
        test_loader,
        DEVICE,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")

    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        "\nValidation — BIPARJOY"
    )

    print(
        f"Loss: "
        f"{final_val['loss']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{final_val['accuracy'] * 100:.2f}%"
    )

    print(
        f"Wind MAE: "
        f"{final_val['wind_mae']:.2f} kt"
    )

    print(
        "\nTest — MICHAUNG"
    )

    print(
        f"Loss: "
        f"{final_test['loss']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{final_test['accuracy'] * 100:.2f}%"
    )

    print(
        f"Wind MAE: "
        f"{final_test['wind_mae']:.2f} kt"
    )

    # ========================================================
    # SAVE STATS
    # ========================================================

    history = np.array(
        history,
        dtype=np.float32
    )

    np.savez(
        OUTPUT_STATS,

        history=history,

        tir_mean=tir_mean,
        tir_std=tir_std,

        wv_mean=wv_mean,
        wv_std=wv_std,

        train_count=len(
            train_samples
        ),

        val_count=len(
            val_samples
        ),

        test_count=len(
            test_samples
        ),

        best_epoch=best_epoch,

        best_val_loss=best_val_loss,

        final_val_accuracy=(
            final_val["accuracy"]
        ),

        final_val_wind_mae=(
            final_val["wind_mae"]
        ),

        final_test_accuracy=(
            final_test["accuracy"]
        ),

        final_test_wind_mae=(
            final_test["wind_mae"]
        ),
    )

    # ========================================================
    # DONE
    # ========================================================

    print("\n")

    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Model saved to:\n"
        f"{OUTPUT_MODEL}"
    )

    print(
        f"\nTraining stats saved to:\n"
        f"{OUTPUT_STATS}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()