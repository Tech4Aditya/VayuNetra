import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from backend.data.tcir_dataset import TCIRDataset
from backend.models.tcir_intensity import TCIRIntensityCNN


# ============================================================
# CONFIG
# ============================================================

H5_PATH = "backend/data/raw/tcir/Cyclone_Images.h5"

TRAIN_CSV = "backend/data/labels/tcir_train.csv"
VAL_CSV = "backend/data/labels/tcir_val.csv"
TEST_CSV = "backend/data/labels/tcir_test.csv"

CHECKPOINT_DIR = "backend/checkpoints"

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "tcir_intensity_best.pt"
)

SEED = 42

BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

CHANNELS = (
    "IR",
    "WV",
    "VIS",
    "PMW",
)

# Loss weights
WIND_WEIGHT = 1.0
PRESSURE_WEIGHT = 0.5
SIZE_WEIGHT = 0.25


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("CUDA available.")

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        device = torch.device("cpu")

        print("CUDA not available.")
        print("Using CPU.")

    return device


# ============================================================
# TARGET NORMALIZATION
# ============================================================

# These are deliberately fixed physical-scale values.
#
# Wind:
#   approximately 10 - 168 kt
#
# Pressure:
#   approximately 879 - 1024 hPa
#
# Size:
#   approximately 0 - 375 nmi
#
# Normalizing targets makes the three losses comparable.

TARGET_STATS = {
    "wind": {
        "mean": 51.0,
        "std": 27.0,
    },

    "pressure": {
        "mean": 988.4,
        "std": 20.0,
    },

    "size": {
        "mean": 54.5,
        "std": 55.0,
    },
}


def normalize_target(value, name):

    mean = TARGET_STATS[name]["mean"]

    std = TARGET_STATS[name]["std"]

    return (
        value - mean
    ) / std


def denormalize_target(value, name):

    mean = TARGET_STATS[name]["mean"]

    std = TARGET_STATS[name]["std"]

    return (
        value * std
    ) + mean


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(preds, targets):

    preds = np.asarray(preds)

    targets = np.asarray(targets)

    mae = np.mean(
        np.abs(preds - targets)
    )

    rmse = np.sqrt(
        np.mean(
            (preds - targets) ** 2
        )
    )

    ss_res = np.sum(
        (targets - preds) ** 2
    )

    ss_tot = np.sum(
        (targets - np.mean(targets)) ** 2
    )

    if ss_tot == 0:

        r2 = 0.0

    else:

        r2 = 1.0 - (
            ss_res / ss_tot
        )

    return mae, rmse, r2


# ============================================================
# DATA
# ============================================================

def create_datasets():

    train_dataset = TCIRDataset(
        manifest_path=TRAIN_CSV,
        h5_path=H5_PATH,
        channels=CHANNELS,
    )

    val_dataset = TCIRDataset(
        manifest_path=VAL_CSV,
        h5_path=H5_PATH,
        channels=CHANNELS,
    )

    test_dataset = TCIRDataset(
        manifest_path=TEST_CSV,
        h5_path=H5_PATH,
        channels=CHANNELS,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# DATALOADERS
# ============================================================

def create_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# LOSS
# ============================================================

def calculate_loss(
    outputs,
    batch,
    device,
):

    wind = batch["wind_kt"].to(device)

    pressure = batch[
        "pressure_hpa"
    ].to(device)

    size = batch[
        "size_nmi"
    ].to(device)

    # Normalize targets
    wind = normalize_target(
        wind,
        "wind"
    )

    pressure = normalize_target(
        pressure,
        "pressure"
    )

    size = normalize_target(
        size,
        "size"
    )

    criterion = nn.SmoothL1Loss()

    wind_loss = criterion(
        outputs["wind"],
        wind,
    )

    pressure_loss = criterion(
        outputs["pressure"],
        pressure,
    )

    size_loss = criterion(
        outputs["size"],
        size,
    )

    total_loss = (
        WIND_WEIGHT * wind_loss
        +
        PRESSURE_WEIGHT * pressure_loss
        +
        SIZE_WEIGHT * size_loss
    )

    return (
        total_loss,
        wind_loss,
        pressure_loss,
        size_loss,
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0

    for batch_idx, batch in enumerate(loader):

        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        (
            loss,
            wind_loss,
            pressure_loss,
            size_loss,
        ) = calculate_loss(
            outputs,
            batch,
            device,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        running_loss += loss.item()

        if (
            batch_idx + 1
        ) % 50 == 0:

            print(
                f"    Batch "
                f"{batch_idx + 1}/"
                f"{len(loader)} "
                f"Loss={loss.item():.4f}"
            )

    return (
        running_loss
        / len(loader)
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    total_loss = 0.0

    wind_preds = []
    wind_targets = []

    pressure_preds = []
    pressure_targets = []

    size_preds = []
    size_targets = []

    for batch in loader:

        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        (
            loss,
            _,
            _,
            _,
        ) = calculate_loss(
            outputs,
            batch,
            device,
        )

        total_loss += loss.item()

        # ----------------------------------------
        # Wind
        # ----------------------------------------

        wind_pred = denormalize_target(
            outputs["wind"]
            .cpu()
            .numpy(),
            "wind",
        )

        wind_true = (
            batch["wind_kt"]
            .numpy()
        )

        wind_preds.extend(
            wind_pred.tolist()
        )

        wind_targets.extend(
            wind_true.tolist()
        )

        # ----------------------------------------
        # Pressure
        # ----------------------------------------

        pressure_pred = denormalize_target(
            outputs["pressure"]
            .cpu()
            .numpy(),
            "pressure",
        )

        pressure_true = (
            batch["pressure_hpa"]
            .numpy()
        )

        pressure_preds.extend(
            pressure_pred.tolist()
        )

        pressure_targets.extend(
            pressure_true.tolist()
        )

        # ----------------------------------------
        # Size
        # ----------------------------------------

        size_pred = denormalize_target(
            outputs["size"]
            .cpu()
            .numpy(),
            "size",
        )

        size_true = (
            batch["size_nmi"]
            .numpy()
        )

        size_preds.extend(
            size_pred.tolist()
        )

        size_targets.extend(
            size_true.tolist()
        )

    wind_metrics = calculate_metrics(
        wind_preds,
        wind_targets,
    )

    pressure_metrics = calculate_metrics(
        pressure_preds,
        pressure_targets,
    )

    size_metrics = calculate_metrics(
        size_preds,
        size_targets,
    )

    return {
        "loss": (
            total_loss
            / len(loader)
        ),

        "wind": wind_metrics,

        "pressure": pressure_metrics,

        "size": size_metrics,
    }


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    name,
    metrics,
):

    wind_mae, wind_rmse, wind_r2 = (
        metrics["wind"]
    )

    pressure_mae, pressure_rmse, pressure_r2 = (
        metrics["pressure"]
    )

    size_mae, size_rmse, size_r2 = (
        metrics["size"]
    )

    print(
        f"\n{name}"
    )

    print(
        f"  Loss: {metrics['loss']:.4f}"
    )

    print(
        f"  Wind     | "
        f"MAE={wind_mae:.2f} kt | "
        f"RMSE={wind_rmse:.2f} | "
        f"R²={wind_r2:.4f}"
    )

    print(
        f"  Pressure | "
        f"MAE={pressure_mae:.2f} hPa | "
        f"RMSE={pressure_rmse:.2f} | "
        f"R²={pressure_r2:.4f}"
    )

    print(
        f"  Size     | "
        f"MAE={size_mae:.2f} nmi | "
        f"RMSE={size_rmse:.2f} | "
        f"R²={size_r2:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "VAYUNETRA - TCIR REAL SATELLITE TRAINING"
    )

    print("=" * 70)

    set_seed(SEED)

    device = get_device()

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    print(
        "\nChannels:",
        CHANNELS
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Epochs:",
        EPOCHS
    )

    # --------------------------------------------
    # Dataset
    # --------------------------------------------

    print(
        "\nLoading datasets..."
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_datasets()

    print(
        "Train frames:",
        len(train_dataset)
    )

    print(
        "Val frames:",
        len(val_dataset)
    )

    print(
        "Test frames:",
        len(test_dataset)
    )

    # --------------------------------------------
    # Loaders
    # --------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    # --------------------------------------------
    # Model
    # --------------------------------------------

    model = TCIRIntensityCNN(
        in_channels=len(CHANNELS)
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTrainable parameters:",
        f"{parameters:,}"
    )

    # --------------------------------------------
    # Optimizer
    # --------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    # --------------------------------------------
    # Training
    # --------------------------------------------

    best_val_loss = float("inf")

    patience = 7

    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        print(
            "=" * 70
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Learning rate: {current_lr:.7f}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        scheduler.step(
            val_metrics["loss"]
        )

        print(
            f"\nTrain loss: "
            f"{train_loss:.4f}"
        )

        print_metrics(
            "VALIDATION",
            val_metrics,
        )

        # ----------------------------------------
        # Save best
        # ----------------------------------------

        if (
            val_metrics["loss"]
            < best_val_loss
        ):

            best_val_loss = (
                val_metrics["loss"]
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "epoch":
                        epoch,

                    "val_loss":
                        best_val_loss,

                    "channels":
                        CHANNELS,

                    "target_stats":
                        TARGET_STATS,
                },
                BEST_MODEL_PATH,
            )

            print(
                "\n✓ BEST MODEL SAVED"
            )

            print(
                BEST_MODEL_PATH
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\nNo improvement "
                f"({epochs_without_improvement}/"
                f"{patience})"
            )

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "\nEarly stopping."
            )

            break

    # ========================================================
    # TEST
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING BEST MODEL"
    )

    print(
        "=" * 70
    )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "Best epoch:",
        checkpoint["epoch"]
    )

    print(
        "Best validation loss:",
        checkpoint["val_loss"]
    )

    test_metrics = evaluate(
        model,
        test_loader,
        device,
    )

    print_metrics(
        "FINAL TEST RESULTS",
        test_metrics,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()