import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.data.tcir_temporal_dataset import TCIRTemporalDataset
from backend.models.tcir_temporal import TCIRTemporalModel


SEED = 42

H5_PATH = "backend/data/raw/tcir/Cyclone_Images.h5"

TRAIN_MANIFEST = "backend/data/labels/tcir_temporal_train.csv"
VAL_MANIFEST = "backend/data/labels/tcir_temporal_val.csv"
TEST_MANIFEST = "backend/data/labels/tcir_temporal_test.csv"

CHECKPOINT_PATH = "backend/checkpoints/tcir_temporal_best.pt"

CHANNELS = ("IR", "PMW")

# Same normalization used by the single-frame experiment
TARGET_STATS = {
    "wind": (51.0, 27.0),
    "pressure": (988.4, 20.0),
    "size": (54.5, 55.0),
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_target(value, name):
    mean, std = TARGET_STATS[name]
    return (value - mean) / std


def denormalize_target(value, name):
    mean, std = TARGET_STATS[name]
    return value * std + mean


def create_dataset(manifest_path):
    return TCIRTemporalDataset(
        manifest_path=manifest_path,
        h5_path=H5_PATH,
        channels=CHANNELS
    )


def create_loaders(batch_size):

    train_dataset = create_dataset(TRAIN_MANIFEST)
    val_dataset = create_dataset(VAL_MANIFEST)
    test_dataset = create_dataset(TEST_MANIFEST)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    )


def calculate_metrics(predictions, targets):

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    mae = np.mean(np.abs(predictions - targets))

    rmse = np.sqrt(
        np.mean((predictions - targets) ** 2)
    )

    ss_res = np.sum(
        (targets - predictions) ** 2
    )

    ss_tot = np.sum(
        (targets - np.mean(targets)) ** 2
    )

    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return mae, rmse, r2


def run_epoch(model, loader, optimizer, criterion, device, training=True):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    wind_preds = []
    wind_targets = []

    pressure_preds = []
    pressure_targets = []

    size_preds = []
    size_targets = []

    for batch in loader:

        frames = batch["frames"].to(device)

        wind = batch["wind_kt"].to(device)
        pressure = batch["pressure_hpa"].to(device)
        size = batch["size_nmi"].to(device)

        # Normalize targets
        wind_norm = normalize_target(wind, "wind")
        pressure_norm = normalize_target(
            pressure, "pressure"
        )
        size_norm = normalize_target(size, "size")

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            output = model(frames)

            wind_loss = criterion(
                output["wind"],
                wind_norm
            )

            pressure_loss = criterion(
                output["pressure"],
                pressure_norm
            )

            size_loss = criterion(
                output["size"],
                size_norm
            )

            loss = (
                1.0 * wind_loss
                + 0.5 * pressure_loss
                + 0.25 * size_loss
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

        total_loss += loss.item() * frames.size(0)

        # Convert predictions back to physical units
        wind_pred = denormalize_target(
            output["wind"].detach().cpu().numpy(),
            "wind"
        )

        pressure_pred = denormalize_target(
            output["pressure"].detach().cpu().numpy(),
            "pressure"
        )

        size_pred = denormalize_target(
            output["size"].detach().cpu().numpy(),
            "size"
        )

        wind_preds.extend(wind_pred)
        wind_targets.extend(
            wind.detach().cpu().numpy()
        )

        pressure_preds.extend(pressure_pred)
        pressure_targets.extend(
            pressure.detach().cpu().numpy()
        )

        size_preds.extend(size_pred)
        size_targets.extend(
            size.detach().cpu().numpy()
        )

    n = len(loader.dataset)

    wind_metrics = calculate_metrics(
        wind_preds,
        wind_targets
    )

    pressure_metrics = calculate_metrics(
        pressure_preds,
        pressure_targets
    )

    size_metrics = calculate_metrics(
        size_preds,
        size_targets
    )

    return {
        "loss": total_loss / n,

        "wind": wind_metrics,

        "pressure": pressure_metrics,

        "size": size_metrics
    }


def print_metrics(prefix, metrics):

    wind_mae, wind_rmse, wind_r2 = metrics["wind"]
    pressure_mae, pressure_rmse, pressure_r2 = metrics["pressure"]
    size_mae, size_rmse, size_r2 = metrics["size"]

    print(
        f"{prefix} | "
        f"Loss {metrics['loss']:.4f} | "
        f"Wind MAE {wind_mae:.2f} | "
        f"Pressure MAE {pressure_mae:.2f} | "
        f"Size MAE {size_mae:.2f}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7
    )

    args = parser.parse_args()

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("TCIR TEMPORAL INTENSITY TRAINING")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"Channels: {CHANNELS}")
    print("Sequence length: 4")
    print("Interval: 3 hours")
    print("History: 9 hours")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    ) = create_loaders(args.batch_size)

    print()
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences:   {len(val_dataset)}")
    print(f"Test sequences:  {len(test_dataset)}")

    model = TCIRTemporalModel(
        in_channels=len(CHANNELS)
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {parameters:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    os.makedirs(
        os.path.dirname(CHECKPOINT_PATH),
        exist_ok=True
    )

    for epoch in range(1, args.epochs + 1):

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            training=True
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            device,
            training=False
        )

        scheduler.step(val_metrics["loss"])

        lr = optimizer.param_groups[0]["lr"]

        print()
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Learning rate: {lr:.6f}")

        print_metrics("TRAIN", train_metrics)
        print_metrics("VAL  ", val_metrics)

        if val_metrics["loss"] < best_val_loss:

            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "channels": CHANNELS,
                    "sequence_length": 4,
                    "interval_hours": 3,
                    "target_stats": TARGET_STATS,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch
                },
                CHECKPOINT_PATH
            )

            print(
                f"✓ Best checkpoint saved "
                f"(val_loss={best_val_loss:.4f})"
            )

        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print()
                print("Early stopping.")
                break

    print()
    print("=" * 70)
    print("LOADING BEST CHECKPOINT")
    print("=" * 70)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best epoch: {checkpoint['epoch']}"
    )

    print(
        f"Best validation loss: "
        f"{checkpoint['best_val_loss']:.4f}"
    )

    print()
    print("=" * 70)
    print("FINAL TEST")
    print("=" * 70)

    test_metrics = run_epoch(
        model,
        test_loader,
        optimizer,
        criterion,
        device,
        training=False
    )

    print_metrics("TEST", test_metrics)

    print()
    print("Wind:")
    print(
        f"  MAE:  {test_metrics['wind'][0]:.2f} kt"
    )
    print(
        f"  RMSE: {test_metrics['wind'][1]:.2f} kt"
    )
    print(
        f"  R²:   {test_metrics['wind'][2]:.4f}"
    )

    print()
    print("Pressure:")
    print(
        f"  MAE:  {test_metrics['pressure'][0]:.2f} hPa"
    )
    print(
        f"  RMSE: {test_metrics['pressure'][1]:.2f} hPa"
    )
    print(
        f"  R²:   {test_metrics['pressure'][2]:.4f}"
    )

    print()
    print("Size:")
    print(
        f"  MAE:  {test_metrics['size'][0]:.2f} nmi"
    )
    print(
        f"  RMSE: {test_metrics['size'][1]:.2f} nmi"
    )
    print(
        f"  R²:   {test_metrics['size'][2]:.4f}"
    )


if __name__ == "__main__":
    main()