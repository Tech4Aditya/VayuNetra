"""
Train CycloneCNN on the real INSAT cyclone dataset.

Uses the existing calibrated outputs produced by:
    backend/data/scripts/process_insat_storm.py

Input:
    7 storm manifests in backend/data/labels/*_insat_manifest.csv

Tensor:
    (2, 128, 128)
    C0 = TIR1 radiance
    C1 = WV radiance

Important:
    The current INSAT collection contains cyclone-positive samples only.
    Therefore the presence head is NOT used as a meaningful classifier yet.
    This script trains the 7-class intensity category + wind regression.
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.classifier import CycloneCNN


ROOT = Path(__file__).resolve().parent

STORMS = [
    "AMPHAN",
    "NISARGA",
    "YAAS",
    "GULAAB",
    "ASANI",
    "BIPARJOY",
    "MICHAUNG",
]

# Deterministic storm-level split.
TRAIN_STORMS = ["AMPHAN", "NISARGA", "YAAS", "GULAAB", "ASANI"]
VAL_STORMS = ["BIPARJOY"]
TEST_STORMS = ["MICHAUNG"]

CATEGORIES = [
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm",
    "Super Cyclonic Storm",
]
CATEGORY_TO_ID = {name: i for i, name in enumerate(CATEGORIES)}

BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
INTENSITY_WEIGHT = 0.5
SEED = 42

# Existing model's intensity head is sigmoid -> [0, 1].
# Wind is normalized to the physical range used by this dataset.
WIND_MIN = 20.0
WIND_MAX = 130.0


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def manifest_path(storm):
    return ROOT / "data" / "labels" / f"{storm.lower()}_insat_manifest.csv"


def calibrated_dir(storm):
    return ROOT / "data" / "processed" / f"insat_{storm.lower()}_calibrated"


def load_manifest():
    frames = []

    for storm in STORMS:
        path = manifest_path(storm)

        if not path.exists():
            raise FileNotFoundError(f"Missing manifest: {path}")

        df = pd.read_csv(path)

        if len(df) == 0:
            raise ValueError(f"Empty manifest: {path}")

        # process_insat_storm.py writes "cyclone".
        if "cyclone" not in df.columns:
            raise ValueError(f"{path} has no 'cyclone' column")

        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    print("=" * 70)
    print("REAL INSAT DATASET")
    print("=" * 70)
    print(f"Total samples: {len(df)}")
    print("\nStorm distribution:")
    print(df["cyclone"].value_counts())

    print("\nCategory distribution:")
    print(df["category"].value_counts())

    return df


def add_paths_and_labels(df):
    df = df.copy()

    image_paths = []
    category_ids = []
    intensity_targets = []

    for _, row in df.iterrows():
        storm = str(row["cyclone"]).upper()
        filename = Path(str(row["file"])).stem

        sample_dir = calibrated_dir(storm) / filename

        tir = sample_dir / "tir1_radiance.npy"
        wv = sample_dir / "wv_radiance.npy"

        if not tir.exists() or not wv.exists():
            raise FileNotFoundError(
                f"Missing calibrated sample for {storm}/{filename}\n"
                f"TIR: {tir}\n"
                f"WV : {wv}"
            )

        category = str(row["category"]).strip()

        if category not in CATEGORY_TO_ID:
            raise ValueError(
                f"Unknown category: {category!r}\n"
                f"Known: {CATEGORIES}"
            )

        wind = float(row["wind_kt"])

        if not np.isfinite(wind):
            raise ValueError(f"Invalid wind: {wind}")

        # Keep target inside [0,1] for the existing sigmoid intensity head.
        intensity = np.clip(
            (wind - WIND_MIN) / (WIND_MAX - WIND_MIN),
            0.0,
            1.0,
        )

        image_paths.append((tir, wv))
        category_ids.append(CATEGORY_TO_ID[category])
        intensity_targets.append(intensity)

    df["image_paths"] = image_paths
    df["category_id"] = category_ids
    df["intensity_target"] = intensity_targets

    return df


def calculate_train_stats(df):
    sums = np.zeros(2, dtype=np.float64)
    sq_sums = np.zeros(2, dtype=np.float64)
    count = 0

    for _, row in df.iterrows():
        tir_path, wv_path = row["image_paths"]

        tir = np.load(tir_path).astype(np.float32)
        wv = np.load(wv_path).astype(np.float32)

        if tir.shape != (128, 128) or wv.shape != (128, 128):
            raise ValueError(
                f"Bad shape: {tir_path}\n"
                f"TIR={tir.shape}, WV={wv.shape}"
            )

        if not np.isfinite(tir).all() or not np.isfinite(wv).all():
            raise ValueError(f"NaN/Inf in {tir_path}")

        arrays = [tir, wv]

        for c, x in enumerate(arrays):
            sums[c] += float(x.sum())
            sq_sums[c] += float((x * x).sum())

        count += tir.size

    mean = sums / count
    var = sq_sums / count - mean * mean
    std = np.sqrt(np.maximum(var, 1e-12))

    return mean.astype(np.float32), std.astype(np.float32)


class INSATDataset(Dataset):
    def __init__(self, df, mean, std):
        self.df = df.reset_index(drop=True)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        tir_path, wv_path = row["image_paths"]

        tir = np.load(tir_path).astype(np.float32)
        wv = np.load(wv_path).astype(np.float32)

        image = np.stack(
            [
                (tir - self.mean[0]) / self.std[0],
                (wv - self.mean[1]) / self.std[1],
            ],
            axis=0,
        ).astype(np.float32)

        return (
            torch.from_numpy(image),
            torch.tensor(int(row["category_id"]), dtype=torch.long),
            torch.tensor(float(row["intensity_target"]), dtype=torch.float32),
        )


def make_loss():
    # Class weights from the TRAIN split only.
    counts = np.bincount(
        TRAIN_DF["category_id"].to_numpy(),
        minlength=len(CATEGORIES),
    ).astype(np.float32)

    weights = np.zeros_like(counts)

    nonzero = counts > 0
    weights[nonzero] = len(TRAIN_DF) / (
        len(CATEGORIES) * counts[nonzero]
    )

    # Categories absent from train get zero weight.
    # They cannot be learned from a training split that does not contain them.
    weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    return nn.CrossEntropyLoss(weight=weights)


def run_epoch(model, loader, optimizer, category_loss_fn):
    model.train()

    total_loss = 0.0
    total_cat = 0.0
    total_int = 0.0
    n = 0

    for images, categories, intensity in loader:
        images = images.to(DEVICE)
        categories = categories.to(DEVICE)
        intensity = intensity.to(DEVICE)

        optimizer.zero_grad()

        _, category_logits, intensity_pred = model(images)

        cat_loss = category_loss_fn(
            category_logits,
            categories,
        )

        int_loss = nn.functional.smooth_l1_loss(
            intensity_pred,
            intensity,
        )

        loss = cat_loss + INTENSITY_WEIGHT * int_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_cat += cat_loss.item() * bs
        total_int += int_loss.item() * bs
        n += bs

    return (
        total_loss / max(n, 1),
        total_cat / max(n, 1),
        total_int / max(n, 1),
    )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    total_cat = 0.0
    total_int = 0.0
    correct = 0
    n = 0

    all_pred = []
    all_true = []
    all_int_pred = []
    all_int_true = []

    for images, categories, intensity in loader:
        images = images.to(DEVICE)
        categories = categories.to(DEVICE)
        intensity = intensity.to(DEVICE)

        _, category_logits, intensity_pred = model(images)

        cat_loss = nn.functional.cross_entropy(
            category_logits,
            categories,
        )

        int_loss = nn.functional.smooth_l1_loss(
            intensity_pred,
            intensity,
        )

        loss = cat_loss + INTENSITY_WEIGHT * int_loss

        pred = category_logits.argmax(dim=1)

        bs = images.size(0)

        total_loss += loss.item() * bs
        total_cat += cat_loss.item() * bs
        total_int += int_loss.item() * bs

        correct += (pred == categories).sum().item()
        n += bs

        all_pred.append(pred.cpu())
        all_true.append(categories.cpu())
        all_int_pred.append(intensity_pred.cpu())
        all_int_true.append(intensity.cpu())

    pred = torch.cat(all_pred)
    true = torch.cat(all_true)

    int_pred = torch.cat(all_int_pred)
    int_true = torch.cat(all_int_true)

    # Convert normalized MAE back to knots.
    wind_mae = (
        torch.abs(int_pred - int_true).mean().item()
        * (WIND_MAX - WIND_MIN)
    )

    return {
        "loss": total_loss / max(n, 1),
        "category_loss": total_cat / max(n, 1),
        "intensity_loss": total_int / max(n, 1),
        "category_accuracy": correct / max(n, 1),
        "wind_mae_kt": wind_mae,
        "pred": pred,
        "true": true,
    }


def save_checkpoint(model, mean, std, epoch, metrics, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "num_categories": len(CATEGORIES),
            "categories": CATEGORIES,
            "in_channels": 2,
            "normalization_mean": mean,
            "normalization_std": std,
            "wind_min_kt": WIND_MIN,
            "wind_max_kt": WIND_MAX,
            "metrics": metrics,
        },
        path,
    )


def main():
    set_seed()

    global TRAIN_DF, DEVICE

    DEVICE = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    df = add_paths_and_labels(load_manifest())

    train_df = df[df["cyclone"].isin(TRAIN_STORMS)].copy()
    val_df = df[df["cyclone"].isin(VAL_STORMS)].copy()
    test_df = df[df["cyclone"].isin(TEST_STORMS)].copy()

    TRAIN_DF = train_df

    print("\n" + "=" * 70)
    print("STORM-LEVEL SPLIT")
    print("=" * 70)
    print(f"Train: {len(train_df)} -> {TRAIN_STORMS}")
    print(f"Val  : {len(val_df)} -> {VAL_STORMS}")
    print(f"Test : {len(test_df)} -> {TEST_STORMS}")

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Train/val/test split is empty.")

    # IMPORTANT: statistics come from TRAIN ONLY.
    mean, std = calculate_train_stats(train_df)

    print("\nTRAIN-ONLY NORMALIZATION")
    print(f"TIR1 mean={mean[0]:.8f}, std={std[0]:.8f}")
    print(f"WV   mean={mean[1]:.8f}, std={std[1]:.8f}")

    train_ds = INSATDataset(train_df, mean, std)
    val_ds = INSATDataset(val_df, mean, std)
    test_ds = INSATDataset(test_df, mean, std)

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

    model = CycloneCNN(
        num_categories=len(CATEGORIES),
        in_channels=2,
    ).to(DEVICE)

    category_loss_fn = make_loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_path = ckpt_dir / "classifier_insat.pt"
    stats_path = (
        ROOT
        / "data"
        / "processed"
        / "insat_training_stats.npz"
    )

    np.savez(
        stats_path,
        mean=mean,
        std=std,
        wind_min=WIND_MIN,
        wind_max=WIND_MAX,
    )

    best_val = float("inf")

    print("\n" + "=" * 70)
    print("TRAINING REAL INSAT CLASSIFIER")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_cat, train_int = run_epoch(
            model,
            train_loader,
            optimizer,
            category_loss_fn,
        )

        val = evaluate(model, val_loader)
        scheduler.step(val["loss"])

        marker = ""

        if val["loss"] < best_val:
            best_val = val["loss"]

            save_checkpoint(
                model,
                mean,
                std,
                epoch,
                val,
                best_path,
            )

            marker = "  <-- BEST"

        print(
            f"epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_loss:.4f} | "
            f"cat={train_cat:.4f} | "
            f"int={train_int:.4f} | "
            f"val={val['loss']:.4f} | "
            f"val_acc={val['category_accuracy']:.3f} | "
            f"val_wind_mae={val['wind_mae_kt']:.2f}kt"
            f"{marker}"
        )

    print("\n" + "=" * 70)
    print("FINAL TEST")
    print("=" * 70)

    checkpoint = torch.load(
        best_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test = evaluate(model, test_loader)

    print(f"Test loss       : {test['loss']:.4f}")
    print(f"Test category   : {test['category_accuracy']:.3f}")
    print(f"Test wind MAE   : {test['wind_mae_kt']:.2f} kt")

    print("\nTest predictions:")
    for p, t in zip(
        test["pred"].tolist(),
        test["true"].tolist(),
    ):
        print(
            f"  predicted={CATEGORIES[p]:35s} "
            f"actual={CATEGORIES[t]}"
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Checkpoint : {best_path}")
    print(f"Stats      : {stats_path}")

    print(
        "\nNOTE: presence/cyclone-vs-no-cyclone is NOT a valid "
        "metric yet because this dataset contains only positive "
        "cyclone samples. Add negative INSAT samples before using "
        "the presence head."
    )


if __name__ == "__main__":
    main()
