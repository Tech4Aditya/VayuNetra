"""
train.py

Training pipeline for:

    Task 1:
        Cyclone identification
        cyclone vs no-cyclone

    Task 2:
        Cyclone intensity classification
        6-category classification

    Task 2B:
        Continuous intensity regression

    Task 3:
        Short-horizon trend + track prediction

Current dataset:
    backend/data/synthetic_dataset.npz

Run:

    python backend/data/generate_synthetic.py \
        --n_sequences 2000 \
        --seq_len 8 \
        --negative_ratio 0.30

    python backend/train.py

IMPORTANT:
    Synthetic data is for pipeline validation only.
    It is NOT evidence of real-world cyclone performance.
"""

import os
import random
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, random_split

from models.classifier import CycloneCNN, loss_fn as classifier_loss
from models.predictor import CycloneTrendLSTM, loss_fn as predictor_loss


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "synthetic_dataset.npz"
)

CKPT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints"
)

os.makedirs(CKPT_DIR, exist_ok=True)


SEED = 42

VAL_RATIO = 0.15

CLASSIFIER_EPOCHS = 30
PREDICTOR_EPOCHS = 30

BATCH_SIZE = 32

CLASSIFIER_LR = 1e-3
PREDICTOR_LR = 1e-3

NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is more important than tiny speed gains
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET — CLASSIFIER
# ============================================================

class ClassifierDataset(Dataset):
    """
    One final frame from each sequence.

    Returns:

        frame
            (1, H, W)

        presence
            0 = no cyclone
            1 = cyclone

        category
            0...5 for cyclone
            -1 internally for negative samples,
            but converted to 0 because CrossEntropy cannot
            receive -1 unless explicitly ignored.

        has_category
            1 = category target is valid
            0 = category target should be ignored

        intensity
            final normalized intensity [0, 1]

        has_intensity
            1 = valid cyclone intensity
            0 = negative sample
    """

    def __init__(self, data):

        self.frames = data["frames"][:, -1]

        self.category_labels = data["category_labels"]

        self.presence_labels = data["presence_labels"]

        # Final frame intensity
        self.intensity_labels = data["intensities"][:, -1]

    def __len__(self):

        return len(self.frames)

    def __getitem__(self, idx):

        frame = torch.from_numpy(
            self.frames[idx]
        ).unsqueeze(0).float()

        presence = torch.tensor(
            float(self.presence_labels[idx]),
            dtype=torch.float32
        )

        raw_category = int(
            self.category_labels[idx]
        )

        if raw_category >= 0:

            category = torch.tensor(
                raw_category,
                dtype=torch.long
            )

            has_category = torch.tensor(
                1.0,
                dtype=torch.float32
            )

        else:

            # Dummy value.
            # It will be masked during loss calculation.
            category = torch.tensor(
                0,
                dtype=torch.long
            )

            has_category = torch.tensor(
                0.0,
                dtype=torch.float32
            )

        intensity = torch.tensor(
            float(self.intensity_labels[idx]),
            dtype=torch.float32
        )

        has_intensity = torch.tensor(
            float(self.presence_labels[idx]),
            dtype=torch.float32
        )

        return (
            frame,
            presence,
            category,
            has_category,
            intensity,
            has_intensity,
        )


# ============================================================
# DATASET — TEMPORAL PREDICTOR
# ============================================================

class SequenceDataset(Dataset):
    """
    Full cyclone sequence.

    Only cyclone-present sequences are used.

    Returns:

        seq
            (T, 1, H, W)

        trend
            0 = weakening
            1 = steady
            2 = intensifying

        track_delta
            (dx, dy)
    """

    def __init__(self, data):

        positive_mask = (
            data["presence_labels"] == 1
        )

        self.frames = data["frames"][positive_mask]

        self.positions = data["positions"][positive_mask]

        self.trend_labels = data["trend_labels"][positive_mask]

    def __len__(self):

        return len(self.frames)

    def __getitem__(self, idx):

        seq = torch.from_numpy(
            self.frames[idx]
        ).unsqueeze(1).float()

        # Original generator:
        #
        # -1 = weakening
        #  0 = steady
        # +1 = intensifying
        #
        # Convert:
        #
        # -1 -> 0
        #  0 -> 1
        # +1 -> 2

        trend = torch.tensor(
            int(self.trend_labels[idx]) + 1,
            dtype=torch.long
        )

        track_delta = torch.from_numpy(
            self.positions[idx][-1]
            - self.positions[idx][-2]
        ).float()

        return (
            seq,
            trend,
            track_delta,
        )


# ============================================================
# DATA SPLITTING
# ============================================================

def make_split(dataset, val_ratio=VAL_RATIO):

    n_total = len(dataset)

    n_val = max(
        1,
        int(n_total * val_ratio)
    )

    n_train = n_total - n_val

    generator = torch.Generator()

    generator.manual_seed(SEED)

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=generator
    )

    return train_ds, val_ds


# ============================================================
# CLASSIFIER METRICS
# ============================================================

def binary_metrics(pred, target):

    pred = pred.int()
    target = target.int()

    tp = ((pred == 1) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()

    accuracy = (
        (tp + tn) /
        max(tp + tn + fp + fn, 1)
    )

    precision = (
        tp /
        max(tp + fp, 1)
    )

    recall = (
        tp /
        max(tp + fn, 1)
    )

    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-8)
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# CLASSIFIER TRAINING
# ============================================================

def train_classifier(
    data,
    epochs=CLASSIFIER_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=CLASSIFIER_LR,
    device="cpu",
):

    print("\n" + "=" * 70)
    print("TRAINING CYCLONE CLASSIFIER")
    print("=" * 70)

    ds = ClassifierDataset(data)

    train_ds, val_ds = make_split(ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = CycloneCNN().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    best_val_loss = float("inf")

    best_path = os.path.join(
        CKPT_DIR,
        "classifier.pt"
    )

    for epoch in range(epochs):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        total_loss = 0.0

        for (
            frame,
            presence,
            category,
            has_category,
            intensity,
            has_intensity,
        ) in train_loader:

            frame = frame.to(device)
            presence = presence.to(device)
            category = category.to(device)
            has_category = has_category.to(device)
            intensity = intensity.to(device)
            has_intensity = has_intensity.to(device)

            optimizer.zero_grad()

            (
                presence_logit,
                category_logits,
                intensity_pred,
            ) = model(frame)

            loss = classifier_loss(
                presence_logit,
                category_logits,
                intensity_pred,
                presence,
                category,
                intensity,
            )

            loss.backward()

            # Prevent unstable gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            total_loss += (
                loss.item() * frame.size(0)
            )

        train_loss = (
            total_loss /
            max(len(train_ds), 1)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss_total = 0.0

        presence_preds = []
        presence_targets = []

        category_preds = []
        category_targets = []

        intensity_preds = []
        intensity_targets = []

        with torch.no_grad():

            for (
                frame,
                presence,
                category,
                has_category,
                intensity,
                has_intensity,
            ) in val_loader:

                frame = frame.to(device)
                presence = presence.to(device)
                category = category.to(device)
                has_category = has_category.to(device)
                intensity = intensity.to(device)
                has_intensity = has_intensity.to(device)

                (
                    presence_logit,
                    category_logits,
                    intensity_pred,
                ) = model(frame)

                loss = classifier_loss(
                    presence_logit,
                    category_logits,
                    intensity_pred,
                    presence,
                    category,
                    intensity,
                )

                val_loss_total += (
                    loss.item() * frame.size(0)
                )

                # --------------------------------------------
                # Presence
                # --------------------------------------------

                presence_pred = (
                    torch.sigmoid(
                        presence_logit
                    ) >= 0.5
                ).long()

                presence_preds.append(
                    presence_pred.cpu()
                )

                presence_targets.append(
                    presence.long().cpu()
                )

                # --------------------------------------------
                # Category
                # --------------------------------------------

                category_mask = (
                    has_category.bool()
                )

                if category_mask.any():

                    pred_cat = (
                        category_logits
                        .argmax(dim=1)
                    )

                    category_preds.append(
                        pred_cat[
                            category_mask
                        ].cpu()
                    )

                    category_targets.append(
                        category[
                            category_mask
                        ].cpu()
                    )

                # --------------------------------------------
                # Intensity
                # --------------------------------------------

                intensity_mask = (
                    has_intensity.bool()
                )

                if intensity_mask.any():

                    intensity_preds.append(
                        intensity_pred[
                            intensity_mask
                        ].cpu()
                    )

                    intensity_targets.append(
                        intensity[
                            intensity_mask
                        ].cpu()
                    )

        val_loss = (
            val_loss_total /
            max(len(val_ds), 1)
        )

        scheduler.step(val_loss)

        # ====================================================
        # METRICS
        # ====================================================

        presence_preds = torch.cat(
            presence_preds
        )

        presence_targets = torch.cat(
            presence_targets
        )

        presence_stats = binary_metrics(
            presence_preds,
            presence_targets,
        )

        if category_preds:

            category_preds = torch.cat(
                category_preds
            )

            category_targets = torch.cat(
                category_targets
            )

            category_accuracy = (
                category_preds ==
                category_targets
            ).float().mean().item()

        else:

            category_accuracy = 0.0

        if intensity_preds:

            intensity_preds = torch.cat(
                intensity_preds
            )

            intensity_targets = torch.cat(
                intensity_targets
            )

            intensity_mae = torch.mean(
                torch.abs(
                    intensity_preds -
                    intensity_targets
                )
            ).item()

        else:

            intensity_mae = 0.0

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "presence_f1": presence_stats["f1"],
                    "category_accuracy": category_accuracy,
                    "intensity_mae": intensity_mae,
                },
                best_path,
            )

            marker = "  ← BEST"

        else:

            marker = ""

        print(
            f"[classifier] "
            f"epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"presence_acc={presence_stats['accuracy']:.3f} | "
            f"presence_f1={presence_stats['f1']:.3f} | "
            f"category_acc={category_accuracy:.3f} | "
            f"intensity_mae={intensity_mae:.4f}"
            f"{marker}"
        )

    print(
        f"\nBest classifier checkpoint saved to:\n"
        f"{best_path}"
    )

    return model


# ============================================================
# PREDICTOR TRAINING
# ============================================================

def train_predictor(
    data,
    epochs=PREDICTOR_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=PREDICTOR_LR,
    device="cpu",
):

    print("\n" + "=" * 70)
    print("TRAINING TEMPORAL PREDICTOR")
    print("=" * 70)

    ds = SequenceDataset(data)

    train_ds, val_ds = make_split(ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = CycloneTrendLSTM().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    best_val_loss = float("inf")

    best_path = os.path.join(
        CKPT_DIR,
        "predictor.pt"
    )

    for epoch in range(epochs):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        total_loss = 0.0

        for seq, trend, track_delta in train_loader:

            seq = seq.to(device)
            trend = trend.to(device)
            track_delta = track_delta.to(device)

            optimizer.zero_grad()

            trend_logits, pred_delta = model(seq)

            loss = predictor_loss(
                trend_logits,
                pred_delta,
                trend,
                track_delta,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            total_loss += (
                loss.item() *
                seq.size(0)
            )

        train_loss = (
            total_loss /
            max(len(train_ds), 1)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss_total = 0.0

        correct = 0
        total = 0

        with torch.no_grad():

            for seq, trend, track_delta in val_loader:

                seq = seq.to(device)
                trend = trend.to(device)
                track_delta = track_delta.to(device)

                trend_logits, pred_delta = model(seq)

                loss = predictor_loss(
                    trend_logits,
                    pred_delta,
                    trend,
                    track_delta,
                )

                val_loss_total += (
                    loss.item() *
                    seq.size(0)
                )

                pred = (
                    trend_logits
                    .argmax(dim=1)
                )

                correct += (
                    pred == trend
                ).sum().item()

                total += trend.size(0)

        val_loss = (
            val_loss_total /
            max(len(val_ds), 1)
        )

        trend_accuracy = (
            correct /
            max(total, 1)
        )

        scheduler.step(val_loss)

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "trend_accuracy": trend_accuracy,
                },
                best_path,
            )

            marker = "  ← BEST"

        else:

            marker = ""

        print(
            f"[predictor] "
            f"epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"trend_acc={trend_accuracy:.3f}"
            f"{marker}"
        )

    print(
        f"\nBest predictor checkpoint saved to:\n"
        f"{best_path}"
    )

    return model


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_dataset_summary(data):

    presence = data["presence_labels"]

    categories = data["category_labels"]

    trends = data["trend_labels"]

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Sequences       : {len(presence)}"
    )

    print(
        f"Frame shape     : {data['frames'].shape}"
    )

    n_positive = int(
        (presence == 1).sum()
    )

    n_negative = int(
        (presence == 0).sum()
    )

    print(
        f"Cyclone         : {n_positive}"
    )

    print(
        f"No cyclone      : {n_negative}"
    )

    print(
        f"Negative ratio  : "
        f"{n_negative / max(len(presence), 1):.2%}"
    )

    print("\nCategory distribution:")

    for category_id in range(6):

        count = int(
            (categories == category_id).sum()
        )

        print(
            f"  {category_id}: {count}"
        )

    print("\nTrend distribution:")

    trend_names = {
        -1: "Weakening",
         0: "Steady",
         1: "Intensifying",
    }

    for trend_id, name in trend_names.items():

        count = int(
            (trends == trend_id).sum()
        )

        print(
            f"  {name:<12}: {count}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    set_seed()

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{DATA_PATH}\n\n"
            f"Generate it first:\n"
            f"python backend/data/generate_synthetic.py "
            f"--n_sequences 2000 "
            f"--seq_len 8 "
            f"--negative_ratio 0.30\n"
        )

    raw = np.load(DATA_PATH)

    data = {
        key: raw[key]
        for key in raw.files
    }

    print_dataset_summary(data)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nTraining device: {device}"
    )

    if device == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # TRAIN CLASSIFIER
    # --------------------------------------------------------

    train_classifier(
        data,
        device=device,
    )

    # --------------------------------------------------------
    # TRAIN TEMPORAL PREDICTOR
    # --------------------------------------------------------

    train_predictor(
        data,
        device=device,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Classifier: "
        f"{os.path.join(CKPT_DIR, 'classifier.pt')}"
    )

    print(
        f"Predictor : "
        f"{os.path.join(CKPT_DIR, 'predictor.pt')}"
    )