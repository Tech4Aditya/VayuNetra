"""
generate_synthetic.py

Produces synthetic INSAT-style IR brightness-temperature frames and matching
cyclone track/intensity sequences, so the pipeline is runnable end-to-end
without waiting on MOSDAC/IMD data access approval.

Swap this out for real data later:
  - Imagery:  MOSDAC (ISRO) INSAT-3D/3DR IR + visible bands
  - Labels:   IMD best-track cyclone reports (T-number, category, lat/lon)

Usage:
    python generate_synthetic.py --n_sequences 500 --seq_len 8
"""
import argparse
import numpy as np
import os

CATEGORIES = [
    "Depression", "Deep Depression", "Cyclonic Storm",
    "Severe Cyclonic Storm", "Very Severe Cyclonic Storm", "Super Cyclonic Storm"
]

def make_cyclone_frame(size, center, radius, intensity, noise_std=0.05):
    """Fake IR brightness-temperature frame: cold spiral core + warm background."""
    y, x = np.mgrid[0:size, 0:size]
    cy, cx = center
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    core = np.exp(-(dist ** 2) / (2 * (radius ** 2))) * intensity
    spiral = 0.15 * intensity * np.sin(dist / 4.0 - np.arctan2(y - cy, x - cx) * 2)
    frame = 0.5 + core + spiral
    frame += np.random.normal(0, noise_std, size=frame.shape)
    return np.clip(frame, 0, 1).astype(np.float32)

def make_sequence(size=64, seq_len=8, rng=None):
    rng = rng or np.random.default_rng()
    cat_idx = rng.integers(0, len(CATEGORIES))
    base_intensity = 0.3 + 0.7 * (cat_idx / (len(CATEGORIES) - 1))
    pos = np.array([rng.uniform(size * 0.2, size * 0.8) for _ in range(2)])
    vel = rng.normal(0, 1.2, size=2)
    intensity_trend = rng.choice([-1, 0, 1])  # weakening / steady / intensifying

    frames, intensities, positions = [], [], []
    intensity = base_intensity
    for t in range(seq_len):
        radius = 6 + 10 * intensity
        frame = make_cyclone_frame(size, pos, radius, intensity)
        frames.append(frame)
        intensities.append(intensity)
        positions.append(pos.copy())
        pos = pos + vel + rng.normal(0, 0.5, size=2)
        pos = np.clip(pos, 5, size - 5)
        intensity = np.clip(intensity + intensity_trend * 0.02 + rng.normal(0, 0.01), 0.05, 1.0)

    label_cat = int(np.clip(round(intensities[-1] * (len(CATEGORIES) - 1)), 0, len(CATEGORIES) - 1))
    return {
        "frames": np.stack(frames),
        "positions": np.stack(positions),
        "intensities": np.array(intensities),
        "category_label": label_cat,
        "trend_label": intensity_trend,
    }

def make_clear_sky_frame(size, noise_std=0.08, rng=None):
    """Fake 'nothing here' frame: warm background, occasional non-spiral cloud
    clumps, no cold coherent core. This is the negative class the identification
    head needs — without it, the model has never seen a non-cyclone example."""
    rng = rng or np.random.default_rng()
    frame = np.full((size, size), 0.25, dtype=np.float32)  # warm baseline

    n_patches = rng.integers(0, 4)
    y, x = np.mgrid[0:size, 0:size]
    for _ in range(n_patches):
        cy, cx = rng.uniform(0, size, size=2)
        r = rng.uniform(3, 10)
        strength = rng.uniform(0.05, 0.25)
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        frame += (strength * np.exp(-(dist ** 2) / (2 * r ** 2))).astype(np.float32)

    frame += rng.normal(0, noise_std, size=frame.shape).astype(np.float32)
    return np.clip(frame, 0, 1).astype(np.float32)


def make_negative_sequence(size=64, seq_len=8, rng=None):
    """A full sequence with no cyclone present, for identification negatives."""
    rng = rng or np.random.default_rng()
    frames = [make_clear_sky_frame(size, rng=rng) for _ in range(seq_len)]
    return {
        "frames": np.stack(frames),
        "positions": np.zeros((seq_len, 2), dtype=np.float32),
        "intensities": np.zeros(seq_len, dtype=np.float32),
        "category_label": -1,   # invalid; excluded from classifier loss
        "trend_label": 0,       # unused; excluded from predictor loss
        "presence_label": 0,
    }


def build_dataset(n_sequences=500, size=64, seq_len=8, seed=42, negative_ratio=0.3):
    """negative_ratio: fraction of the dataset with NO cyclone present."""
    rng = np.random.default_rng(seed)
    n_negative = int(n_sequences * negative_ratio)
    n_positive = n_sequences - n_negative

    frames, positions, intensities, cat_labels, trend_labels, presence_labels = [], [], [], [], [], []

    for _ in range(n_positive):
        s = make_sequence(size, seq_len, rng)
        frames.append(s["frames"]); positions.append(s["positions"])
        intensities.append(s["intensities"]); cat_labels.append(s["category_label"])
        trend_labels.append(s["trend_label"]); presence_labels.append(1)

    for _ in range(n_negative):
        s = make_negative_sequence(size, seq_len, rng)
        frames.append(s["frames"]); positions.append(s["positions"])
        intensities.append(s["intensities"]); cat_labels.append(s["category_label"])
        trend_labels.append(s["trend_label"]); presence_labels.append(0)

    order = rng.permutation(n_sequences)
    frames = np.stack(frames)[order]
    positions = np.stack(positions)[order]
    intensities = np.stack(intensities)[order]
    cat_labels = np.array(cat_labels)[order]
    trend_labels = np.array(trend_labels)[order]
    presence_labels = np.array(presence_labels)[order]

    return {
        "frames": frames, "positions": positions, "intensities": intensities,
        "category_labels": cat_labels, "trend_labels": trend_labels,
        "presence_labels": presence_labels,
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sequences", type=int, default=500)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--negative_ratio", type=float, default=0.3,
                     help="Fraction of sequences with NO cyclone present")
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "synthetic_dataset.npz"))
    args = ap.parse_args()

    ds = build_dataset(args.n_sequences, args.size, args.seq_len, negative_ratio=args.negative_ratio)
    np.savez_compressed(args.out, **ds)
    n_neg = int((ds["presence_labels"] == 0).sum())
    print(f"Saved {args.n_sequences} sequences to {args.out} ({n_neg} negative / no-cyclone)")
    print(f"Frame shape: {ds['frames'].shape}, categories: {CATEGORIES}")