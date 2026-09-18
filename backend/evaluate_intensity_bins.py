from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PREDICTIONS = Path(
    "backend/data/predictions/tcir_temporal_test_predictions.csv"
)

OUTPUT_DIR = Path("backend/data/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

df = pd.read_csv(PREDICTIONS)

print("=" * 70)
print("TCIR TEMPORAL MODEL — INTENSITY BIN AUDIT")
print("=" * 70)

print(f"\nRows: {len(df)}")
print(f"Columns: {list(df.columns)}")


# ---------------------------------------------------------
# FIND COLUMNS
# ---------------------------------------------------------

# Expected names from predict_tcir_temporal.py
actual_col = "true_wind_kt"
pred_col = "pred_wind_kt"

if actual_col not in df.columns or pred_col not in df.columns:
    raise ValueError(
        f"Expected columns '{actual_col}' and '{pred_col}' "
        f"were not found."
    )


actual = df[actual_col].astype(float)
predicted = df[pred_col].astype(float)

df["error_kt"] = predicted - actual
df["abs_error_kt"] = np.abs(df["error_kt"])
df["squared_error"] = df["error_kt"] ** 2


# ---------------------------------------------------------
# INTENSITY BINS
# ---------------------------------------------------------
#
# These are deliberately based on actual wind.
#
# < 34       : Depression / weaker systems
# 34-47      : Tropical/Cyclonic Storm range
# 48-63      : stronger storm range
# 64-95      : hurricane/major-cyclone equivalent range
# >= 96      : extreme intensity
#
# We are NOT calling these official IMD categories here.
# This is an evaluation grouping.
# ---------------------------------------------------------

bins = [-np.inf, 33, 47, 63, 95, np.inf]

labels = [
    "<34 kt",
    "34–47 kt",
    "48–63 kt",
    "64–95 kt",
    ">=96 kt",
]

df["intensity_bin"] = pd.cut(
    actual,
    bins=bins,
    labels=labels,
    right=True,
)


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

rows = []

for label in labels:

    subset = df[df["intensity_bin"] == label]

    if len(subset) == 0:
        continue

    y_true = subset[actual_col].values
    y_pred = subset[pred_col].values

    error = y_pred - y_true

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    bias = np.mean(error)

    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot > 0:
        r2 = 1 - ss_res / ss_tot
    else:
        r2 = np.nan

    rows.append({
        "intensity_bin": label,
        "n": len(subset),
        "actual_mean_kt": np.mean(y_true),
        "predicted_mean_kt": np.mean(y_pred),
        "mae_kt": mae,
        "rmse_kt": rmse,
        "bias_kt": bias,
        "min_actual_kt": np.min(y_true),
        "max_actual_kt": np.max(y_true),
        "r2": r2,
    })


metrics = pd.DataFrame(rows)

metrics.to_csv(
    OUTPUT_DIR / "intensity_bin_metrics.csv",
    index=False
)


# ---------------------------------------------------------
# PRINT TABLE
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PER-INTENSITY PERFORMANCE")
print("=" * 70)

print(
    metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ---------------------------------------------------------
# UNDERPREDICTION ANALYSIS
# ---------------------------------------------------------

df["underprediction"] = df[pred_col] < df[actual_col]

under_rate = (
    df["underprediction"].mean() * 100
)

print("\n" + "=" * 70)
print("UNDERPREDICTION ANALYSIS")
print("=" * 70)

print(
    f"Overall underprediction rate: "
    f"{under_rate:.2f}%"
)

for label in labels:

    subset = df[df["intensity_bin"] == label]

    if len(subset) == 0:
        continue

    rate = subset["underprediction"].mean() * 100

    print(
        f"{label:>10}: "
        f"{rate:6.2f}% underpredicted"
    )


# ---------------------------------------------------------
# EXTREME STORM AUDIT
# ---------------------------------------------------------

extreme = df[df[actual_col] >= 96].copy()

print("\n" + "=" * 70)
print("EXTREME INTENSITY AUDIT (>=96 kt)")
print("=" * 70)

print(f"Extreme samples: {len(extreme)}")

if len(extreme):

    extreme["abs_error_kt"] = np.abs(
        extreme[pred_col] - extreme[actual_col]
    )

    extreme = extreme.sort_values(
        "abs_error_kt",
        ascending=False
    )

    print(
        extreme[
            [
                "cyclone_id",
                "timestamp",
                actual_col,
                pred_col,
                "error_kt",
                "abs_error_kt",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    extreme.to_csv(
        OUTPUT_DIR / "extreme_intensity_predictions.csv",
        index=False
    )


# ---------------------------------------------------------
# PLOT 1 — ACTUAL VS PREDICTED
# ---------------------------------------------------------

plt.figure(figsize=(8, 7))

plt.scatter(
    actual,
    predicted,
    alpha=0.45
)

minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Wind (kt)")
plt.ylabel("Predicted Wind (kt)")
plt.title("TCIR Temporal Model — Actual vs Predicted Wind")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted_wind.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# PLOT 2 — ERROR VS ACTUAL
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    actual,
    df["error_kt"],
    alpha=0.45
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Actual Wind (kt)")
plt.ylabel("Prediction Error (kt)")
plt.title("Wind Prediction Error vs Actual Intensity")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_vs_actual_wind.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# PLOT 3 — ERROR DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    df["error_kt"],
    bins=30
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Prediction Error (Predicted − Actual) [kt]")
plt.ylabel("Number of Samples")
plt.title("Wind Prediction Error Distribution")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "wind_error_distribution.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

summary = pd.DataFrame([{
    "samples": len(df),
    "actual_mean_kt": actual.mean(),
    "predicted_mean_kt": predicted.mean(),
    "mae_kt": np.mean(np.abs(predicted - actual)),
    "rmse_kt": np.sqrt(np.mean((predicted - actual) ** 2)),
    "bias_kt": np.mean(predicted - actual),
    "underprediction_rate_pct": under_rate,
    "extreme_samples_ge_96kt": len(extreme),
}])

summary.to_csv(
    OUTPUT_DIR / "intensity_bin_summary.csv",
    index=False
)


print("\n" + "=" * 70)
print("FILES GENERATED")
print("=" * 70)

print("✓ intensity_bin_metrics.csv")
print("✓ intensity_bin_summary.csv")
print("✓ actual_vs_predicted_wind.png")
print("✓ error_vs_actual_wind.png")
print("✓ wind_error_distribution.png")

print("\nAudit complete.")