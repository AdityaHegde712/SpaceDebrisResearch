"""
T3a.4 — TCN Baseline (Per Size Range) — Memory-Efficient Positional Lookup
===========================================================================
Temporal Convolutional Network for binary collision risk classification
on 20-epoch sliding windows.
"""

import polars as pl
import numpy as np
import time
import gc
import pickle
from pathlib import Path

import torch

from sklearn.preprocessing import LabelEncoder

from src.constants import CAT_COLS, SEQ_LEN
from src.io_utils import (
    get_data_dir,
    get_models_dir,
    get_reports_dir,
    load_pickle,
    save_json,
    timer,
)
from src.feature_utils import (
    get_feature_cols,
    encode_categoricals,
    scale_numeric_features,
    SIZE_CONFIGS,
)
from src.window_utils import extract_windows_positional
from src.models.tcn import TCNClassifier, train_tcn, evaluate_tcn

DATA_DIR = get_data_dir()
MODELS_DIR = get_models_dir()
REPORTS_DIR = get_reports_dir()

# Training config
BATCH_SIZE = 128
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-3
N_FILTERS = 64
N_LAYERS = 4
KERNEL_SIZE = 3


def process_size_range(size_label, feat_name, target_name, sample_frac):
    """Process one size range end-to-end."""
    t_size = timer()
    print(f"\n{'=' * 60}")
    print(f"  Size: {size_label}  (sample={sample_frac})")
    print(f"{'=' * 60}")

    # 1. Determine feature columns
    print("  [1] Determining feature columns...")
    schema = pl.scan_parquet(DATA_DIR / feat_name).head(1).collect()
    feature_cols = get_feature_cols(available_cols=set(schema.columns))
    num_cat = [c for c in feature_cols if c in CAT_COLS]
    print(
        f"    Total features: {len(feature_cols)} "
        f"({len(feature_cols) - len(num_cat)} num + {len(num_cat)} cat)"
    )

    # 2. Load targets, optionally sample
    print("  [2] Loading targets...")
    targets = pl.read_parquet(DATA_DIR / target_name)
    if sample_frac < 1.0:
        targets = targets.group_by("split").map_groups(
            lambda g: g.sample(fraction=sample_frac, seed=42)
        )
        gc.collect()
    n_windows = len(targets)
    print(
        f"    {n_windows:,} windows "
        f"(high-risk: {targets['collision_risk'].mean() * 100:.1f}%)"
    )

    # 3. Load features
    print("  [3] Loading features...")
    load_cols = ["satellite_number", "epoch_dt"] + feature_cols
    features = pl.scan_parquet(DATA_DIR / feat_name).select(load_cols).collect()
    print(f"    {len(features):,} rows, {len(features.columns)} columns")

    # 4. Encode categoricals and scale numerics
    print("  [4] Encoding & scaling...")
    features, label_encoders = encode_categoricals(features, num_cat)
    scaler = load_pickle(MODELS_DIR / "scaler.pkl")
    features = scale_numeric_features(features, feature_cols, scaler)
    del scaler
    gc.collect()
    print(f"    Encoded {len(num_cat)} categoricals, scaled numerics")

    # 5. Extract windows via positional lookup
    print("  [5] Extracting windows (positional lookup)...")
    X_train, y_train, X_val, y_val, X_test, y_test = extract_windows_positional(
        features, targets, feature_cols
    )

    del targets
    gc.collect()

    print(f"\n    Train: {len(X_train):,} (high-risk: {y_train.mean() * 100:.1f}%)")
    print(f"    Val:   {len(X_val):,} (high-risk: {y_val.mean() * 100:.1f}%)")
    print(f"    Test:  {len(X_test):,} (high-risk: {y_test.mean() * 100:.1f}%)")

    # 6. Train
    print("\n  [6] Training TCN...")
    model, history = train_tcn(
        X_train,
        y_train,
        X_val,
        y_val,
        len(feature_cols),
        size_label,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        lr=LR,
        n_filters=N_FILTERS,
        n_layers=N_LAYERS,
        kernel_size=KERNEL_SIZE,
    )

    # 7. Evaluate
    print("\n  [7] Evaluating...")
    metrics = evaluate_tcn(model, X_test, y_test, batch_size=BATCH_SIZE)
    metrics.update(
        {
            "size_range": size_label,
            "sample_fraction": sample_frac,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "n_features": int(len(feature_cols)),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "n_windows_raw": int(n_windows),
            "train_high_risk_ratio": float(y_train.mean()),
            "val_high_risk_ratio": float(y_val.mean()),
            "training_time_seconds": round(t_size.elapsed, 1),
            "total_time_seconds": round(t_size.elapsed, 1),
        }
    )

    print(f"\n    ROC-AUC: {metrics['roc_auc']:.4f}  PR-AUC: {metrics['pr_auc']:.4f}")
    print(
        f"    F1:      {metrics['f1_score']:.4f}  Brier:   {metrics['brier_score']:.4f}"
    )
    print(f"    Acc:     {metrics['accuracy']:.4f}")
    print(f"    CM:      {metrics['confusion_matrix']}")

    # 8. Save model
    model_path = MODELS_DIR / f"tcn_{size_label}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_cols": feature_cols,
            "n_features": len(feature_cols),
            "model_config": {
                "n_filters": N_FILTERS,
                "n_layers": N_LAYERS,
                "kernel_size": KERNEL_SIZE,
            },
            "metrics": metrics,
            "history": history,
        },
        model_path,
    )
    print(f"\n  Model saved: {model_path}")

    # 9. Save metrics
    metrics_path = REPORTS_DIR / f"tcn_{size_label}_metrics.json"
    save_json(metrics, metrics_path)
    print(f"  Metrics saved: {metrics_path}")

    del X_train, y_train, X_val, y_val, X_test, y_test, model
    gc.collect()

    elapsed = t_size.elapsed
    print(f"\n  Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return metrics


def main():
    print("=" * 60)
    print("  T3a.4 — TCN Baseline (Positional Lookup)")
    print("  Dilated Conv1d | seq_len=20 | chunked window extraction")
    print("=" * 60)

    t_start = time.time()
    all_metrics = {}

    for cfg in SIZE_CONFIGS:
        metrics = process_size_range(*cfg)
        all_metrics[cfg[0]] = metrics
        gc.collect()

    print(f"\n{'=' * 60}")
    print("  TCN BASELINE — FINAL SUMMARY")
    print(f"{'=' * 60}")
    header = f"  {'Size':>10s} | {'ROC-AUC':>8s} | {'PR-AUC':>8s} | {'F1':>6s} | {'Brier':>7s} | {'Time':>8s}"
    print(header)
    print(f"  {'-' * len(header.strip())}")
    for label, m in all_metrics.items():
        t = m.get("training_time_seconds", 0)
        print(
            f"  {label:>10s} | {m['roc_auc']:>8.4f} | {m['pr_auc']:>8.4f} | "
            f"{m['f1_score']:>6.4f} | {m['brier_score']:>7.4f} | {t:>7.0f}s"
        )

    total = time.time() - t_start
    print(f"\n  Total wall time: {total:.1f}s ({total / 60:.1f} min)")
    print(f"{'=' * 60}")
    print("  T3a.4 Complete.")


if __name__ == "__main__":
    main()
