"""
T3a.2 — Random Forest Baseline (Snapshot per Size Range) — Memory Optimized
============================================================================
Trains RF classifier on the last-epoch snapshot for each size range.

Memory optimizations:
- Reduce n_estimators to 100, max_depth to 15
- max_samples=0.3 to limit bootstrap sample size
- Sample 5-10cm to 10% of windows
- Process sequentially, free memory between sizes
- Use float32 for numpy arrays
"""

import polars as pl
import numpy as np
import json
import time
import gc
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    accuracy_score,
)

from src.constants import CAT_COLS, EXCLUDE_FEATURES
from src.io_utils import (
    get_data_dir,
    get_models_dir,
    get_reports_dir,
    load_json,
    save_json,
    timer,
)
from src.feature_utils import get_feature_cols, encode_categoricals, SIZE_CONFIGS

DATA_DIR = get_data_dir()
MODELS_DIR = get_models_dir()
REPORTS_DIR = get_reports_dir()


def process_size_range(size_label, input_name, target_name, sample_frac=1.0):
    """Process one size range with memory-efficient approach."""
    t_size = timer()
    print(f"\n{'=' * 60}")
    print(f"  Size Range: {size_label}  (sample={sample_frac})")
    print(f"{'=' * 60}")

    # 1. Get available features
    print("  Inspecting feature columns...")
    temp = pl.scan_parquet(DATA_DIR / input_name).head(1).collect()
    feature_cols = get_feature_cols(available_cols=set(temp.columns))
    num_cols = [c for c in feature_cols if c not in CAT_COLS]
    cat_cols = [c for c in feature_cols if c in CAT_COLS]
    print(
        f"    {len(feature_cols)} features ({len(num_cols)} num + {len(cat_cols)} cat)"
    )

    # 2. Load targets, optionally sample
    print("  Loading targets...")
    targets = pl.read_parquet(DATA_DIR / target_name)
    if sample_frac < 1.0:
        targets = targets.group_by("split").map_groups(
            lambda g: g.sample(fraction=sample_frac, seed=42)
        )
        gc.collect()
    print(f"    {len(targets):,} windows after sampling")

    # 3. Load features and add epoch_idx
    print("  Loading features and indexing by epoch...")
    df = pl.scan_parquet(DATA_DIR / input_name)
    df = df.sort(["satellite_number", "epoch_dt"])
    df = df.with_columns(
        [pl.int_range(0, pl.len()).over("satellite_number").alias("epoch_idx")]
    )
    keep_cols = ["satellite_number", "epoch_idx", "split"] + feature_cols
    df = df.select(keep_cols).collect()
    print(f"    {len(df):,} rows with {len(df.columns)} cols")

    # 4. Join (smaller targets first, then join features)
    print("  Joining targets to features...")
    joined = targets.join(
        df,
        left_on=["satellite_number", "window_end_idx"],
        right_on=["satellite_number", "epoch_idx"],
        how="inner",
    )
    del df, targets
    gc.collect()
    print(f"    Joined: {len(joined):,} rows")

    # 5. Encode categoricals
    joined, label_encoders = encode_categoricals(joined, cat_cols)

    # 6. Split into train/val/test
    train = joined.filter(pl.col("split") == "train")
    val = joined.filter(pl.col("split") == "val")
    test = joined.filter(pl.col("split") == "test")

    del joined
    gc.collect()

    X_train = train.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train["collision_risk"].to_numpy().astype(np.float32)
    X_val = val.select(feature_cols).to_numpy().astype(np.float32)
    y_val = val["collision_risk"].to_numpy().astype(np.float32)
    X_test = test.select(feature_cols).to_numpy().astype(np.float32)
    y_test = test["collision_risk"].to_numpy().astype(np.float32)

    n_train_high = int(y_train.sum())
    print(
        f"\n    Train: {len(X_train):,} ({n_train_high:,} high, {y_train.mean() * 100:.1f}%)"
    )
    print(f"    Val:   {len(X_val):,}")
    print(f"    Test:  {len(X_test):,}")

    del train, val, test
    gc.collect()

    # 7. Train RF
    print(
        "\n  Training RandomForest (n_estimators=100, max_depth=15, max_samples=0.3)..."
    )
    t_train = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=10,
        max_samples=0.3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    rf.fit(X_train, y_train)
    train_time = time.time() - t_train

    del X_train, y_train
    gc.collect()

    print(f"    Trained in {train_time:.1f}s")
    print(f"    Model size: {len(rf.estimators_)} trees")

    # 8. Evaluate on test
    print("\n  Evaluating on test set...")
    y_prob = rf.predict_proba(X_test)[:, 1].astype(np.float64)
    y_pred = rf.predict(X_test)

    metrics = {
        "size_range": size_label,
        "sampling_fraction": sample_frac,
        "n_train": int(len(X_test) + len(X_val)),
        "n_test": int(len(X_test)),
        "test_high_risk_ratio": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "training_time_seconds": float(train_time),
        "n_features": len(feature_cols),
    }

    print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"    PR-AUC:  {metrics['pr_auc']:.4f}")
    print(f"    F1:      {metrics['f1_score']:.4f}")
    print(f"    Brier:   {metrics['brier_score']:.4f}")
    print(f"    Acc:     {metrics['accuracy']:.4f}")

    importances = sorted(
        zip(feature_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True
    )[:10]
    metrics["top_features"] = [
        {"feature": f, "importance": round(float(i), 6)} for f, i in importances
    ]
    print("\n  Top 5 features:")
    for f, i in importances[:5]:
        print(f"    {f}: {i:.6f}")

    # 9. Save
    model_path = MODELS_DIR / f"rf_{size_label}.pkl"
    joblib.dump(
        {
            "model": rf,
            "label_encoders": label_encoders,
            "feature_cols": feature_cols,
        },
        model_path,
        compress=3,
    )
    print(f"\n  Saved: {model_path} ({model_path.stat().st_size / 1024**2:.1f} MB)")

    metrics_path = REPORTS_DIR / f"rf_{size_label}_metrics.json"
    save_json(metrics, metrics_path)
    print(f"  Saved: {metrics_path}")

    del X_test, y_test, y_prob, y_pred
    gc.collect()

    print(f"  Time: {t_size.elapsed:.1f}s")
    return metrics


def main():
    print("=" * 60)
    print("  T3a.2 — Random Forest Baseline (Memory Optimized)")
    print("  n_est=100 | max_depth=15 | max_samples=0.3 | float32")
    print("=" * 60)

    t_start = time.time()
    all_metrics = {}

    for size_label, input_name, target_name, sample_frac in SIZE_CONFIGS:
        metrics = process_size_range(size_label, input_name, target_name, sample_frac)
        all_metrics[size_label] = metrics
        gc.collect()

    print(f"\n{'=' * 60}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"  {'Size':>10s} | {'ROC-AUC':>8s} | {'PR-AUC':>8s} | {'F1':>6s} | {'Brier':>7s}"
    )
    print(f"  {'-' * 48}")
    for label, m in all_metrics.items():
        print(
            f"  {label:>10s} | {m['roc_auc']:>8.4f} | {m['pr_auc']:>8.4f} | "
            f"{m['f1_score']:>6.4f} | {m['brier_score']:>7.4f}"
        )

    print(f"\n  Total time: {time.time() - t_start:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
