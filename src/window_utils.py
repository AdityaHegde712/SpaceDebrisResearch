"""Sliding-window extraction using positional dictionary lookup.

Pulled from t3a4_tcn_baseline.py (memory-efficient approach).
"""

from __future__ import annotations

import gc
import time

import polars as pl
import numpy as np

from src.constants import SEQ_LEN
from src.polars_utils import sort_sat_epoch, add_epoch_idx

CHUNK_SIZE = 200_000


def extract_windows_positional(
    features: pl.DataFrame,
    targets: pl.DataFrame,
    feature_cols: list[str],
    seq_len: int = SEQ_LEN,
    chunk_size: int = CHUNK_SIZE,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract sliding-window arrays using positional dictionary lookup.

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test) — all float32.
    """

    def _log(msg: str) -> None:
        if verbose:
            print(f"      {msg}")

    t0_all = time.time()

    # 1. Sort features and add epoch_idx
    _log("[1/5] Sorting features and adding epoch_idx...")
    t0 = time.time()
    features = add_epoch_idx(sort_sat_epoch(features))
    _log(f"Done ({time.time() - t0:.1f}s)")

    # 2. Convert features to float32 numpy
    _log(
        "[2/5] Converting features to numpy "
        f"({len(features):,} rows x {len(feature_cols)} features)..."
    )
    t0 = time.time()
    features_np = features.select(feature_cols).to_numpy().astype(np.float32)
    _log(
        f"features_np shape: {features_np.shape}, "
        f"size: {features_np.nbytes / 1e6:.1f} MB ({time.time() - t0:.1f}s)"
    )

    # 3. Build position map
    _log("[3/5] Building position map...")
    t0 = time.time()
    sat_list = features["satellite_number"].to_list()
    epoch_list = features["epoch_idx"].to_list()
    pos_map = {}
    for i in range(len(sat_list)):
        pos_map[(sat_list[i], epoch_list[i])] = i
    _log(f"{len(pos_map):,} entries ({time.time() - t0:.1f}s)")

    del features, sat_list, epoch_list
    gc.collect()

    # 4. Prepare target arrays
    _log(f"[4/5] Preparing target arrays ({len(targets):,} windows)...")
    t0 = time.time()
    sat_nos = targets["satellite_number"].to_numpy()
    start_idxs = targets["window_start_idx"].to_numpy()
    y_cls = targets["collision_risk"].to_numpy().astype(np.float32)
    splits = targets["split"].to_list()
    _log(
        f"y mean: {y_cls.mean():.4f}, n_high: {int(y_cls.sum()):,} "
        f"({time.time() - t0:.1f}s)"
    )

    n_windows = len(targets)
    n_features = len(feature_cols)

    # 5. Build window arrays in chunks
    _log(f"[5/5] Building window arrays in chunks of {chunk_size:,}...")
    t0 = time.time()

    n_train = sum(1 for s in splits if s == "train")
    n_val = sum(1 for s in splits if s == "val")
    n_test = sum(1 for s in splits if s == "test")

    X_train = np.empty((n_train, seq_len, n_features), dtype=np.float32)
    y_train = np.empty(n_train, dtype=np.float32)
    X_val = np.empty((n_val, seq_len, n_features), dtype=np.float32)
    y_val = np.empty(n_val, dtype=np.float32)
    X_test = np.empty((n_test, seq_len, n_features), dtype=np.float32)
    y_test = np.empty(n_test, dtype=np.float32)

    train_ptr, val_ptr, test_ptr = 0, 0, 0
    chunk_start = 0
    n_chunks = (n_windows + chunk_size - 1) // chunk_size

    while chunk_start < n_windows:
        chunk_end = min(chunk_start + chunk_size, n_windows)
        chunk_n = chunk_end - chunk_start
        chunk_idx = chunk_start // chunk_size + 1
        _log(
            f"Chunk {chunk_idx}/{n_chunks} "
            f"(windows {chunk_start:,}–{chunk_end - 1:,})..."
        )

        for i in range(chunk_start, chunk_end):
            sat_no = int(sat_nos[i])
            start_idx = int(start_idxs[i])
            start_pos = pos_map[(sat_no, start_idx)]
            window_data = features_np[start_pos : start_pos + seq_len]

            s = splits[i]
            if s == "train":
                X_train[train_ptr] = window_data
                y_train[train_ptr] = y_cls[i]
                train_ptr += 1
            elif s == "val":
                X_val[val_ptr] = window_data
                y_val[val_ptr] = y_cls[i]
                val_ptr += 1
            else:
                X_test[test_ptr] = window_data
                y_test[test_ptr] = y_cls[i]
                test_ptr += 1

        chunk_start = chunk_end
        gc.collect()

    elapsed = time.time() - t0
    _log(f"Done ({elapsed:.1f}s)")
    _log(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    _log(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    _log(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    total_time = time.time() - t0_all
    peak_features_mb = features_np.nbytes / 1e6
    total_windows_mb = (X_train.nbytes + X_val.nbytes + X_test.nbytes) / 1e6
    _log(
        f"Peak memory: features={peak_features_mb:.0f} MB, "
        f"windows={total_windows_mb:.0f} MB, total={total_time:.1f}s"
    )

    del pos_map, features_np, sat_nos, start_idxs, y_cls, splits
    gc.collect()

    return X_train, y_train, X_val, y_val, X_test, y_test
