"""Feature loading, categorical encoding, and numeric scaling utilities."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.constants import EXCLUDE_FEATURES, CAT_COLS
from src.io_utils import get_models_dir, load_json


def get_feature_cols(available_cols: set[str] | None = None,
                     exclude: set[str] | None = None) -> list[str]:
    """Load feature list from models/feature_list.json, minus exclusions.

    Parameters
    ----------
    available_cols :
        If given, only return features present in this set.
    exclude :
        Extra columns to exclude (merged with EXCLUDE_FEATURES).
    """
    fl = load_json(get_models_dir() / "feature_list.json")
    exclude_set = set(EXCLUDE_FEATURES) | (exclude or set())
    features = [c for c in fl["all_features"] if c not in exclude_set]
    if available_cols is not None:
        features = [c for c in features if c in available_cols]
    return features


def encode_categoricals(df: pl.DataFrame,
                        cat_cols: list[str] | None = None
                        ) -> tuple[pl.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns in-place."""
    cat_cols = cat_cols or CAT_COLS
    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].to_numpy())
            df = df.with_columns([pl.Series(encoded).alias(col)])
            encoders[col] = le
    return df, encoders


def scale_numeric_features(df: pl.DataFrame,
                           feature_cols: list[str],
                           scaler: dict[str, dict[str, float]]) -> pl.DataFrame:
    """Apply standard scaling to numeric features from a scaler dict."""
    for col in feature_cols:
        if col in scaler:
            params = scaler[col]
            if params["std"] > 0:
                df = df.with_columns([
                    ((pl.col(col) - params["mean"]) / params["std"]).alias(col)
                ])
    return df


# Size-range config tuples: (label, input_name, target_name, sample_fraction)
SIZE_CONFIGS = [
    ("25_50cm", "vcm_ml_ready_25_50cm.parquet", "targets_25_50cm.parquet", 1.0),
    ("10_25cm", "vcm_ml_ready_10_25cm.parquet", "targets_10_25cm.parquet", 1.0),
    ("5_10cm",  "vcm_ml_ready_5_10cm.parquet",  "targets_5_10cm.parquet",  0.1),
]
