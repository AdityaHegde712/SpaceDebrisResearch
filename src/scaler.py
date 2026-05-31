"""StandardScaler wrapper that persists as a dict-of-params.

Replaces both phase2b_create_scaler.py and the inline scaler logic
in phase2_feature_engineering.py.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import numpy as np

from src.io_utils import save_pickle, load_pickle


class Scaler:
    """Standard scaler (mean, std) for numeric features.

    Usage:
        scaler = Scaler()
        scaler.fit(train_df, numeric_cols)
        train_scaled = scaler.transform(train_df)
        test_scaled  = scaler.transform(test_df)
        scaler.save("models/scaler.pkl")
        scaler2 = Scaler.load("models/scaler.pkl")
    """

    def __init__(self):
        self.params: dict[str, dict[str, float]] = {}
        self.feature_cols: list[str] = []

    def fit(self, df: pl.DataFrame, numeric_cols: list[str]) -> None:
        """Compute mean / std from *df* for each column in *numeric_cols*."""
        self.feature_cols = numeric_cols
        self.params = {}
        for col in numeric_cols:
            col_vals = df[col].to_numpy()
            valid = ~np.isnan(col_vals) & ~np.isinf(col_vals)
            if valid.sum() > 0:
                self.params[col] = {
                    "mean": float(np.mean(col_vals[valid])),
                    "std": float(np.std(col_vals[valid])),
                }
            else:
                self.params[col] = {"mean": 0.0, "std": 1.0}

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply standard scaling (in-place via new columns)."""
        for col in self.feature_cols:
            if col in self.params and self.params[col]["std"] > 0:
                p = self.params[col]
                df = df.with_columns([
                    ((pl.col(col) - p["mean"]) / p["std"]).alias(col)
                ])
        return df

    def fit_transform(self, df: pl.DataFrame, numeric_cols: list[str]) -> pl.DataFrame:
        self.fit(df, numeric_cols)
        return self.transform(df)

    def save(self, path: str | Path) -> None:
        save_pickle({"params": self.params, "feature_cols": self.feature_cols}, path)

    @staticmethod
    def load(path: str | Path) -> "Scaler":
        data = load_pickle(path)
        s = Scaler()
        s.params = data["params"]
        s.feature_cols = data.get("feature_cols", list(data["params"].keys()))
        return s
