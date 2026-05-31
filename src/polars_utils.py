"""Polars DataFrame helpers for common VCM pipeline transformations."""

from __future__ import annotations

import polars as pl


def sort_sat_epoch(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by satellite_number then epoch_dt."""
    return df.sort(["satellite_number", "epoch_dt"])


def add_epoch_idx(df: pl.DataFrame) -> pl.DataFrame:
    """Add zero-based epoch index per satellite."""
    return df.with_columns([
        pl.int_range(0, pl.len()).over("satellite_number").alias("epoch_idx"),
    ])


def add_altitude_band(df: pl.DataFrame, alt_col: str = "perigee_alt_km") -> pl.DataFrame:
    if alt_col not in df.columns:
        return df
    return df.with_columns([
        pl.when(pl.col(alt_col) < 200).then(pl.lit("<200km"))
        .when(pl.col(alt_col) < 400).then(pl.lit("200-400km"))
        .when(pl.col(alt_col) < 600).then(pl.lit("400-600km"))
        .when(pl.col(alt_col) < 800).then(pl.lit("600-800km"))
        .when(pl.col(alt_col) < 1000).then(pl.lit("800-1000km"))
        .otherwise(pl.lit("1000+km"))
        .alias("altitude_band"),
    ])


def add_orbit_regime(df: pl.DataFrame, alt_col: str = "perigee_alt_km") -> pl.DataFrame:
    if alt_col not in df.columns:
        return df
    return df.with_columns([
        pl.when(pl.col(alt_col) < 2000).then(pl.lit("LEO"))
        .when(pl.col(alt_col) < 35786).then(pl.lit("MEO"))
        .otherwise(pl.lit("GEO"))
        .alias("orbit_regime"),
    ])


def compute_time_gap_days(df: pl.DataFrame, time_col: str = "epoch_dt") -> pl.DataFrame:
    """Add time_gap_days between consecutive observations per satellite."""
    return df.with_columns([
        (pl.col(time_col).diff().over("satellite_number")
         .dt.total_seconds().fill_null(0.0)
         .alias("time_gap_seconds")),
    ]).with_columns([
        (pl.col("time_gap_seconds") / 86400.0).alias("time_gap_days"),
    ]).drop("time_gap_seconds")
