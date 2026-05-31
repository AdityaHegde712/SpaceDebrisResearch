"""
T3a.1 — Synthetic Collision Target Creation
=============================================
Creates per-satellite sliding-window targets for Phase 3a ML Baselines.

Approach:
  1. For each size-range parquet, load and sort by satellite + epoch.
  2. Restructure into per-satellite sliding windows (len=20, stride=1).
  3. Label collision risk via **per-band** percentile of log_uncertainty_volume
     targeting ~30% high-risk overall.
  4. Extract multi-task regression targets from the *next* epoch after each window.
  5. Propagate train/val/test split labels (no boundary leakage).
  6. Save per-size target parquets + metadata JSON + summary report.

Outputs:
  data/targets_25_50cm.parquet   — one row per window
  data/targets_10_25cm.parquet
  data/targets_5_10cm.parquet
  data/targets_metadata.json     — per-band thresholds & class balance
  reports/t3a1_target_summary.json  — summary stats for verification
"""

import polars as pl
import numpy as np
import json
import time
from pathlib import Path

from src.constants import SEQ_LEN
from src.io_utils import get_data_dir, get_reports_dir, save_parquet, save_json, timer
from src.feature_utils import SIZE_CONFIGS

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR = get_data_dir()
REPORTS_DIR = get_reports_dir()

TARGET_HIGH_RISK_RATIO = 0.30  # aim for ~30% high-risk overall per size

SIZE_RANGES = [
    (label, input_name, target_name)
    for label, input_name, target_name, _ in SIZE_CONFIGS
]

TARGET_COLS = [
    "j2k_pos_x",
    "j2k_pos_y",
    "j2k_pos_z",
    "rtn_sigma_r_km",
    "rtn_sigma_t_km",
    "rtn_sigma_w_km",
]

METADATA_COLS = ["satellite_number", "sequence_id", "seq_pos", "split", "epoch_dt"]

REQUIRED_COLS = (
    TARGET_COLS
    + METADATA_COLS
    + ["altitude_band", "log_uncertainty_volume", "uncertainty_volume_km3"]
)


# ── Helpers ───────────────────────────────────────────────────────────────


def find_threshold_for_target_ratio(values, target_ratio):
    """Find the percentile threshold that gives approx `target_ratio` high-risk.

    Since we want top `target_ratio * 100` % of values to be high-risk,
    the threshold is the (1 - target_ratio) quantile.
    Returns (threshold, actual_ratio_above_threshold).
    """
    if len(values) == 0:
        return 0.0, 0.0
    q = 1.0 - target_ratio
    threshold = float(np.quantile(values.to_numpy(), q))
    # Because of discrete distribution, actual ratio may differ slightly
    actual_ratio = (values > threshold).mean()
    return threshold, float(actual_ratio)


def compute_per_band_thresholds(df, target_ratio=0.30):
    """Compute per-altitude-band log_uncertainty_volume thresholds.

    Returns:
      thresholds: dict {band_name: threshold_value}
      band_stats: dict with counts and actual ratios per band
    """
    bands = df["altitude_band"].unique().to_list()
    bands.sort()

    thresholds = {}
    band_stats = {}
    overall_count = 0
    overall_high = 0

    for band in bands:
        band_vals = df.filter(pl.col("altitude_band") == band)["log_uncertainty_volume"]
        n_band = len(band_vals)
        if n_band < 10:
            # Too few samples — use a fallback
            thresholds[band] = float(band_vals.quantile(0.70)) if n_band > 0 else 0.0
            actual_ratio = (band_vals > thresholds[band]).mean() if n_band > 0 else 0.0
        else:
            thresh, actual_ratio = find_threshold_for_target_ratio(
                band_vals, target_ratio
            )
            thresholds[band] = thresh

        n_high = int((band_vals > thresholds[band]).sum())
        overall_count += n_band
        overall_high += n_high

        band_stats[band] = {
            "count": int(n_band),
            "threshold_log_uncertainty_volume": thresholds[band],
            "threshold_uncertainty_volume_km3": float(np.exp(thresholds[band])),
            "high_risk_count": n_high,
            "high_risk_ratio": float(n_high / n_band) if n_band > 0 else 0.0,
        }

        # Also compute unc_volume p90 for secondary check
        if n_band > 0:
            vol_p90 = float(band_vals.quantile(0.90))
            band_stats[band]["log_uncertainty_volume_p90"] = vol_p90
            band_stats[band]["uncertainty_volume_p90_km3"] = float(np.exp(vol_p90))

    band_stats["_overall"] = {
        "count": overall_count,
        "high_risk_count": overall_high,
        "high_risk_ratio": float(overall_high / overall_count)
        if overall_count > 0
        else 0.0,
    }

    return thresholds, band_stats


def process_size_range(size_label, input_name, output_name, target_ratio=0.30):
    """Process a single size range: load → windows → targets → save."""
    t = timer()
    input_path = DATA_DIR / input_name
    output_path = DATA_DIR / output_name

    print(f"\n{'=' * 60}")
    print(f"  Processing: {size_label}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}")

    # ── 1. Load ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    df = pl.read_parquet(input_path)
    n_rows = len(df)
    print(f"  Loaded {n_rows:,} rows, {len(df.columns)} columns")

    # Validate required columns exist
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        # Some may have different names; try to handle gracefully
        print(f"  WARNING: Missing columns: {missing}")
        # We can proceed with what we have

    # Check for epoch_delta_days — might not be in all files
    has_epoch_delta = "epoch_delta_days" in df.columns
    print(f"  Has epoch_delta_days: {has_epoch_delta}")

    # ── 2. Sort & index ──────────────────────────────────────────────────
    print("[2/7] Sorting by satellite and epoch...")
    df = df.sort(["satellite_number", "epoch_dt"])

    df = df.with_columns(
        [
            pl.int_range(0, pl.len()).over("satellite_number").alias("epoch_idx"),
        ]
    )

    # Filter to satellites with at least SEQ_LEN epochs (needed for any window)
    sat_epoch_counts = df.group_by("satellite_number").agg(pl.len().alias("n_epochs"))
    valid_sats = sat_epoch_counts.filter(pl.col("n_epochs") >= SEQ_LEN)[
        "satellite_number"
    ]
    n_valid_sats = len(valid_sats)
    n_total_sats = len(sat_epoch_counts)

    df = df.filter(pl.col("satellite_number").is_in(valid_sats.implode()))
    print(f"  Satellites with >= {SEQ_LEN} epochs: {n_valid_sats:,} / {n_total_sats:,}")
    print(f"  Rows after filter: {len(df):,}")

    # ── 3. Per-band percentile thresholds ────────────────────────────────
    print("[3/7] Computing per-band collision risk thresholds...")
    band_thresholds, band_stats = compute_per_band_thresholds(df, target_ratio)

    # Label each row with collision risk based on its band
    # Use a join approach for clean mapping
    thresh_df = pl.DataFrame(
        {
            "altitude_band": list(band_thresholds.keys()),
            "risk_threshold": list(band_thresholds.values()),
        }
    )
    df = df.join(thresh_df, on="altitude_band", how="left")

    df = df.with_columns(
        [
            (pl.col("log_uncertainty_volume") > pl.col("risk_threshold"))
            .fill_null(False)  # shouldn't happen, but safety
            .alias("collision_risk"),
        ]
    )

    # Also compute uncertainty_volume > p90 as secondary check
    # We need to compute p90 per band for uncertainty_volume_km3
    # (using raw volume, not log)
    vol_p90_per_band = {}
    for band in band_thresholds.keys():
        if band in band_stats:
            vol_p90_per_band[band] = band_stats[band]["uncertainty_volume_p90_km3"]

    # Map the vol p90 to each row
    if vol_p90_per_band:
        vol_p90_df = pl.DataFrame(
            {
                "altitude_band": list(vol_p90_per_band.keys()),
                "vol_p90_threshold": list(vol_p90_per_band.values()),
            }
        )
        df = df.join(vol_p90_df, on="altitude_band", how="left")
        df = df.with_columns(
            [
                (pl.col("uncertainty_volume_km3") > pl.col("vol_p90_threshold"))
                .fill_null(False)
                .alias("unc_vol_above_p90"),
            ]
        )
    else:
        df = df.with_columns([pl.lit(False).alias("unc_vol_above_p90")])

    risk_count = df["collision_risk"].sum()
    risk_ratio = risk_count / len(df)
    print(
        f"  Overall collision risk ratio: {risk_ratio:.4f} ({risk_count:,}/{len(df):,})"
    )
    print("  Per-band thresholds:")
    for band in sorted(band_thresholds.keys()):
        bs = band_stats[band]
        print(
            f"    {band:12s}: thresh={bs['threshold_log_uncertainty_volume']:7.3f}, "
            f"n={bs['count']:>8,}, high={bs['high_risk_count']:>8,} "
            f"({bs['high_risk_ratio'] * 100:5.1f}%)"
        )

    # ── 4. Shift to get future targets ────────────────────────────────────
    print("[4/7] Shifting to get next-epoch future targets...")
    # Use shift(-1) within each satellite to get the NEXT epoch's values.
    # The last row of each satellite gets null — correctly marking no future.
    df = df.with_columns(
        [
            pl.col("j2k_pos_x")
            .shift(-1)
            .over("satellite_number")
            .alias("future_pos_x"),
            pl.col("j2k_pos_y")
            .shift(-1)
            .over("satellite_number")
            .alias("future_pos_y"),
            pl.col("j2k_pos_z")
            .shift(-1)
            .over("satellite_number")
            .alias("future_pos_z"),
            pl.col("rtn_sigma_r_km")
            .shift(-1)
            .over("satellite_number")
            .alias("future_sigma_r"),
            pl.col("rtn_sigma_t_km")
            .shift(-1)
            .over("satellite_number")
            .alias("future_sigma_t"),
            pl.col("rtn_sigma_w_km")
            .shift(-1)
            .over("satellite_number")
            .alias("future_sigma_w"),
            pl.col("epoch_dt")
            .shift(-1)
            .over("satellite_number")
            .alias("future_epoch_dt"),
        ]
    )

    # has_future = True if any future column is non-null
    df = df.with_columns(
        [
            pl.col("future_pos_x").is_not_null().alias("has_future"),
        ]
    )

    # ── 5. Create window-end rows ────────────────────────────────────────
    print("[5/7] Creating sliding window targets...")
    # Every row with epoch_idx >= 19 is the LAST epoch of a window.
    # The future values (shifted) for that row ARE the next epoch's values.
    windows = df.filter(pl.col("epoch_idx") >= (SEQ_LEN - 1)).select(
        [
            "satellite_number",
            (pl.col("epoch_idx") - (SEQ_LEN - 1)).alias("window_start_idx"),
            pl.col("epoch_idx").alias("window_end_idx"),
            "collision_risk",
            "unc_vol_above_p90",
            "has_future",
            "future_pos_x",
            "future_pos_y",
            "future_pos_z",
            "future_sigma_r",
            "future_sigma_t",
            "future_sigma_w",
            "future_epoch_dt",
            "split",
            "altitude_band",
        ]
    )
    n_windows = len(windows)

    # Verify has_future
    n_with_future = windows["has_future"].sum()
    print(f"  Window-end rows: {n_windows:,}")
    print(
        f"  Windows with future: {n_with_future:,} / {n_windows:,} "
        f"({n_with_future / n_windows * 100:.1f}%)"
    )

    # ── 6. Add window_id ─────────────────────────────────────────────────
    print("[6/7] Adding window IDs...")
    windows = windows.with_columns(
        [
            pl.int_range(0, pl.len()).over("satellite_number").alias("window_id"),
        ]
    )

    # Reorder columns for clean output
    windows = windows.select(
        [
            "satellite_number",
            "window_id",
            "window_start_idx",
            "window_end_idx",
            "collision_risk",
            "unc_vol_above_p90",
            "has_future",
            "future_pos_x",
            "future_pos_y",
            "future_pos_z",
            "future_sigma_r",
            "future_sigma_t",
            "future_sigma_w",
            "future_epoch_dt",
            "split",
            "altitude_band",
        ]
    )

    # ── 7. Save ──────────────────────────────────────────────────────────
    print(f"[7/7] Saving to {output_path}...")
    save_parquet(windows, output_path)
    fsize_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {len(windows):,} windows, {fsize_mb:.1f} MB")

    # Summary stats
    n_high = windows["collision_risk"].sum()
    n_low = n_windows - n_high
    ratio = n_high / n_windows if n_windows > 0 else 0

    split_counts = (
        windows.group_by("split")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("collision_risk").sum().alias("high_risk_count"),
            ]
        )
        .sort("split")
    )

    elapsed = t.elapsed

    summary = {
        "size_range": size_label,
        "n_satellites": int(n_valid_sats),
        "n_satellites_total": int(n_total_sats),
        "n_windows": int(n_windows),
        "n_high_risk": int(n_high),
        "n_low_risk": int(n_low),
        "high_risk_ratio": float(ratio),
        "n_with_future": int(n_with_future),
        "has_future_ratio": float(n_with_future / n_windows) if n_windows > 0 else 0.0,
        "elapsed_seconds": float(elapsed),
        "per_band_thresholds": {
            b: band_stats[b] for b in sorted(band_thresholds.keys())
        },
        "overall_band_stats": band_stats["_overall"],
        "split_distribution": {
            row["split"]: {
                "count": int(row["count"]),
                "high_risk_count": int(row["high_risk_count"]),
                "high_risk_ratio": float(row["high_risk_count"] / row["count"]),
            }
            for row in split_counts.iter_rows(named=True)
        },
    }

    print("\n  -- Summary --")
    print(f"  Windows:     {n_windows:>10,}")
    print(f"  High-risk:   {n_high:>10,} ({ratio * 100:5.1f}%)")
    print(f"  Low-risk:    {n_low:>10,} ({(1 - ratio) * 100:5.1f}%)")
    print(
        f"  With future: {n_with_future:>10,} ({n_with_future / n_windows * 100:5.1f}%)"
    )
    print(f"  Time:        {elapsed:.1f}s")

    return summary, band_stats


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  T3a.1 — Synthetic Collision Target Creation")
    print("  Per-satellite sliding windows | Per-band percentile labeling")
    print("=" * 60)

    t_start = time.time()

    all_summaries = {}
    all_band_stats = {}

    # Process each size range
    for size_label, input_name, output_name in SIZE_RANGES:
        summary, band_stats = process_size_range(
            size_label,
            input_name,
            output_name,
            target_ratio=TARGET_HIGH_RISK_RATIO,
        )
        all_summaries[size_label] = summary
        all_band_stats[size_label] = band_stats

    # ── Save metadata ────────────────────────────────────────────────────
    metadata = {
        "created_by": "T3a.1_create_targets.py",
        "description": "Per-satellite sliding window targets for Phase 3a ML baselines",
        "sequence_length": SEQ_LEN,
        "stride": 1,
        "target_high_risk_ratio": TARGET_HIGH_RISK_RATIO,
        "labeling_method": "per_band_percentile_log_uncertainty_volume",
        "per_size_band_thresholds": all_band_stats,
        "per_size_summaries": {
            label: {k: v for k, v in summ.items() if k != "per_band_thresholds"}
            for label, summ in all_summaries.items()
        },
    }

    meta_path = DATA_DIR / "targets_metadata.json"
    save_json(metadata, meta_path)
    print(f"\nMetadata saved to {meta_path}")

    # ── Save summary report ──────────────────────────────────────────────
    summary_report = {
        "script": "t3a1_create_targets.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": time.time() - t_start,
        "target_high_risk_ratio": TARGET_HIGH_RISK_RATIO,
        "per_size_range": {},
    }

    for label, summ in all_summaries.items():
        # Compact summary for the report
        s = {
            "n_windows": summ["n_windows"],
            "n_high_risk": summ["n_high_risk"],
            "n_low_risk": summ["n_low_risk"],
            "high_risk_ratio": summ["high_risk_ratio"],
            "n_with_future": summ["n_with_future"],
            "has_future_ratio": summ["has_future_ratio"],
            "n_satellites_used": summ["n_satellites"],
            "n_satellites_total": summ["n_satellites_total"],
            "elapsed_seconds": summ["elapsed_seconds"],
            "per_band_thresholds": {
                b: {
                    "threshold_log_unc_vol": sbs["threshold_log_uncertainty_volume"],
                    "high_risk_ratio": sbs["high_risk_ratio"],
                    "count": sbs["count"],
                }
                for b, sbs in summ["per_band_thresholds"].items()
            },
            "split_distribution": summ["split_distribution"],
        }
        summary_report["per_size_range"][label] = s

    report_path = REPORTS_DIR / "t3a1_target_summary.json"
    save_json(summary_report, report_path)
    print(f"Report saved to {report_path}")

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  T3a.1 Complete!")
    print(f"  Total time: {time.time() - t_start:.1f}s")
    print(f"{'=' * 60}")
    print(f"\n{'-' * 60}")
    print(
        f"  {'Size':>12s} | {'Windows':>10s} | {'HighRisk':>8s} | {'Ratio':>6s} | {'Future%':>7s}"
    )
    print(f"{'-' * 60}")
    for label, summ in all_summaries.items():
        print(
            f"  {label:>12s} | {summ['n_windows']:>10,d} | {summ['n_high_risk']:>8,d} | "
            f"{summ['high_risk_ratio'] * 100:5.1f}% | {summ['has_future_ratio'] * 100:5.1f}%"
        )
    print(f"{'-' * 60}")

    # Verification checks
    print(f"\n{'=' * 60}")
    print("  Verification Checks")
    print(f"{'=' * 60}")

    all_ok = True

    for label, summ in all_summaries.items():
        print(f"\n  -- {label} --")

        # Check 1: High-risk ratio 25-35%
        ratio = summ["high_risk_ratio"]
        if 0.25 <= ratio <= 0.35:
            print(f"  [OK] [1] High-risk ratio {ratio * 100:.1f}% is in [25%, 35%]")
        else:
            print(
                f"  [WARN] [1] High-risk ratio {ratio * 100:.1f}% is outside [25%, 35%]"
            )
            all_ok = False

        # Check 2: All splits present
        splits = summ["split_distribution"]
        expected_splits = {"train", "val", "test"}
        found_splits = set(splits.keys())
        missing_splits = expected_splits - found_splits
        if not missing_splits:
            print(f"  [OK] [2] All splits present: {', '.join(found_splits)}")
            for split_name, split_data in splits.items():
                print(
                    f"       {split_name}: {split_data['count']:>10,} windows "
                    f"(high-risk: {split_data['high_risk_ratio'] * 100:.1f}%)"
                )
        else:
            print(f"  [WARN] [2] Missing splits: {missing_splits}")
            all_ok = False

        # Check 3: has_future >= 80%
        future_ratio = summ["has_future_ratio"]
        if future_ratio >= 0.80:
            print(f"  [OK] [3] has_future = {future_ratio * 100:.1f}% (>= 80%)")
        else:
            print(f"  [WARN] [3] has_future = {future_ratio * 100:.1f}% (< 80%)")
            all_ok = False

    print(f"\n{'=' * 60}")
    if all_ok:
        print("  [OK] All verification checks passed!")
    else:
        print("  [WARN] Some checks need attention (see above)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
