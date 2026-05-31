"""
Phase 3b — Physics Baseline (Optimized Segment-by-Segment)
============================================================

Propagates test satellites from first VCM epoch to each subsequent
epoch using segment-by-segment fixed-step RK4 (dt=120s).
Error = direct 3D position difference at exact same epoch time.

Bucketed by horizon (1d±0.5, 3d±1, 7d±2) and altitude band.

Output:
  reports/physics_baseline_errors.json
  reports/physics_baseline_errors.csv

Usage:
  python scripts/phase3b_physics_baseline.py
"""

import numpy as np
import polars as pl
import json
import time
import sys

sys.path.insert(0, ".")

from models.physics_propagator import propagate_fixed_rk4

from src.constants import SECONDS_PER_DAY
from src.io_utils import get_data_dir, get_reports_dir, save_json, timer

DATA_DIR = get_data_dir()
REPORTS_DIR = get_reports_dir()
INPUT_FILE = DATA_DIR / "vcm_ml_ready.parquet"
OUTPUT_JSON = REPORTS_DIR / "physics_baseline_errors.json"
OUTPUT_CSV = REPORTS_DIR / "physics_baseline_errors.csv"

SAMPLE_SIZE = 200
DT = 120.0
MIN_SPAN_HOURS = 6
MAX_SPAN_DAYS = 30

HORIZON_BUCKETS = {
    "1d": (1.0, 0.5),
    "3d": (3.0, 1.0),
    "7d": (7.0, 2.0),
}


def main():
    t = timer()

    print("[1/4] Loading test-set data...")
    df = pl.read_parquet(
        INPUT_FILE,
        columns=[
            "satellite_number",
            "epoch_dt",
            "split",
            "j2k_pos_x",
            "j2k_pos_y",
            "j2k_pos_z",
            "j2k_vel_x",
            "j2k_vel_y",
            "j2k_vel_z",
            "mean_altitude_km",
            "size_range",
        ],
    )
    test = df.filter(pl.col("split") == "test")
    print(f"  Test rows: {len(test):,}")

    # Sort by sat + epoch FIRST so group_by preserves alignment
    test = test.sort(["satellite_number", "epoch_dt"])
    sats = (
        test.group_by("satellite_number", maintain_order=True)
        .agg(
            [
                pl.col("epoch_dt").alias("epochs"),
                pl.col("j2k_pos_x").alias("pos_x"),
                pl.col("j2k_pos_y").alias("pos_y"),
                pl.col("j2k_pos_z").alias("pos_z"),
                pl.col("j2k_vel_x").first().alias("vx0"),
                pl.col("j2k_vel_y").first().alias("vy0"),
                pl.col("j2k_vel_z").first().alias("vz0"),
                pl.col("mean_altitude_km").mean().alias("alt_km"),
                pl.col("size_range").first().alias("size_range"),
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("epochs").list.last() - pl.col("epochs").list.first())
                    .dt.total_seconds()
                    .alias("time_span_s")
                ),
            ]
        )
        .filter(pl.col("time_span_s") > MIN_SPAN_HOURS * 3600)
    )

    print(f"  Sats > {MIN_SPAN_HOURS}h: {len(sats):,}")
    sample = sats.sample(n=min(SAMPLE_SIZE, len(sats)), seed=42).to_dicts()
    n_sats = len(sample)
    print(f"  Sampled: {n_sats} sats (max {MAX_SPAN_DAYS}d each)\n")

    # ----------------------------------------------------------------
    # Propagate
    # ----------------------------------------------------------------
    print(f"[2/4] Propagating segment-by-segment (dt={DT}s)...")
    all_errors = []
    total_steps = 0

    for idx, row in enumerate(sample):
        if idx % 50 == 0 and idx > 0:
            elapsed = time.time() - t_start
            rate = idx / elapsed
            rem = (n_sats - idx) / rate
            print(f"  [{idx}/{n_sats}] {elapsed:.0f}s, ~{rem:.0f}s remaining")

        epochs = row["epochs"]
        pos_x = np.array(row["pos_x"], dtype=float)
        pos_y = np.array(row["pos_y"], dtype=float)
        pos_z = np.array(row["pos_z"], dtype=float)
        epochs_s = np.array([e.timestamp() for e in epochs], dtype=float)
        epoch0_s = epochs_s[0]

        state = np.array(
            [pos_x[0], pos_y[0], pos_z[0], row["vx0"], row["vy0"], row["vz0"]],
            dtype=float,
        )

        last_s = epoch0_s
        max_s = epoch0_s + MAX_SPAN_DAYS * 86400

        for i in range(1, len(epochs_s)):
            t_epoch = epochs_s[i]
            if t_epoch > max_s:
                break

            delta_s = t_epoch - last_s
            if delta_s <= 0:
                continue

            try:
                state = propagate_fixed_rk4(state, delta_s, dt=DT)
            except Exception:
                break

            true_pos = np.array([pos_x[i], pos_y[i], pos_z[i]])
            err_3d = float(np.linalg.norm(state[:3] - true_pos))
            days = (t_epoch - epoch0_s) / SECONDS_PER_DAY
            all_errors.append(
                (
                    row["satellite_number"],
                    row["alt_km"],
                    row["size_range"],
                    days,
                    err_3d,
                )
            )
            last_s = t_epoch

    # ----------------------------------------------------------------
    print("\n[3/4] Horizon-bucket statistics...")
    err_arr = np.array(
        all_errors,
        dtype=[
            ("sat_id", int),
            ("alt", float),
            ("size", object),
            ("days", float),
            ("err_km", float),
        ],
    )

    summary = {}
    for h_name, (h_day, h_hw) in HORIZON_BUCKETS.items():
        mask = (err_arr["days"] >= h_day - h_hw) & (err_arr["days"] <= h_day + h_hw)
        subset = err_arr[mask]
        if len(subset) == 0:
            continue
        vals = subset["err_km"]
        summary[h_name] = {
            "n": int(len(vals)),
            "mean_km": float(np.mean(vals)),
            "median_km": float(np.median(vals)),
            "std_km": float(np.std(vals)),
            "p90_km": float(np.percentile(vals, 90)),
            "p95_km": float(np.percentile(vals, 95)),
            "p99_km": float(np.percentile(vals, 99)),
            "min_km": float(np.min(vals)),
            "max_km": float(np.max(vals)),
        }
        print(
            f"  {h_name}: n={summary[h_name]['n']}, "
            f"mean={summary[h_name]['mean_km']:.1f} km, "
            f"median={summary[h_name]['median_km']:.1f} km, "
            f"p90={summary[h_name]['p90_km']:.1f} km"
        )

    if len(err_arr) > 0:
        for label, lo, hi in [
            ("LEO", 0, 2000),
            ("MEO", 2000, 20000),
            ("GEO", 20000, 50000),
        ]:
            mask = (
                (err_arr["days"] >= 0.5)
                & (err_arr["days"] <= 1.5)
                & (err_arr["alt"] >= lo)
                & (err_arr["alt"] < hi)
            )
            sub = err_arr[mask]
            if len(sub) > 0:
                print(
                    f"    {label} 1d: n={len(sub)}, "
                    f"median={np.median(sub['err_km']):.1f} km"
                )

    # ----------------------------------------------------------------
    print("\n[4/4] Saving...")
    pl.DataFrame(
        {
            "satellite_number": [e[0] for e in all_errors],
            "alt_km": [e[1] for e in all_errors],
            "size_range": [e[2] for e in all_errors],
            "days": [e[3] for e in all_errors],
            "err_km": [e[4] for e in all_errors],
        }
    ).write_csv(OUTPUT_CSV)

    summary["meta"] = {
        "n_satellites": n_sats,
        "n_epoch_comparisons": len(all_errors),
        "dt": DT,
        "propagator": "fixed_rk4_j2",
        "method": "segment_by_segment_capped_30d",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_json(summary, OUTPUT_JSON)

    print(f"  CSV: {len(all_errors)} rows")
    t_elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Phase 3b Complete | {t_elapsed:.0f}s | {len(all_errors)} pairs")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
