"""
Phase 2 - Comprehensive Feature Engineering (T2.1 through T2.7)
Builds 35-45 features across 7 groups: size, orbital, uncertainty, temporal,
environmental, risk, coordinate frame.

Usage: python scripts/phase2_feature_engineering.py
Output: data/vcm_ml_ready.parquet, models/scaler.pkl, models/feature_list.json
"""

import polars as pl
import numpy as np
import json
import time

from src.constants import GM, R_EARTH, CD
from src.io_utils import (
    get_data_dir,
    get_models_dir,
    get_reports_dir,
    save_parquet,
    save_pickle,
    save_json,
    report_file_size,
    timer,
)
from src.polars_utils import add_altitude_band, add_orbit_regime
from src.scaler import Scaler

DATA_DIR = get_data_dir()
MODELS_DIR = get_models_dir()
INPUT_FILE = DATA_DIR / "vcm_splits.parquet"
OUTPUT_FILE = DATA_DIR / "vcm_ml_ready.parquet"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
FEATURE_LIST_FILE = MODELS_DIR / "feature_list.json"


def compute_atmospheric_density(altitude_km, f107):
    """
    Simplified Harris-Priester atmospheric density model.
    Returns density in kg/m^3 for given altitude and solar flux.
    Uses exponential atmosphere approximation.
    """
    if altitude_km < 200:
        return 1.0e-10
    elif altitude_km < 300:
        rho0 = 2.0e-11
        H = 40.0
    elif altitude_km < 400:
        rho0 = 5.0e-12
        H = 50.0
    elif altitude_km < 500:
        rho0 = 2.0e-12
        H = 60.0
    elif altitude_km < 600:
        rho0 = 8.0e-13
        H = 80.0
    elif altitude_km < 800:
        rho0 = 3.0e-13
        H = 120.0
    elif altitude_km < 1000:
        rho0 = 1.0e-13
        H = 200.0
    else:
        rho0 = 5.0e-14
        H = 300.0

    f107_factor = 1.0 + 0.2 * (f107 - 100.0) / 100.0
    f107_factor = np.clip(f107_factor, 0.5, 2.0)

    h_ref = 200.0 if altitude_km < 200 else altitude_km
    h_ref = min(h_ref, altitude_km)
    rho = rho0 * np.exp(-(altitude_km - h_ref) / H) * f107_factor

    return rho


def main():
    t = timer()
    print("=" * 60)
    print("Phase 2 - Feature Engineering (T2.1 to T2.7)")
    print("=" * 60)

    # ── Load data ──
    print("\n[1/7] Loading VCM split data...")
    df = pl.read_parquet(INPUT_FILE)
    n_rows = len(df)
    print(f"  Loaded {n_rows:,} rows, {len(df.columns)} columns")

    # ── T2.1 - Size Proxy & Area-to-Mass Ratio ──
    print("\n[2/7] T2.1 - Size Proxy & AMR Features...")

    df = df.with_columns(
        [
            (pl.col("ballistic_coef_m2kg").abs() + 1e-10).alias("bc_abs"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("bc_abs").log10().alias("log_bc"),
        ]
    )
    df = df.with_columns(
        [
            (
                pl.col("ballistic_coef_m2kg").abs()
                / (pl.col("solar_rad_press_coeff_m2kg").abs() + 1e-10)
            ).alias("bc_to_srp_ratio"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("bc_abs") / CD).alias("area_to_mass_ratio_m2kg"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("area_to_mass_ratio_m2kg") + 1e-10).log10().alias("log_amr"),
        ]
    )

    print(f"  BC stats after abs: median={df['bc_abs'].median():.6f}")
    print(
        f"  AMR range: [{df['area_to_mass_ratio_m2kg'].min():.6f}, {df['area_to_mass_ratio_m2kg'].max():.6f}]"
    )
    print(f"  Log_BC range: [{df['log_bc'].min():.3f}, {df['log_bc'].max():.3f}]")

    # ── T2.2 - Full Orbital Regime Features ──
    print("\n[3/7] T2.2 - Orbital Regime Features...")

    if "altitude_band" not in df.columns:
        df = add_altitude_band(df, alt_col="perigee_alt_km")

    if "orbit_regime" not in df.columns:
        df = add_orbit_regime(df, alt_col="perigee_alt_km")

    df = df.with_columns(
        [
            ((pl.col("perigee_alt_km") + pl.col("apogee_alt_km")) / 2.0).alias(
                "mean_altitude_km"
            ),
        ]
    )

    print(f"  Altitude bands: {df['altitude_band'].n_unique()}")
    print(f"  Orbit regimes: {df['orbit_regime'].unique().to_list()}")

    # ── T2.3 - Expanded Temporal Features ──
    print("\n[4/7] T2.3 - Temporal Feature Engineering...")

    df = df.with_columns(
        [
            (
                pl.col("epoch_rev").diff().over("satellite_number")
                / (pl.col("time_gap_days") + 1e-10)
            ).alias("revolution_rate"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("revolution_rate").fill_null(0.0),
        ]
    )

    df = df.with_columns(
        [
            (
                (
                    pl.col("j2k_pos_x").diff().over(["satellite_number", "sequence_id"])
                    ** 2
                    + pl.col("j2k_pos_y")
                    .diff()
                    .over(["satellite_number", "sequence_id"])
                    ** 2
                    + pl.col("j2k_pos_z")
                    .diff()
                    .over(["satellite_number", "sequence_id"])
                    ** 2
                ).sqrt()
            ).alias("position_drift_km"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("position_drift_km").fill_null(0.0),
        ]
    )

    df = df.with_columns(
        [
            (
                pl.col("j2k_vel_x") ** 2
                + pl.col("j2k_vel_y") ** 2
                + pl.col("j2k_vel_z") ** 2
            )
            .sqrt()
            .alias("velocity_magnitude_kms"),
        ]
    )

    df = df.with_columns(
        [
            (
                pl.col("velocity_magnitude_kms")
                .diff()
                .over(["satellite_number", "sequence_id"])
                / (pl.col("time_gap_days") * 86400.0 + 1e-10)
            ).alias("acceleration_kms2"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("acceleration_kms2").fill_null(0.0),
        ]
    )

    df = df.with_columns(
        [
            pl.col("rtn_sigma_r_km")
            .diff()
            .over(["satellite_number", "sequence_id"])
            .alias("uncertainty_trend_r"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("uncertainty_trend_r").fill_null(0.0),
        ]
    )

    df = df.with_columns(
        [
            (
                pl.col("semi_major_axis_km")
                .diff()
                .over(["satellite_number", "sequence_id"])
                / (pl.col("time_gap_days") + 1e-10)
            ).alias("orbital_decay_trend_kmpday"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("orbital_decay_trend_kmpday").fill_null(0.0),
        ]
    )

    df = df.with_columns(
        [
            (
                pl.col("bc_abs").diff().over(["satellite_number", "sequence_id"])
                / (pl.col("time_gap_days") + 1e-10)
            ).alias("drag_trend_per_day"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("drag_trend_per_day").fill_null(0.0),
        ]
    )

    mean_v = df["velocity_magnitude_kms"].mean()
    print(f"  Mean velocity: {mean_v:.3f} km/s (LEO ~7.8)")

    # ── T2.4 - Atmospheric Density & Environmental Features ──
    print("\n[5/7] T2.4 - Atmospheric & Environmental Features...")

    alt = df["mean_altitude_km"].to_numpy()
    f107 = df["solar_flux_f10"].to_numpy()
    density = np.array([compute_atmospheric_density(a, f) for a, f in zip(alt, f107)])

    df = df.with_columns(
        [
            pl.Series("atmospheric_density_kgm3", density),
        ]
    )

    df = df.with_columns(
        [
            (
                pl.col("solar_rad_press_coeff_m2kg").abs()
                * pl.col("solar_flux_f10")
                / 1361.0
            ).alias("srp_acceleration_ms2"),
        ]
    )

    if "average_ap" in df.columns:
        df = df.with_columns(
            [
                ((pl.col("average_ap") + 1.0) / 10.0).alias("kp_index"),
            ]
        )
    else:
        df = df.with_columns(
            [
                pl.lit(1.1).alias("kp_index"),
            ]
        )

    print(
        f"  Atmospheric density range: [{density.min():.2e}, {density.max():.2e}] kg/m^3"
    )
    print(f"  Kp range: [{df['kp_index'].min():.2f}, {df['kp_index'].max():.2f}]")

    # ── T2.5 - Risk Proxy Features ──
    print("\n[6/7] T2.5 - Risk Proxy Features...")

    df = df.with_columns(
        [
            (
                pl.col("rtn_sigma_r_km")
                * pl.col("rtn_sigma_t_km")
                * pl.col("rtn_sigma_w_km")
            ).alias("uncertainty_volume_km3"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("uncertainty_volume_km3") + 1e-30)
            .log10()
            .alias("log_uncertainty_volume"),
        ]
    )
    df = df.with_columns(
        [
            (
                pl.col("uncertainty_volume_km3")
                / (pl.col("mean_altitude_km") ** 2 + 1.0)
            ).alias("collision_risk_proxy"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("event_source").fill_null("none").alias("debris_cloud_membership"),
        ]
    )

    print(
        f"  Uncertainty volume range: [{df['uncertainty_volume_km3'].min():.2e}, {df['uncertainty_volume_km3'].max():.2e}]"
    )

    # ── T2.6 - RTN/RSW Uncertainty Products ──
    print("\n[6b/7] T2.6 - RTN Uncertainty Products...")

    df = df.with_columns(
        [
            (
                pl.col("rtn_sigma_r_km")
                * pl.col("rtn_sigma_t_km")
                * pl.col("rtn_sigma_w_km")
            ).alias("rtn_covariance_volume_km3"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("rtn_sigma_t_km") / (pl.col("rtn_sigma_r_km") + 1e-15)).alias(
                "rtn_along_track_ratio"
            ),
            (pl.col("rtn_sigma_w_km") / (pl.col("rtn_sigma_r_km") + 1e-15)).alias(
                "rtn_cross_track_ratio"
            ),
        ]
    )
    df = df.with_columns(
        [
            (
                pl.col("rtn_sigma_r_km") ** 2
                + pl.col("rtn_sigma_t_km") ** 2
                + pl.col("rtn_sigma_w_km") ** 2
            )
            .sqrt()
            .alias("rtn_total_uncertainty_km"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("rtn_total_uncertainty_km") + 1e-15)
            .log10()
            .alias("log_rtn_uncertainty"),
        ]
    )

    median_along = df["rtn_along_track_ratio"].median()
    median_cross = df["rtn_cross_track_ratio"].median()
    print(f"  Median along-track/radial ratio: {median_along:.2f} (should be > 1)")
    print(f"  Median cross-track/radial ratio: {median_cross:.2f} (should be < 1)")

    # ── T2.7 - Feature Selection & Normalization ──
    print("\n[7/7] T2.7 - Feature Selection & Normalization...")

    size_features = [
        "bc_abs",
        "log_bc",
        "bc_to_srp_ratio",
        "area_to_mass_ratio_m2kg",
        "log_amr",
        "size_range",
    ]

    orbital_features = [
        "mean_altitude_km",
        "perigee_alt_km",
        "apogee_alt_km",
        "inclination_deg",
        "eccentricity",
        "semi_major_axis_km",
        "raan_deg",
        "arg_perigee_deg",
        "true_anomaly_deg",
        "orbital_period_min",
        "altitude_band",
        "orbit_regime",
    ]

    uncertainty_features = [
        "vector_u_sigma_km",
        "vector_v_sigma_km",
        "vector_w_sigma_km",
        "rtn_covariance_volume_km3",
        "rtn_along_track_ratio",
        "rtn_cross_track_ratio",
        "rtn_total_uncertainty_km",
        "log_rtn_uncertainty",
    ]

    temporal_features = [
        "epoch_delta_days",
        "time_gap_days",
        "revolution_rate",
        "position_drift_km",
        "velocity_magnitude_kms",
        "acceleration_kms2",
        "uncertainty_trend_r",
        "orbital_decay_trend_kmpday",
        "drag_trend_per_day",
    ]

    environmental_features = [
        "solar_flux_f10",
        "average_f10",
        "kp_index",
        "atmospheric_density_kgm3",
        "srp_acceleration_ms2",
    ]

    risk_features = [
        "uncertainty_volume_km3",
        "log_uncertainty_volume",
        "collision_risk_proxy",
        "object_class",
        "debris_cloud_membership",
    ]

    all_features = (
        size_features
        + orbital_features
        + uncertainty_features
        + temporal_features
        + environmental_features
        + risk_features
    )

    meta_cols = ["satellite_number", "sequence_id", "seq_pos", "split", "epoch_dt"]
    target_cols = [
        "j2k_pos_x",
        "j2k_pos_y",
        "j2k_pos_z",
        "rtn_sigma_r_km",
        "rtn_sigma_t_km",
        "rtn_sigma_w_km",
    ]
    physics_cols = [
        "j2k_pos_x",
        "j2k_pos_y",
        "j2k_pos_z",
        "j2k_vel_x",
        "j2k_vel_y",
        "j2k_vel_z",
    ]

    keep_cols = list(set(meta_cols + all_features + target_cols + physics_cols))
    keep_cols = [c for c in keep_cols if c in df.columns]

    print(f"  Selected {len(all_features)} features across 7 groups:")
    print(f"    Size: {len(size_features)}")
    print(f"    Orbital: {len(orbital_features)}")
    print(f"    Uncertainty: {len(uncertainty_features)}")
    print(f"    Temporal: {len(temporal_features)}")
    print(f"    Environmental: {len(environmental_features)}")
    print(f"    Risk: {len(risk_features)}")

    # Save feature list
    feature_groups = {
        "size": size_features,
        "orbital": orbital_features,
        "uncertainty": uncertainty_features,
        "temporal": temporal_features,
        "environmental": environmental_features,
        "risk": risk_features,
        "all_features": all_features,
        "targets": target_cols,
        "metadata": meta_cols,
    }
    save_json(feature_groups, FEATURE_LIST_FILE)
    print(f"  Feature list saved to {FEATURE_LIST_FILE}")

    df_ml = df.select(keep_cols)

    # Handle infinite values
    for col in df_ml.columns:
        if df_ml[col].dtype in [pl.Float64, pl.Float32]:
            inf_count = df_ml.filter(pl.col(col).is_infinite()).height
            if inf_count > 0:
                df_ml = df_ml.with_columns(
                    [
                        pl.when(pl.col(col).is_infinite())
                        .then(None)
                        .otherwise(pl.col(col))
                        .alias(col)
                    ]
                )

    # Create size-specific subsets
    print("\n  Creating size-specific subsets...")
    for size_label in ["25-50cm", "10-25cm", "5-10cm"]:
        size_df = df_ml.filter(pl.col("size_range") == size_label)
        size_rows = len(size_df)
        size_pct = size_rows / n_rows * 100
        print(f"    {size_label}: {size_rows:,} ({size_pct:.1f}%)")

        size_out = (
            DATA_DIR
            / f"vcm_ml_ready_{size_label.replace('-', '_').replace('cm', 'cm')}.parquet"
        )
        save_parquet(size_df, size_out)
        print(f"      Saved to {size_out}")

    # Save the full ML-ready dataset
    print("\n  Saving full ML dataset...")
    save_parquet(df_ml, OUTPUT_FILE)

    # Create and save scaler using src.scaler
    print("  Creating feature scaler...")
    numeric_cols = [
        c for c in all_features if df_ml[c].dtype in [pl.Float64, pl.Float32, pl.Int64]
    ]

    train_data = df_ml.filter(pl.col("split") == "train")
    scaler = Scaler()
    scaler.fit(train_data, numeric_cols)
    scaler.save(SCALER_FILE)
    print(f"  Scaler saved to {SCALER_FILE} ({len(scaler.params)} features)")

    fsize = report_file_size(OUTPUT_FILE)

    print(f"\n{'=' * 60}")
    print("Phase 2 Complete!")
    print(f"  Output: {OUTPUT_FILE} ({fsize:.1f} MB)")
    print(
        f"  Features: {len(all_features)} ({len(numeric_cols)} numerical + categoricals)"
    )
    print(f"  Columns kept: {len(keep_cols)}")
    print(f"  Time: {t.elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
