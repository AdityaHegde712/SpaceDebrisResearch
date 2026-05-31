"""
Phase 1 VCM Pipeline — sequential stages run in-memory.
Replaces: t1.2, t1.3, t1.4, t1.5, t1.6, t1.8.

Usage:
    python scripts/pipeline_phase1.py
    python scripts/pipeline_phase1.py --skip-to 3         # start at sequences
    python scripts/pipeline_phase1.py --skip-to sequences
    python scripts/pipeline_phase1.py --save-intermediates
"""

import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from src.constants import (
    GM,
    R_EARTH,
    SEQ_LEN,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    MIN_EPOCHS,
    SIZE_25CM_BC,
    SIZE_10CM_BC,
    SIZE_5CM_BC,
)
from src.io_utils import get_data_dir, save_parquet, report_file_size, timer
from src.polars_utils import add_altitude_band, compute_time_gap_days

DATA_DIR = get_data_dir()
INPUT_FILE = DATA_DIR / "vcm_output_51_folders.parquet"
CELESTRAK_DIR = Path("celestrak_data")

STAGE_NAMES = ["rtn", "elements", "celestrak", "sequences", "split", "quality"]


def banner(title: str, t: timer):
    print("=" * 60)
    print(title)
    print("=" * 60)
    return t


# ---------------------------------------------------------------------------
# Stage 0 — RTN coordinate derivation
# ---------------------------------------------------------------------------


def compute_rtn_basis(r_x, r_y, r_z, v_x, v_y, v_z):
    r = np.column_stack([r_x, r_y, r_z])
    v = np.column_stack([v_x, v_y, v_z])

    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    r_norm = np.maximum(r_norm, 1e-15)
    R_hat = r / r_norm

    h = np.cross(r, v)
    h_norm = np.linalg.norm(h, axis=1, keepdims=True)
    h_norm = np.maximum(h_norm, 1e-15)
    W_hat = h / h_norm

    T_hat = np.cross(W_hat, R_hat)
    T_norm = np.linalg.norm(T_hat, axis=1, keepdims=True)
    T_norm = np.maximum(T_norm, 1e-15)
    T_hat = T_hat / T_norm

    return (
        R_hat[:, 0],
        R_hat[:, 1],
        R_hat[:, 2],
        T_hat[:, 0],
        T_hat[:, 1],
        T_hat[:, 2],
        W_hat[:, 0],
        W_hat[:, 1],
        W_hat[:, 2],
    )


def project_vectors(r_x, r_y, r_z, basis_x, basis_y, basis_z):
    return r_x * basis_x + r_y * basis_y + r_z * basis_z


def validate_rtn(df, sample_size=10000):
    np.random.seed(42)
    idx = np.random.choice(len(df), min(sample_size, len(df)), replace=False)

    r_x = df["j2k_pos_x"].to_numpy()[idx]
    r_y = df["j2k_pos_y"].to_numpy()[idx]
    r_z = df["j2k_pos_z"].to_numpy()[idx]
    v_x = df["j2k_vel_x"].to_numpy()[idx]
    v_y = df["j2k_vel_y"].to_numpy()[idx]
    v_z = df["j2k_vel_z"].to_numpy()[idx]

    r = np.column_stack([r_x, r_y, r_z])
    v = np.column_stack([v_x, v_y, v_z])

    R_hat = np.column_stack(
        [
            df["rtn_Rx"].to_numpy()[idx],
            df["rtn_Ry"].to_numpy()[idx],
            df["rtn_Rz"].to_numpy()[idx],
        ]
    )
    T_hat = np.column_stack(
        [
            df["rtn_Tx"].to_numpy()[idx],
            df["rtn_Ty"].to_numpy()[idx],
            df["rtn_Tz"].to_numpy()[idx],
        ]
    )
    W_hat = np.column_stack(
        [
            df["rtn_Wx"].to_numpy()[idx],
            df["rtn_Wy"].to_numpy()[idx],
            df["rtn_Wz"].to_numpy()[idx],
        ]
    )

    rt_dot = np.abs(np.sum(R_hat * T_hat, axis=1))
    rw_dot = np.abs(np.sum(R_hat * W_hat, axis=1))
    tw_dot = np.abs(np.sum(T_hat * W_hat, axis=1))

    print(f"  Max |R*T|: {rt_dot.max():.2e}")
    print(f"  Max |R*W|: {rw_dot.max():.2e}")
    print(f"  Max |T*W|: {tw_dot.max():.2e}")

    R_norm = np.abs(1.0 - np.linalg.norm(R_hat, axis=1))
    T_norm = np.abs(1.0 - np.linalg.norm(T_hat, axis=1))
    W_norm = np.abs(1.0 - np.linalg.norm(W_hat, axis=1))

    print(f"  Max |1-|R||: {R_norm.max():.2e}")
    print(f"  Max |1-|T||: {T_norm.max():.2e}")
    print(f"  Max |1-|W||: {W_norm.max():.2e}")

    r_j2k_norm = np.linalg.norm(r, axis=1)
    r_rtn = np.column_stack(
        [
            df["rtn_pos_r_km"].to_numpy()[idx],
            df["rtn_pos_t_km"].to_numpy()[idx],
            df["rtn_pos_w_km"].to_numpy()[idx],
        ]
    )
    r_rtn_norm = np.linalg.norm(r_rtn, axis=1)
    rel_err = np.abs(r_j2k_norm - r_rtn_norm) / np.maximum(r_j2k_norm, 1e-15)
    print(f"  Max relative magnitude error: {rel_err.max():.2e}")

    det_vals = np.array(
        [
            np.linalg.det(np.column_stack([R_hat[i], T_hat[i], W_hat[i]]))
            for i in range(len(idx))
        ]
    )
    det_err = np.abs(1.0 - det_vals)
    print(f"  Max |1-det|: {det_err.max():.2e}")

    u_sigma = df["vector_u_sigma_km"].to_numpy()[idx]
    rtn_sig_r = df["rtn_sigma_r_km"].to_numpy()[idx]
    sigma_diff = np.abs(u_sigma - rtn_sig_r).max()
    print(f"  Max sigma mismatch: {sigma_diff:.2e}")
    print(f"\n  Validation sample size: {len(idx)} rows")

    return True


def stage_rtn(df):
    t = timer()
    print(f"\n[Stage 0 — RTN] Loading {len(df):,} rows, {len(df.columns)} columns")

    j2k_pos_x = df["j2k_pos_x"].to_numpy()
    j2k_pos_y = df["j2k_pos_y"].to_numpy()
    j2k_pos_z = df["j2k_pos_z"].to_numpy()
    j2k_vel_x = df["j2k_vel_x"].to_numpy()
    j2k_vel_y = df["j2k_vel_y"].to_numpy()
    j2k_vel_z = df["j2k_vel_z"].to_numpy()

    (Rx, Ry, Rz, Tx, Ty, Tz, Wx, Wy, Wz) = compute_rtn_basis(
        j2k_pos_x, j2k_pos_y, j2k_pos_z, j2k_vel_x, j2k_vel_y, j2k_vel_z
    )

    rtn_pos_r = project_vectors(j2k_pos_x, j2k_pos_y, j2k_pos_z, Rx, Ry, Rz)
    rtn_pos_t = project_vectors(j2k_pos_x, j2k_pos_y, j2k_pos_z, Tx, Ty, Tz)
    rtn_pos_w = project_vectors(j2k_pos_x, j2k_pos_y, j2k_pos_z, Wx, Wy, Wz)

    rtn_vel_r = project_vectors(j2k_vel_x, j2k_vel_y, j2k_vel_z, Rx, Ry, Rz)
    rtn_vel_t = project_vectors(j2k_vel_x, j2k_vel_y, j2k_vel_z, Tx, Ty, Tz)
    rtn_vel_w = project_vectors(j2k_vel_x, j2k_vel_y, j2k_vel_z, Wx, Wy, Wz)

    rtn_sigma_r = df["vector_u_sigma_km"].to_numpy()
    rtn_sigma_t = df["vector_v_sigma_km"].to_numpy()
    rtn_sigma_w = df["vector_w_sigma_km"].to_numpy()
    rtn_sig_vel_r = df["vector_ud_sigma_kms"].to_numpy()
    rtn_sig_vel_t = df["vector_vd_sigma_kms"].to_numpy()
    rtn_sig_vel_w = df["vector_wd_sigma_kms"].to_numpy()

    rtn_columns = {
        "rtn_Rx": Rx,
        "rtn_Ry": Ry,
        "rtn_Rz": Rz,
        "rtn_Tx": Tx,
        "rtn_Ty": Ty,
        "rtn_Tz": Tz,
        "rtn_Wx": Wx,
        "rtn_Wy": Wy,
        "rtn_Wz": Wz,
        "rtn_pos_r_km": rtn_pos_r,
        "rtn_pos_t_km": rtn_pos_t,
        "rtn_pos_w_km": rtn_pos_w,
        "rtn_vel_r_kms": rtn_vel_r,
        "rtn_vel_t_kms": rtn_vel_t,
        "rtn_vel_w_kms": rtn_vel_w,
        "rtn_sigma_r_km": rtn_sigma_r,
        "rtn_sigma_t_km": rtn_sigma_t,
        "rtn_sigma_w_km": rtn_sigma_w,
        "rtn_sig_vel_r_kms": rtn_sig_vel_r,
        "rtn_sig_vel_t_kms": rtn_sig_vel_t,
        "rtn_sig_vel_w_kms": rtn_sig_vel_w,
    }

    df_with_rtn = df.with_columns(
        [pl.Series(name, vals) for name, vals in rtn_columns.items()]
    )
    print(f"  Added {len(rtn_columns)} RTN columns (total {len(df_with_rtn.columns)})")

    print("\n  Validation:")
    validate_rtn(df_with_rtn)
    print(f"\n  Stage 0 complete ({t.elapsed:.1f}s)")
    return df_with_rtn


# ---------------------------------------------------------------------------
# Stage 1 — Orbital elements
# ---------------------------------------------------------------------------


def compute_orbital_elements(
    j2k_pos_x, j2k_pos_y, j2k_pos_z, j2k_vel_x, j2k_vel_y, j2k_vel_z
):
    r = np.column_stack([j2k_pos_x, j2k_pos_y, j2k_pos_z])
    v = np.column_stack([j2k_vel_x, j2k_vel_y, j2k_vel_z])

    r_norm = np.linalg.norm(r, axis=1)
    v_norm = np.linalg.norm(v, axis=1)

    h = np.cross(r, v)
    h_norm = np.linalg.norm(h, axis=1)

    r_norm_safe = np.maximum(r_norm, 1e-15)
    h_norm_safe = np.maximum(h_norm, 1e-15)

    specific_energy = 0.5 * v_norm**2 - GM / r_norm_safe

    semi_major_axis = np.where(
        specific_energy < 0, -GM / (2.0 * np.maximum(specific_energy, -1e15)), np.nan
    )

    v_cross_h = np.cross(v, h)
    r_hat = r / r_norm_safe[:, np.newaxis]
    e_vec = v_cross_h / GM - r_hat
    eccentricity = np.linalg.norm(e_vec, axis=1)
    eccentricity = np.clip(eccentricity, 0, 0.999)

    z_hat = np.array([0.0, 0.0, 1.0])
    n = np.cross(z_hat, h)
    n_norm = np.linalg.norm(n, axis=1)
    n_norm_safe = np.maximum(n_norm, 1e-15)

    cos_i = h[:, 2] / h_norm_safe
    cos_i = np.clip(cos_i, -1.0, 1.0)
    inclination = np.arccos(cos_i)

    raan = np.arctan2(n[:, 1], n[:, 0])
    raan = np.where(n_norm < 1e-10, 0.0, raan)

    n_dot_e = np.sum(n * e_vec, axis=1)
    cos_omega = n_dot_e / (n_norm_safe * np.maximum(eccentricity, 1e-15))
    cos_omega = np.clip(cos_omega, -1.0, 1.0)
    omega = np.arccos(cos_omega)
    omega = np.where(e_vec[:, 2] < 0, 2.0 * np.pi - omega, omega)

    e_dot_r = np.sum(e_vec * r, axis=1)
    cos_nu = e_dot_r / (np.maximum(eccentricity, 1e-15) * r_norm_safe)
    cos_nu = np.clip(cos_nu, -1.0, 1.0)
    r_dot_v = np.sum(r * v, axis=1)
    nu = np.arccos(cos_nu)
    nu = np.where(r_dot_v > 0, nu, 2.0 * np.pi - nu)

    perigee_alt = semi_major_axis * (1.0 - eccentricity) - R_EARTH
    apogee_alt = semi_major_axis * (1.0 + eccentricity) - R_EARTH

    orbital_period = np.where(
        semi_major_axis > 0,
        2.0 * np.pi * np.sqrt(semi_major_axis**3 / GM) / 60.0,
        np.nan,
    )

    inclination_deg = np.degrees(inclination)
    raan_deg = np.degrees(raan) % 360.0
    arg_perigee_deg = np.degrees(omega) % 360.0
    true_anomaly_deg = np.degrees(nu) % 360.0

    return {
        "semi_major_axis_km": semi_major_axis,
        "eccentricity": eccentricity,
        "inclination_deg": inclination_deg,
        "raan_deg": raan_deg,
        "arg_perigee_deg": arg_perigee_deg,
        "true_anomaly_deg": true_anomaly_deg,
        "perigee_alt_km": perigee_alt,
        "apogee_alt_km": apogee_alt,
        "orbital_period_min": orbital_period,
    }


def validate_elements(df, tol_fraction=0.01):
    print("\n  Validation checks:")
    samp = df.sample(n=min(10000, len(df)), seed=42)

    j2k_v = np.column_stack(
        [
            samp["j2k_vel_x"].to_numpy(),
            samp["j2k_vel_y"].to_numpy(),
            samp["j2k_vel_z"].to_numpy(),
        ]
    )
    v_sq = np.sum(j2k_v**2, axis=1)

    j2k_r = np.column_stack(
        [
            samp["j2k_pos_x"].to_numpy(),
            samp["j2k_pos_y"].to_numpy(),
            samp["j2k_pos_z"].to_numpy(),
        ]
    )
    r_mag = np.linalg.norm(j2k_r, axis=1)
    a = samp["semi_major_axis_km"].to_numpy()
    r_safe = np.maximum(r_mag, 1e-15)
    a_safe = np.maximum(a, 1e-15)

    v_sq_predicted = GM * (2.0 / r_safe - 1.0 / a_safe)
    valid = a > 0
    if valid.sum() > 0:
        vis_viva_err = np.abs(v_sq[valid] - v_sq_predicted[valid]) / np.maximum(
            v_sq[valid], 1e-15
        )
        print(f"  Vis-viva max relative error: {vis_viva_err.max():.4e}")
    else:
        print("  Vis-viva: No bound orbits found!")

    ecc = samp["eccentricity"].to_numpy()
    print(f"  Eccentricity range: [{ecc.min():.6f}, {ecc.max():.6f}]")
    ecc_invalid = (ecc < 0) | (ecc >= 1)
    print(f"  Eccentricity out of [0,1): {ecc_invalid.sum()} / {len(ecc)}")

    inc = samp["inclination_deg"].to_numpy()
    inc_invalid = (inc < 0) | (inc > 180)
    print(f"  Inclination range: [{inc.min():.2f}, {inc.max():.2f}] deg")
    print(f"  Inclination out of [0,180]: {inc_invalid.sum()} / {len(inc_invalid)}")

    raan = samp["raan_deg"].to_numpy()
    aop = samp["arg_perigee_deg"].to_numpy()
    ta = samp["true_anomaly_deg"].to_numpy()
    print(f"  RAAN range: [{raan.min():.2f}, {raan.max():.2f}]")
    print(f"  Arg Perigee range: [{aop.min():.2f}, {aop.max():.2f}]")
    print(f"  True Anomaly range: [{ta.min():.2f}, {ta.max():.2f}]")

    alt = r_mag - R_EARTH
    perigee = samp["perigee_alt_km"].to_numpy()
    apogee = samp["apogee_alt_km"].to_numpy()
    perigee_valid = np.sum(perigee[valid] <= alt[valid] + 100) / max(valid.sum(), 1)
    print(f"  Perigee <= Altitude (with margin): {perigee_valid:.1%}")

    period = samp["orbital_period_min"].to_numpy()
    valid_p = (period > 80) & (period < 1500)
    print(
        f"  Orbital period range: [{period[valid].min():.1f}, {period[valid].max():.1f}] min"
    )
    print(f"  Period in [80, 1500] min: {valid_p.sum()} / {valid.sum()} bound orbits")


def stage_elements(df):
    t = timer()
    print(
        f"\n[Stage 1 — Orbital Elements] Input: {len(df):,} rows, {len(df.columns)} columns"
    )

    elements = compute_orbital_elements(
        df["j2k_pos_x"].to_numpy(),
        df["j2k_pos_y"].to_numpy(),
        df["j2k_pos_z"].to_numpy(),
        df["j2k_vel_x"].to_numpy(),
        df["j2k_vel_y"].to_numpy(),
        df["j2k_vel_z"].to_numpy(),
    )

    for name, arr in elements.items():
        nan_count = np.isnan(arr).sum()
        print(f"  {name}: shape={arr.shape}, NaNs={nan_count}")

    df_with_elements = df.with_columns(
        [pl.Series(name, vals) for name, vals in elements.items()]
    )
    print(
        f"  Added {len(elements)} element columns (total {len(df_with_elements.columns)})"
    )

    print("\n  Validation:")
    validate_elements(df_with_elements)
    print(f"\n  Stage 1 complete ({t.elapsed:.1f}s)")
    return df_with_elements


# ---------------------------------------------------------------------------
# Stage 2 — Celestrak integration
# ---------------------------------------------------------------------------


def classify_object(name_str):
    if name_str is None or (isinstance(name_str, float) and np.isnan(name_str)):
        return "unknown"

    name = str(name_str).upper().strip()

    if " DEB" in name or name.endswith("DEB") or "DEB " in name:
        return "fragmentation_debris"
    if " R/B" in name or name.endswith("RB") or name.startswith("R/B"):
        return "rocket_body"
    if " PL" in name or name.endswith("PL") or "PAYLOAD" in name:
        return "payload"
    if any(p in name for p in ["COSMOS", "IRIDIUM", "FENGYUN"]):
        return "payload"

    return "unknown"


def compute_altitudes(mean_motion_rev_day, eccentricity):
    n = mean_motion_rev_day * 2.0 * np.pi / 86400.0
    a = (GM / n**2) ** (1.0 / 3.0)
    perigee = a * (1.0 - eccentricity) - R_EARTH
    apogee = a * (1.0 + eccentricity) - R_EARTH
    mean_alt = a - R_EARTH
    return perigee, apogee, mean_alt


def stage_celestrak(df):
    t = timer()
    print(f"\n[Stage 2 — Celestrak Integration] Input: {len(df):,} rows")

    print("  Loading Celestrak CSVs...")
    events = {}
    for event_name in ["cosmos-2251-debris", "fengyun-1c-debris", "iridium-33-debris"]:
        f = CELESTRAK_DIR / f"{event_name}.csv"
        df_event = pl.read_csv(f)
        events[event_name] = df_event
        print(f"    {event_name}: {len(df_event):,} objects")

    celestrak_raw = pl.concat(
        [
            df.with_columns(pl.lit(name).alias("event_source"))
            for name, df in events.items()
        ],
        how="vertical",
    )
    print(f"    Total Celestrak objects: {len(celestrak_raw):,}")

    print("  Classifying objects...")
    celestrak_classes = celestrak_raw.with_columns(
        [
            pl.col("OBJECT_NAME")
            .map_elements(classify_object, return_dtype=pl.String)
            .alias("object_class_celestrak"),
        ]
    )

    mean_motion = celestrak_classes["MEAN_MOTION"].to_numpy()
    ecc = celestrak_classes["ECCENTRICITY"].to_numpy()
    perigee, apogee, mean_alt = compute_altitudes(mean_motion, ecc)

    celestrak_enriched = celestrak_classes.with_columns(
        [
            pl.Series("perigee_alt_km", perigee),
            pl.Series("apogee_alt_km", apogee),
            pl.Series("mean_alt_km", mean_alt),
        ]
    )
    celestrak_enriched = add_altitude_band(celestrak_enriched, alt_col="mean_alt_km")
    print(f"    Altitude range: {mean_alt.min():.0f} - {mean_alt.max():.0f} km")

    print("  Merging with VCM data...")
    vcm_df = df.select(["satellite_number", "int_des", "common_name"])

    celestrak_merge = celestrak_enriched.select(
        [
            pl.col("NORAD_CAT_ID").alias("satellite_number"),
            pl.col("OBJECT_ID").alias("celestrak_int_des"),
            pl.col("event_source"),
            pl.col("object_class_celestrak"),
            pl.col("perigee_alt_km").alias("celestrak_perigee_km"),
            pl.col("apogee_alt_km").alias("celestrak_apogee_km"),
            pl.col("mean_alt_km").alias("celestrak_mean_alt_km"),
        ]
    ).unique(subset=["satellite_number"])

    merged = vcm_df.join(celestrak_merge, on="satellite_number", how="left")

    matched_sats = (
        merged.filter(pl.col("event_source").is_not_null())
        .select(pl.col("satellite_number").n_unique())
        .item()
    )
    total_vcm_sats = vcm_df.select(pl.col("satellite_number").n_unique()).item()
    print(
        f"    VCM satellites: {total_vcm_sats:,}, matched on NORAD ID: {matched_sats:,}"
    )

    merged = merged.with_columns(
        [
            pl.col("object_class_celestrak")
            .fill_null(
                pl.col("common_name").map_elements(
                    classify_object, return_dtype=pl.String
                )
            )
            .alias("object_class"),
        ]
    )

    labels = merged.select(["satellite_number", "object_class", "event_source"])
    vcm_labeled = df.join(labels, on="satellite_number", how="left")

    print(f"  Stage 2 complete ({t.elapsed:.1f}s)")
    return vcm_labeled


# ---------------------------------------------------------------------------
# Stage 3 — Temporal sequences
# ---------------------------------------------------------------------------


def stage_sequences(df):
    t = timer()
    print("\n[Stage 3 — Temporal Sequences]")
    print(f"  Sequence length: {SEQ_LEN}, Min epochs: {MIN_EPOCHS}")
    print(f"  Input: {len(df):,} rows, {len(df.columns)} columns")

    df = df.with_columns(
        [
            pl.col("epoch_time_utc")
            .str.replace(r"\s*\([^)]*\)\s*", " ")
            .alias("epoch_clean")
        ]
    )
    df = df.with_columns(
        [
            pl.col("epoch_clean")
            .str.strptime(pl.Datetime, "%Y %j %H:%M:%S.%3f")
            .alias("epoch_dt")
        ]
    )

    null_epochs = df.filter(pl.col("epoch_dt").is_null()).height
    print(f"  Null epochs: {null_epochs} / {len(df)}")

    df = df.sort(["satellite_number", "epoch_dt"])

    epoch_counts = df.group_by("satellite_number").agg(pl.len().alias("epoch_count"))
    valid_sats = epoch_counts.filter(pl.col("epoch_count") >= MIN_EPOCHS)
    print(
        f"  Satellites with >= {MIN_EPOCHS} epochs: {len(valid_sats)} / {len(epoch_counts)}"
    )

    df = df.join(valid_sats.select("satellite_number"), on="satellite_number")

    df = df.with_columns(
        [
            pl.int_range(0, pl.len()).over("satellite_number").alias("epoch_idx"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("epoch_idx") // SEQ_LEN).alias("sequence_id"),
        ]
    )
    df = df.with_columns(
        [
            pl.len().over(["satellite_number", "sequence_id"]).alias("seq_len"),
        ]
    )

    seq_count = df.select(
        pl.struct(["satellite_number", "sequence_id"]).n_unique()
    ).item()
    print(f"  Total sequences (non-overlapping, len={SEQ_LEN}): {seq_count:,}")

    df = df.filter(pl.col("seq_len") == SEQ_LEN)

    df = df.sort(["satellite_number", "epoch_dt"])
    df = compute_time_gap_days(df)
    df = df.sort(["satellite_number", "sequence_id", "epoch_idx"])

    df = df.with_columns(
        [
            (pl.col("epoch_idx") - pl.col("sequence_id") * SEQ_LEN)
            .cast(pl.Int32)
            .alias("seq_pos"),
        ]
    )

    gap_mean = df.select(pl.col("time_gap_days").mean()).item()
    gap_median = df.select(pl.col("time_gap_days").median()).item()
    print(f"  Mean time gap: {gap_mean:.2f} days ({gap_mean * 24:.1f} hours)")
    print(f"  Median time gap: {gap_median:.2f} days ({gap_median * 24:.1f} hours)")

    df = df.drop(["epoch_clean"])

    print(
        f"  Stage 3 complete: {len(df):,} rows, {seq_count:,} sequences ({t.elapsed:.1f}s)"
    )
    return df


# ---------------------------------------------------------------------------
# Stage 4 — Train/val/test split
# ---------------------------------------------------------------------------


def assign_size_range(bc):
    if bc <= SIZE_25CM_BC:
        return "25-50cm"
    elif bc <= SIZE_10CM_BC:
        return "10-25cm"
    else:
        return "5-10cm"


def stage_split(df):
    t = timer()
    print("\n[Stage 4 — Train/Val/Test Split]")
    print(f"  Ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    print(f"  Input: {len(df):,} rows")

    seq_info = df.group_by(["satellite_number", "sequence_id"]).agg(
        [
            pl.col("epoch_dt").first().alias("seq_first_epoch"),
            pl.col("ballistic_coef_m2kg").first().alias("seq_bc"),
        ]
    )
    n_seq = len(seq_info)
    print(f"  Unique (satellite, sequence) pairs: {n_seq:,}")

    seq_info = seq_info.with_columns(
        [
            pl.col("seq_bc")
            .map_elements(assign_size_range, return_dtype=pl.String)
            .alias("size_range"),
        ]
    )
    seq_info = seq_info.sort(["satellite_number", "seq_first_epoch"])

    sat_time = (
        seq_info.group_by("satellite_number")
        .agg(
            [
                pl.col("seq_first_epoch").median().alias("sat_median_epoch"),
            ]
        )
        .sort("sat_median_epoch")
    )

    n_sats = len(sat_time)
    sat_time = sat_time.with_columns(
        [
            pl.int_range(0, pl.len()).alias("sat_rank"),
            pl.lit(n_sats).alias("sat_count"),
        ]
    )
    sat_time = sat_time.with_columns(
        [
            (pl.col("sat_rank") / pl.col("sat_count").cast(pl.Float64)).alias(
                "sat_progress"
            ),
        ]
    )
    sat_time = sat_time.with_columns(
        [
            pl.when(pl.col("sat_progress") < TRAIN_RATIO)
            .then(pl.lit("train"))
            .when(pl.col("sat_progress") < TRAIN_RATIO + VAL_RATIO)
            .then(pl.lit("val"))
            .otherwise(pl.lit("test"))
            .alias("sat_split"),
        ]
    )

    seq_info = seq_info.join(
        sat_time.select(["satellite_number", "sat_split"]),
        on="satellite_number",
        how="inner",
    ).rename({"sat_split": "split"})

    sat_split_count = seq_info.group_by("satellite_number").agg(
        pl.col("split").n_unique().alias("split_count")
    )
    multi_split = sat_split_count.filter(pl.col("split_count") > 1).height
    print(f"  Satellites in multiple splits: {multi_split} (should be 0)")

    splits_to_add = seq_info.select(
        ["satellite_number", "sequence_id", "split", "size_range"]
    )
    df_splits = df.join(
        splits_to_add, on=["satellite_number", "sequence_id"], how="inner"
    )

    split_dist = (
        df_splits.group_by("split")
        .agg(
            pl.len().alias("rows"),
            pl.col("satellite_number").n_unique().alias("satellites"),
            pl.col("sequence_id").n_unique().alias("sequences"),
        )
        .sort("split")
    )

    print("\n  Split distribution:")
    for row in split_dist.iter_rows():
        print(f"    {row[0]}: {row[1]:,} rows, {row[2]:,} sats, {row[3]:,} seqs")

    print("\n  Distribution consistency:")
    splits_arr = seq_info.select(
        ["satellite_number", "sequence_id", "split", "seq_bc"]
    ).unique()
    train_bc = splits_arr.filter(pl.col("split") == "train")["seq_bc"].to_numpy()
    val_bc = splits_arr.filter(pl.col("split") == "val")["seq_bc"].to_numpy()
    test_bc = splits_arr.filter(pl.col("split") == "test")["seq_bc"].to_numpy()

    if len(val_bc) > 0 and len(test_bc) > 0:
        ks_train_val = scipy_stats.ks_2samp(train_bc, val_bc)
        ks_train_test = scipy_stats.ks_2samp(train_bc, test_bc)
        ks_val_test = scipy_stats.ks_2samp(val_bc, test_bc)
        print(
            f"    KS test (train vs val):  stat={ks_train_val.statistic:.4f}, p={ks_train_val.pvalue:.4f}"
        )
        print(
            f"    KS test (train vs test): stat={ks_train_test.statistic:.4f}, p={ks_train_test.pvalue:.4f}"
        )
        print(
            f"    KS test (val vs test):   stat={ks_val_test.statistic:.4f}, p={ks_val_test.pvalue:.4f}"
        )

    print(f"  Stage 4 complete ({t.elapsed:.1f}s)")
    return df_splits


# ---------------------------------------------------------------------------
# Stage 5 — Quality report
# ---------------------------------------------------------------------------


def stage_quality(df):
    t = timer()
    OUTPUT_FILE = DATA_DIR / "quality_report.html"
    n_rows, n_cols = df.shape

    print(f"\n[Stage 5 — Quality Report] {n_rows:,} rows, {n_cols} columns")

    null_counts = df.select([pl.col(c).is_null().sum().alias(c) for c in df.columns])
    null_cols = {
        c: null_counts[c].item()
        for c in null_counts.columns
        if null_counts[c].item() > 0
    }
    print(
        f"  Nulls: {len(null_cols)} columns with nulls"
        if null_cols
        else "  No null values"
    )

    bc = df["ballistic_coef_m2kg"]
    bc_min, bc_max, bc_mean, bc_median = bc.min(), bc.max(), bc.mean(), bc.median()
    print(f"  BC range: [{bc_min:.6f}, {bc_max:.6f}]")

    srp = df["solar_rad_press_coeff_m2kg"]
    bc_srp_corr = df.select(
        pl.corr("ballistic_coef_m2kg", "solar_rad_press_coeff_m2kg")
    ).item()
    print(f"  BC-SRP correlation: {bc_srp_corr:.4f}")

    size_dist = (
        df.group_by("size_range")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    size_total = size_dist["count"].sum()
    print("  Size ranges:")
    for row in size_dist.iter_rows():
        print(f"    {row[0]}: {row[1]:,} ({row[1] / size_total * 100:.1f}%)")

    sample = df.sample(n=10000, seed=42)
    R = np.column_stack(
        [
            sample["rtn_Rx"].to_numpy(),
            sample["rtn_Ry"].to_numpy(),
            sample["rtn_Rz"].to_numpy(),
        ]
    )
    T = np.column_stack(
        [
            sample["rtn_Tx"].to_numpy(),
            sample["rtn_Ty"].to_numpy(),
            sample["rtn_Tz"].to_numpy(),
        ]
    )
    W = np.column_stack(
        [
            sample["rtn_Wx"].to_numpy(),
            sample["rtn_Wy"].to_numpy(),
            sample["rtn_Wz"].to_numpy(),
        ]
    )

    rt_dot = np.abs(np.sum(R * T, axis=1)).max()
    rw_dot = np.abs(np.sum(R * W, axis=1)).max()
    tw_dot = np.abs(np.sum(T * W, axis=1)).max()
    det_val = np.abs(np.linalg.det(np.column_stack([R[0], T[0], W[0]])))
    rtn_ok = (
        rt_dot < 1e-14
        and rw_dot < 1e-14
        and tw_dot < 1e-14
        and abs(det_val - 1.0) < 1e-14
    )
    print(f"  RTN: {'PASS' if rtn_ok else 'FAIL'}")

    if "celestrak_mean_alt_km" in df.columns:
        cel_alt = df.filter(pl.col("celestrak_mean_alt_km").is_not_null())[
            "celestrak_mean_alt_km"
        ]
        if len(cel_alt) > 0:
            vcm_perigee = df.filter(pl.col("celestrak_mean_alt_km").is_not_null())[
                "perigee_alt_km"
            ]
            alt_corr = np.corrcoef(vcm_perigee.to_numpy(), cel_alt.to_numpy())[0, 1]
            print(f"  VCM vs Celestrak altitude correlation: {alt_corr:.4f}")

    key_cols = [
        "ballistic_coef_m2kg",
        "vector_u_sigma_km",
        "vector_v_sigma_km",
        "vector_w_sigma_km",
        "eccentricity",
        "rtn_sigma_r_km",
    ]
    col_stats = {}
    for col in key_cols:
        if col in df.columns:
            vals = df[col]
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
            n_outliers = df.filter((pl.col(col) < lower) | (pl.col(col) > upper)).height
            col_stats[col] = {
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "outliers": n_outliers,
                "pct": n_outliers / n_rows * 100,
            }

    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>VCM Data Quality Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 960px; margin: 20px auto; padding: 0 20px; background: #f5f5f5; }}
        h1, h2, h3 {{ color: #1a1a2e; }}
        .card {{ background: white; border-radius: 8px; padding: 16px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .pass {{ color: #2e7d32; font-weight: bold; }}
        .fail {{ color: #c62828; font-weight: bold; }}
        .warn {{ color: #f57f17; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ text-align: left; padding: 6px 12px; border-bottom: 1px solid #ddd; }}
        th {{ background: #1a1a2e; color: white; }}
    </style>
</head>
<body>
    <h1>VCM Data Quality Report</h1>
    <p>Generated: 2026-05-30 | Rows: {n_rows:,} | Columns: {n_cols}</p>

    <div class="card">
        <h2>1. Null Values</h2>
        <p class="{'pass' if not null_cols else 'warn'}">
            {'No null values found' if not null_cols else f'{len(null_cols)} columns with nulls'}
        </p>
    </div>

    <div class="card">
        <h2>2. Ballistic Coefficient (Size Proxy)</h2>
        <p>Range: [{bc_min:.6f}, {bc_max:.6f}]</p>
        <p>Mean: {bc_mean:.6f} | Median: {bc_median:.6f}</p>
        <p>BC-SRP Correlation: {bc_srp_corr:.4f}</p>
        <table>
            <tr><th>Size Range</th><th>Count</th><th>Percentage</th></tr>
"""
    for row in size_dist.iter_rows():
        pct = row[1] / size_total * 100
        html += f"            <tr><td>{row[0]}</td><td>{row[1]:,}</td><td>{pct:.1f}%</td></tr>\n"

    html += """        </table>
    </div>

    <div class="card">
        <h2>3. RTN Frame Validation</h2>
        <p>Max |R*T|: {rt_dot:.2e} <span class="pass">{'OK' if rt_dot < 1e-14 else 'FAIL'}</span></p>
        <p>Max |R*W|: {rw_dot:.2e} <span class="pass">{'OK' if rw_dot < 1e-14 else 'FAIL'}</span></p>
        <p>Max |T*W|: {tw_dot:.2e} <span class="pass">{'OK' if tw_dot < 1e-14 else 'FAIL'}</span></p>
        <p>det([R,T,W]): {det_val:.2e} <span class="pass">{'OK' if abs(det_val-1) < 1e-14 else 'FAIL'}</span></p>
        <p>Overall: <span class="pass">{'PASS' if rtn_ok else 'FAIL'}</span></p>
    </div>

    <div class="card">
        <h2>4. Dataset Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total observations</td><td>{n_rows:,}</td></tr>
            <tr><td>Total features</td><td>{n_cols}</td></tr>
            <tr><td>Null columns</td><td>{len(null_cols)}</td></tr>
            <tr><td>BC-SRP correlation</td><td>{bc_srp_corr:.4f}</td></tr>
            <tr><td>Size range coverage</td><td>{len(size_dist)} categories</td></tr>
            <tr><td>RTN validation</td><td>{'PASS' if rtn_ok else 'FAIL'}</td></tr>
        </table>
    </div>

    <p><em>Report generated by Phase 1 Pipeline</em></p>
</body>
</html>"""

    with open(OUTPUT_FILE, "w") as f:
        f.write(html)

    fsize = OUTPUT_FILE.stat().st_size / 1024
    print(f"  Report saved to {OUTPUT_FILE} ({fsize:.1f} KB)")
    print(f"\n  Stage 5 complete ({t.elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 VCM Pipeline")
    parser.add_argument(
        "--skip-to",
        default=None,
        help=f"Stage to start from: 0-5 or {', '.join(STAGE_NAMES)} (default: run all)",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Write intermediate Parquet files for each stage",
    )
    args = parser.parse_args()

    if args.skip_to is not None:
        try:
            skip_idx = int(args.skip_to)
        except ValueError:
            skip_idx = STAGE_NAMES.index(args.skip_to)
    else:
        skip_idx = 0

    stages = [
        (0, "RTN / RSW Coordinate Derivation", stage_rtn),
        (1, "Orbital Element Derivation", stage_elements),
        (2, "Celestrak Data Integration", stage_celestrak),
        (3, "Temporal Sequence Creation", stage_sequences),
        (4, "Train/Val/Test Split", stage_split),
        (5, "Data Profiling & Quality Report", stage_quality),
    ]

    banner("Phase 1 VCM Pipeline", timer())
    print(f"Starting from stage {skip_idx}: {STAGE_NAMES[skip_idx]}")
    print(f"Save intermediates: {args.save_intermediates}")

    t_total = timer()

    df = pl.read_parquet(INPUT_FILE)
    print(f"Loaded input: {INPUT_FILE} — {len(df):,} rows, {len(df.columns)} columns")

    intermediate_files = [
        "vcm_with_rtn.parquet",
        "vcm_with_elements.parquet",
        "vcm_with_labels.parquet",
        "vcm_sequences.parquet",
        "vcm_splits.parquet",
    ]

    for idx, name, func in stages:
        if idx < skip_idx:
            continue

        banner(f"Stage {idx}: {name}", timer())
        result = func(df)

        if result is not None:
            df = result
            if args.save_intermediates and idx < len(intermediate_files):
                save_parquet(df, DATA_DIR / intermediate_files[idx])
                file_size_mb = report_file_size(DATA_DIR / intermediate_files[idx])
                print(
                    f"  Intermediate saved: {intermediate_files[idx]} ({file_size_mb:.1f} MB)"
                )

    if not args.save_intermediates:
        OUTPUT_FILE = DATA_DIR / "vcm_splits.parquet"
        print(f"\nSaving final output to {OUTPUT_FILE}...")
        save_parquet(df, OUTPUT_FILE)
        file_size_mb = report_file_size(OUTPUT_FILE)
        print(f"  Saved: {OUTPUT_FILE} ({file_size_mb:.1f} MB)")

    print(f"\n{'=' * 60}")
    print("Phase 1 Pipeline Complete!")
    print(f"  Total time: {t_total.elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
