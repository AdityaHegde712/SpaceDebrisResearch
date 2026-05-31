"""
build_risk_map.py
-----------------
Part B of the Space Debris Risk Pipeline.

Steps:
  1. Load the combined parquet, keeping only needed columns.
  2. Select the LATEST epoch snapshot per satellite.
  3. Filter: position uncertainty < 100 km  (data quality).
  4. Filter: ballistic coefficient in range for 5-50 cm debris (size proxy).
  5. Convert EFG (Earth-Fixed Geocentric) positions to WGS-84 geodetic
     (latitude, longitude, altitude).
  6. [STOP] Show sanity-check visualisation before voxelisation.
  7. Build a 3-D voxel grid (Alt x Lat x Lon) and accumulate each
     object's positional uncertainty as an axis-aligned 3-D Gaussian
     (Probability of Presence).
  8. [STOP] Show grid slice visualisations.
  9. Save grid + metadata as .npz.

Usage:
    python build_risk_map.py --parquet vcm_output_3_folders.parquet
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# WGS-84 constants
# ---------------------------------------------------------------------------
WGS84_A = 6_378.137  # semi-major axis, km
WGS84_F = 1.0 / 298.257_223_563
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # semi-minor axis, km
WGS84_E2 = 1.0 - (WGS84_B / WGS84_A) ** 2  # first eccentricity²
WGS84_EP2 = (WGS84_A / WGS84_B) ** 2 - 1.0  # second eccentricity²

# ---------------------------------------------------------------------------
# Voxel grid definition
# ---------------------------------------------------------------------------
ALT_MIN, ALT_MAX, ALT_STEP = 200.0, 2_000.0, 50.0  # km
LAT_MIN, LAT_MAX, LAT_STEP = -90.0, 90.0, 2.0  # degrees
LON_MIN, LON_MAX, LON_STEP = -180.0, 180.0, 2.0  # degrees

# Ballistic coefficient bounds for 5-50 cm spherical aluminium debris
# BC = CD × (3 / (4 × rho × d));  CD≈2.2, rho_Al≈2700 kg/m³
# d=5 cm  → BC ≈ 0.0122 m²/kg
# d=50 cm → BC ≈ 0.00122 m²/kg
# We add ±50 % margin to account for non-spherical shapes.
BC_MIN = 0.0006  # m²/kg  (~75-cm upper bound with margin)
BC_MAX = 0.0200  # m²/kg  (~3-cm lower bound with margin)

# Only consider debris whose 3-D position uncertainty is below this.
MAX_UNCERTAINTY_KM = 100.0

# Gaussian accumulation: only update voxels within N_SIGMA of each object.
N_SIGMA = 3.0
MIN_SIGMA_KM = 0.001  # numerical floor to avoid degenerate Gaussians

# km per degree of latitude (constant)
KM_PER_DEG_LAT = 111.32


# ---------------------------------------------------------------------------
# Coordinate conversion: EFG (ECEF) → WGS-84 geodetic
# ---------------------------------------------------------------------------
def efg_to_geodetic(
    x_km: np.ndarray, y_km: np.ndarray, z_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Earth-Fixed Geocentric Cartesian (EFG/ECEF) coordinates to
    WGS-84 geodetic (latitude °, longitude °, altitude km).

    Uses Bowring's iterative method (converges in 2-3 iterations).
    """
    p = np.sqrt(x_km**2 + y_km**2)  # distance from polar axis
    lon = np.degrees(np.arctan2(y_km, x_km))  # longitude — exact

    # Initial parametric latitude estimate
    theta = np.arctan2(z_km * WGS84_A, p * WGS84_B)

    # Two Bowring iterations (more than sufficient for sub-metre accuracy)
    for _ in range(3):
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        lat_rad = np.arctan2(
            z_km + WGS84_EP2 * WGS84_B * sin_t**3,
            p - WGS84_E2 * WGS84_A * cos_t**3,
        )
        theta = np.arctan2(
            np.sin(lat_rad) * WGS84_B,
            np.cos(lat_rad) * WGS84_A,
        )

    lat = np.degrees(lat_rad)
    sin_lat = np.sin(lat_rad)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)  # prime-vertical radius
    alt = np.where(
        np.abs(lat_rad) < np.radians(89.9),
        p / np.cos(lat_rad) - N,
        np.abs(z_km) / np.sin(lat_rad) - N * (1.0 - WGS84_E2),
    )
    return lat, lon, alt


# ---------------------------------------------------------------------------
# Voxel grid helpers
# ---------------------------------------------------------------------------
def make_grid_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alt_centers = np.arange(ALT_MIN + ALT_STEP / 2, ALT_MAX, ALT_STEP)
    lat_centers = np.arange(LAT_MIN + LAT_STEP / 2, LAT_MAX, LAT_STEP)
    lon_centers = np.arange(LON_MIN + LON_STEP / 2, LON_MAX, LON_STEP)
    return alt_centers, lat_centers, lon_centers


def _idx(value: float, centers: np.ndarray) -> int:
    """Nearest voxel index (clipped to valid range)."""
    return int(np.clip(np.searchsorted(centers, value), 0, len(centers) - 1))


def accumulate_gaussian(
    grid: np.ndarray,
    alt_c: np.ndarray,
    lat_c: np.ndarray,
    lon_c: np.ndarray,
    pos_alt: float,
    pos_lat: float,
    pos_lon: float,
    sig_alt: float,
    sig_lat: float,
    sig_lon: float,
) -> None:
    """
    Add one object's axis-aligned 3-D Gaussian PoP contribution to *grid*.

    The Gaussian is evaluated only within an N_SIGMA bounding box for speed.
    """

    # --- bounding box in index space ---
    def bounds(pos, sig, centers, step):
        lo = max(0, _idx(pos - N_SIGMA * sig, centers))
        hi = min(len(centers), _idx(pos + N_SIGMA * sig, centers) + 1)
        return lo, hi

    a0, a1 = bounds(pos_alt, sig_alt, alt_c, ALT_STEP)
    l0, l1 = bounds(pos_lat, sig_lat, lat_c, LAT_STEP)
    o0, o1 = bounds(pos_lon, sig_lon, lon_c, LON_STEP)

    if a0 >= a1 or l0 >= l1 or o0 >= o1:
        return

    A, L, O = np.meshgrid(alt_c[a0:a1], lat_c[l0:l1], lon_c[o0:o1], indexing="ij")  # noqa: E741
    exponent = (
        0.5 * ((A - pos_alt) / sig_alt) ** 2
        + 0.5 * ((L - pos_lat) / sig_lat) ** 2
        + 0.5 * ((O - pos_lon) / sig_lon) ** 2
    )
    grid[a0:a1, l0:l1, o0:o1] += np.exp(-exponent)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def _stop(prompt: str) -> None:
    print()
    input(f">>> {prompt}  [Press ENTER to continue, Ctrl+C to abort] ")
    print()


def plot_bc_distribution(df: pd.DataFrame) -> None:
    valid_bc = df["ballistic_coef_m2kg"]
    valid_bc = valid_bc[(valid_bc > 0) & (valid_bc < 0.1)]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(valid_bc, bins=60, color="steelblue", edgecolor="none")
    ax.axvspan(
        BC_MIN,
        BC_MAX,
        alpha=0.2,
        color="red",
        label=f"5-50 cm window [{BC_MIN:.4f}, {BC_MAX:.4f}]",
    )
    ax.set_xlabel("Ballistic Coefficient (m²/kg)")
    ax.set_ylabel("Object count")
    ax.set_title("BC Distribution — size-proxy filter")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig("bc_distribution.png", dpi=120)
    plt.show()
    print("  Saved: bc_distribution.png")


def plot_geodetic_overview(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["altitude_km"], bins=50, color="darkorange", edgecolor="none")
    axes[0].set_xlabel("Altitude (km)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Debris Altitude Distribution")

    sc = axes[1].scatter(
        df["longitude_deg"],
        df["latitude_deg"],
        c=df["altitude_km"],
        cmap="plasma",
        s=2,
        alpha=0.5,
    )
    plt.colorbar(sc, ax=axes[1], label="Altitude (km)")
    axes[1].set_xlabel("Longitude (°)")
    axes[1].set_ylabel("Latitude (°)")
    axes[1].set_title("Debris Ground Track (coloured by altitude)")

    plt.tight_layout()
    plt.savefig("geodetic_overview.png", dpi=120)
    plt.show()
    print("  Saved: geodetic_overview.png")


def plot_grid_slices(
    grid: np.ndarray, alt_c: np.ndarray, lat_c: np.ndarray, lon_c: np.ndarray
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Altitude-integrated global map  (sum over altitude axis)
    global_map = grid.sum(axis=0)
    im0 = axes[0].imshow(
        global_map,
        origin="lower",
        aspect="auto",
        extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        cmap="hot",
    )
    plt.colorbar(im0, ax=axes[0], label="Cumulative PoP")
    axes[0].set(
        xlabel="Longitude (°)",
        ylabel="Latitude (°)",
        title="Global PoP Map (altitude-integrated)",
    )

    # 2. PoP vs Altitude (summed over lat/lon)
    alt_profile = grid.sum(axis=(1, 2))
    axes[1].plot(alt_profile, alt_c, color="steelblue", lw=1.5)
    axes[1].set(
        xlabel="Cumulative PoP", ylabel="Altitude (km)", title="PoP Altitude Profile"
    )
    axes[1].grid(True, alpha=0.3)

    # 3. Hottest altitude shell
    peak_alt_idx = int(np.argmax(grid.sum(axis=(1, 2))))
    shell = grid[peak_alt_idx, :, :]
    im2 = axes[2].imshow(
        shell,
        origin="lower",
        aspect="auto",
        extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        cmap="hot",
    )
    plt.colorbar(im2, ax=axes[2], label="PoP")
    axes[2].set(
        xlabel="Longitude (°)",
        ylabel="Latitude (°)",
        title=f"Hottest Shell: {alt_c[peak_alt_idx]:.0f} km altitude",
    )

    plt.tight_layout()
    plt.savefig("risk_map_slices.png", dpi=120)
    plt.show()
    print("  Saved: risk_map_slices.png")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build 3-D Probabilistic Debris Occupancy Grid"
    )
    ap.add_argument(
        "--parquet",
        required=True,
        help="Path to combined VCM parquet (from parse_vcm_extended.py)",
    )
    ap.add_argument(
        "--out",
        default="debris_risk_map.npz",
        help="Output .npz path for the voxel grid (default: debris_risk_map.npz)",
    )
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"[ERROR] Parquet not found: {parquet_path}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 1. Load — keep only the columns we need
    # -----------------------------------------------------------------------
    COLS = [
        "satellite_number",
        "epoch_time_utc",
        "efg_pos_x",
        "efg_pos_y",
        "efg_pos_z",
        "j2k_vel_x",
        "j2k_vel_y",
        "j2k_vel_z",
        "vector_u_sigma_km",
        "vector_v_sigma_km",
        "vector_w_sigma_km",
        "ballistic_coef_m2kg",
        "solar_flux_f10",
    ]
    print(f"\nLoading columns from {parquet_path.name} …")
    df = pd.read_parquet(parquet_path, columns=COLS, engine="pyarrow")
    print(
        f"  Raw rows: {len(df):,}   Unique satellites: {df['satellite_number'].nunique():,}"
    )

    # -----------------------------------------------------------------------
    # 2. Latest epoch per satellite
    # -----------------------------------------------------------------------
    print("\nSelecting latest epoch per satellite …")
    # Extract a sortable numeric key from the epoch string "YYYY DDD (DD Mon) HH:MM:SS.sss"
    ep = (
        df["epoch_time_utc"]
        .str.extract(r"(\d{4}) (\d{3}) \(\d{2} \w+\) (\d{2}):(\d{2}):(\d{2})")
        .astype(float)
    )
    df["_epoch_key"] = ep[0] * 1e8 + ep[1] * 1e5 + ep[2] * 1e4 + ep[3] * 1e2 + ep[4]
    idx_latest = df.groupby("satellite_number")["_epoch_key"].idxmax()
    df = df.loc[idx_latest].reset_index(drop=True)
    df.drop(columns=["_epoch_key"], inplace=True)
    print(f"  Snapshot rows (one per satellite): {len(df):,}")

    # -----------------------------------------------------------------------
    # 3. Data-quality filter: position uncertainty
    # -----------------------------------------------------------------------
    df["uncertainty_km"] = np.sqrt(
        df["vector_u_sigma_km"] ** 2
        + df["vector_v_sigma_km"] ** 2
        + df["vector_w_sigma_km"] ** 2
    )
    before = len(df)
    df = df[df["uncertainty_km"] < MAX_UNCERTAINTY_KM].reset_index(drop=True)
    print(
        f"\nUncertainty filter (<{MAX_UNCERTAINTY_KM} km): {before} → {len(df):,} rows"
    )

    # -----------------------------------------------------------------------
    # 4. Ballistic-coefficient filter (size proxy for 5-50 cm debris)
    # -----------------------------------------------------------------------
    print(f"\nBC filter  [{BC_MIN:.4f}, {BC_MAX:.4f}] m²/kg  (5-50 cm size proxy)")
    print("  Plotting BC distribution …")
    plot_bc_distribution(df)

    before = len(df)
    df = df[
        (df["ballistic_coef_m2kg"] >= BC_MIN) & (df["ballistic_coef_m2kg"] <= BC_MAX)
    ].reset_index(drop=True)
    print(f"  BC filter: {before} → {len(df):,} rows retained")

    if len(df) == 0:
        print("[WARNING] No rows remain after BC filter.")
        print(
            "          The bounds BC_MIN/BC_MAX at the top of the script can be widened."
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 5. EFG → Geodetic
    # -----------------------------------------------------------------------
    print("\nConverting EFG → WGS-84 geodetic …")
    df["latitude_deg"], df["longitude_deg"], df["altitude_km"] = efg_to_geodetic(
        df["efg_pos_x"].to_numpy(),
        df["efg_pos_y"].to_numpy(),
        df["efg_pos_z"].to_numpy(),
    )

    # Drop objects outside our altitude shell of interest
    before = len(df)
    df = df[
        (df["altitude_km"] >= ALT_MIN) & (df["altitude_km"] <= ALT_MAX)
    ].reset_index(drop=True)
    print(f"  Altitude clipping ({ALT_MIN}-{ALT_MAX} km): {before} → {len(df):,} rows")

    print("\n--- Geodetic summary ---")
    for col in ["altitude_km", "latitude_deg", "longitude_deg", "uncertainty_km"]:
        print(
            f"  {col:<20s}  min={df[col].min():.2f}  max={df[col].max():.2f}  mean={df[col].mean():.2f}"
        )

    print("\nPlotting geodetic overview …")
    plot_geodetic_overview(df)

    _stop("Sanity check done. Proceed to voxelisation?")

    # -----------------------------------------------------------------------
    # 6. Build voxel grid axes
    # -----------------------------------------------------------------------
    alt_c, lat_c, lon_c = make_grid_axes()
    grid = np.zeros((len(alt_c), len(lat_c), len(lon_c)), dtype=np.float32)
    print(
        f"\nVoxel grid shape: {grid.shape}  "
        f"({len(alt_c)} alt × {len(lat_c)} lat × {len(lon_c)} lon)"
    )
    print(f"Memory footprint: {grid.nbytes / 1e6:.1f} MB\n")

    # -----------------------------------------------------------------------
    # 7. Gaussian PoP accumulation
    # -----------------------------------------------------------------------
    # Map UVW sigmas to (alt, lat, lon) space:
    #   U (radial)      → altitude sigma  (km)
    #   W (cross-track) → latitude sigma  (degrees)
    #   V (along-track) → longitude sigma (degrees)
    df["sig_alt"] = df["vector_u_sigma_km"].clip(lower=MIN_SIGMA_KM)
    df["sig_lat"] = (df["vector_w_sigma_km"] / KM_PER_DEG_LAT).clip(
        lower=MIN_SIGMA_KM / KM_PER_DEG_LAT
    )
    df["sig_lon"] = (
        df["vector_v_sigma_km"]
        / (KM_PER_DEG_LAT * np.cos(np.radians(df["latitude_deg"])).clip(lower=0.017))
    ).clip(lower=MIN_SIGMA_KM / KM_PER_DEG_LAT)

    records = df[
        [
            "altitude_km",
            "latitude_deg",
            "longitude_deg",
            "sig_alt",
            "sig_lat",
            "sig_lon",
        ]
    ].to_numpy(dtype=np.float64)

    print("Accumulating Gaussian PoP into voxel grid …")
    for row in tqdm(records, unit="obj"):
        accumulate_gaussian(
            grid,
            alt_c,
            lat_c,
            lon_c,
            pos_alt=row[0],
            pos_lat=row[1],
            pos_lon=row[2],
            sig_alt=row[3],
            sig_lat=row[4],
            sig_lon=row[5],
        )

    # Normalise to [0, 1] for interpretability
    grid_max = grid.max()
    if grid_max > 0:
        grid /= grid_max
    print(f"  Grid max before normalisation: {grid_max:.4f}")
    print(f"  Non-zero voxels: {np.count_nonzero(grid):,} / {grid.size:,}")

    # -----------------------------------------------------------------------
    # 8. Visualise
    # -----------------------------------------------------------------------
    print("\nPlotting grid slices …")
    plot_grid_slices(grid, alt_c, lat_c, lon_c)

    _stop("Grid looks good? Proceed to save?")

    # -----------------------------------------------------------------------
    # 9. Save
    # -----------------------------------------------------------------------
    out_path = Path(args.out)
    np.savez_compressed(
        out_path,
        grid=grid,
        alt_centers=alt_c,
        lat_centers=lat_c,
        lon_centers=lon_c,
        alt_step=np.float32(ALT_STEP),
        lat_step=np.float32(LAT_STEP),
        lon_step=np.float32(LON_STEP),
        n_objects=np.int32(len(df)),
    )
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"\nSaved: {out_path}  ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
