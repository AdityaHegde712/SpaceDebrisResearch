"""
Unified debug & diagnostics tool.
Replaces: quick_test.py, test_join_where.py, final_error_sat.py,
          trace_bad_sat.py, analyze_dist.py, debug_single_sat.py,
          debug_satellite.py, debug_propagation.py, debug_time_spans.py,
          t3a4_diagnose.py

Usage:
    python scripts/debug.py quick-test
    python scripts/debug.py trace-bad-sat [--sat-id N]
    python scripts/debug.py debug-satellite [--sat-id N]
    ...
    python scripts/debug.py list       # show all commands
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path("data")
ML_READY = DATA_DIR / "vcm_ml_ready.parquet"


# ===================================================================
# quick-test: Basic RK4 propagation sanity check
# ===================================================================


def cmd_quick_test(args):
    from models.physics_propagator import propagate_fixed_rk4, GM, R_EARTH

    print("Starting propagation test...")
    R = R_EARTH + 500
    vc = np.sqrt(GM / R)
    s0 = np.array([R, 0.0, 0.0, 0.0, vc, 0.0])
    E0 = vc**2 / 2 - GM / R
    T = 2 * np.pi * np.sqrt(R**3 / GM)
    print(f"R={R:.1f} vc={vc:.4f} T={T:.0f}")
    print("Propagating 1 orbit...")
    s1 = propagate_fixed_rk4(s0, T, dt=60.0)
    print("Done propagating.")
    r = np.linalg.norm(s1[:3])
    v = np.linalg.norm(s1[3:])
    E1 = v**2 / 2 - GM / r
    print(f"1 orbit: alt={r - R_EARTH:.3f} km, E_rel={abs(E1 - E0) / abs(E0):.2e}")
    print("Test complete.")


# ===================================================================
# test-join-where: Polars join_where test
# ===================================================================


def cmd_test_join_where(args):
    print("Polars version:", pl.__version__)
    f = pl.DataFrame(
        {"a": [1, 1, 1, 2, 2], "i": [0, 1, 2, 0, 1], "x": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    t = pl.DataFrame({"a": [1, 2], "s": [0, 0], "e": [2, 1]}).with_columns(
        pl.int_range(0, pl.len()).alias("wid")
    )
    print("Has join_where:", hasattr(f, "join_where"))
    if hasattr(f, "join_where"):
        r = f.join_where(
            t,
            pl.col("a") == pl.col("a_right"),
            pl.col("i") >= pl.col("s"),
            pl.col("i") <= pl.col("e"),
        )
        print(f"Range-join OK, rows: {len(r)}")
        print(r)
    print("SUCCESS")


# ===================================================================
# final-error-sat: Check final propagation error for a specific satellite
# ===================================================================


def cmd_final_error_sat(args):
    from models.physics_propagator import propagate_fixed_rk4

    sat_id = args.sat_id
    df = pl.read_parquet(
        ML_READY,
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
        ],
    )
    data = df.filter(pl.col("satellite_number") == sat_id).sort("epoch_dt")
    epochs_s = np.array([e.timestamp() for e in data["epoch_dt"]], dtype=float)
    pos_x = np.array(data["j2k_pos_x"], dtype=float)
    pos_y = np.array(data["j2k_pos_y"], dtype=float)
    pos_z = np.array(data["j2k_pos_z"], dtype=float)
    state = np.array(
        [
            pos_x[0],
            pos_y[0],
            pos_z[0],
            data["j2k_vel_x"][0],
            data["j2k_vel_y"][0],
            data["j2k_vel_z"][0],
        ],
        dtype=float,
    )
    last_s = epochs_s[0]
    for i in range(1, len(epochs_s)):
        delta_s = epochs_s[i] - last_s
        if delta_s <= 0:
            continue
        state = propagate_fixed_rk4(state, delta_s, dt=120.0)
        true_pos = np.array([pos_x[i], pos_y[i], pos_z[i]])
        err = np.linalg.norm(state[:3] - true_pos)
        r = np.linalg.norm(state[:3])
        last_s = epochs_s[i]
    print(f"Sat {sat_id}: {len(data)} epochs")
    print(f"  Final error: {err:.1f} km")
    print(f"  Final r: {r:.1f} km")
    print(f"  Final alt: {r - 6378.137:.1f} km")
    print(f"  Time span: {(epochs_s[-1] - epochs_s[0]) / 3600:.1f} hours")


# ===================================================================
# trace-bad-sat: Trace a satellite to find the blowup point
# ===================================================================


def cmd_trace_bad_sat(args):
    from models.physics_propagator import propagate_fixed_rk4, GM

    sat_id = args.sat_id
    df = pl.read_parquet(
        ML_READY,
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
        ],
    )
    data = df.filter(pl.col("satellite_number") == sat_id).sort("epoch_dt")
    epochs_s = np.array([e.timestamp() for e in data["epoch_dt"]], dtype=float)
    pos_x = np.array(data["j2k_pos_x"], dtype=float)
    pos_y = np.array(data["j2k_pos_y"], dtype=float)
    pos_z = np.array(data["j2k_pos_z"], dtype=float)
    vel_x = np.array(data["j2k_vel_x"], dtype=float)
    vel_y = np.array(data["j2k_vel_y"], dtype=float)
    vel_z = np.array(data["j2k_vel_z"], dtype=float)

    print(
        f"Sat {sat_id}: {len(data)} epochs, {data['mean_altitude_km'].mean():.0f} km alt"
    )

    state = np.array(
        [pos_x[0], pos_y[0], pos_z[0], vel_x[0], vel_y[0], vel_z[0]], dtype=float
    )
    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:])
    h = np.cross(state[:3], state[3:])
    e_vec = np.cross(state[3:], h) / GM - state[:3] / r
    e = np.linalg.norm(e_vec)
    i = np.degrees(np.arccos(h[2] / np.linalg.norm(h)))
    E = v**2 / 2 - GM / r
    a = -GM / (2 * E)
    T = 2 * np.pi * np.sqrt(a**3 / GM)
    print(f"  a={a:.1f} km, e={e:.6f}, i={i:.2f}°, T={T / 60:.1f} min")

    last_s = epochs_s[0]
    bad_idx = -1
    for i in range(1, len(epochs_s)):
        delta_s = epochs_s[i] - last_s
        if delta_s <= 0:
            continue
        state = propagate_fixed_rk4(state, delta_s, dt=120.0)
        r = np.linalg.norm(state[:3])
        v = np.linalg.norm(state[3:])
        E = v**2 / 2 - GM / r
        if E >= 0 or r > 1000000:
            bad_idx = i
            print(f"\nUNBOUND at step {i}:")
            print(
                f"  t={((epochs_s[i] - epochs_s[0]) / 3600):.1f}h r={r:.1f} km E={E:.4f}"
            )
            break
        last_s = epochs_s[i]

    if bad_idx == -1:
        print("Propagation completed without going unbound.")
    else:
        state2 = np.array(
            [pos_x[0], pos_y[0], pos_z[0], vel_x[0], vel_y[0], vel_z[0]], dtype=float
        )
        last_s = epochs_s[0]
        for i in range(1, min(bad_idx + 3, len(epochs_s))):
            delta_s = epochs_s[i] - last_s
            state2 = propagate_fixed_rk4(state2, delta_s, dt=120.0)
            r = np.linalg.norm(state2[:3])
            v = np.linalg.norm(state2[3:])
            E = v**2 / 2 - GM / r
            err = np.linalg.norm(state2[:3] - np.array([pos_x[i], pos_y[i], pos_z[i]]))
            hours = (epochs_s[i] - epochs_s[0]) / 3600
            print(
                f"  step={i}, t={hours:.1f}h, err={err:.0f}km, r={r:.0f}km, E={E:.6f}"
            )
            last_s = epochs_s[i]


# ===================================================================
# analyze-dist: Analyze error distribution from physics baseline output
# ===================================================================


def cmd_analyze_dist(args):
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return
    df = pl.read_csv(path)
    print(f"Total epoch pairs: {len(df)}")
    print("Error stats (km):")
    for stat in ["mean", "median"]:
        print(f"  {stat}={getattr(df['err_km'], stat)():.1f}")
    for p in [0.9]:
        print(f"  p{p * 100:.0f}={df['err_km'].quantile(p):.1f}")
    print(f"  min={df['err_km'].min():.1f}")
    print(f"  max={df['err_km'].max():.1f}")

    d1 = df.filter((pl.col("days") >= 0.5) & (pl.col("days") <= 1.5))
    print(f"\n1d bucket ({len(d1)} pairs): mean={d1['err_km'].mean():.1f} km")

    per_sat = (
        df.group_by("satellite_number")
        .agg(
            [
                pl.col("err_km").max().alias("max_err"),
                pl.col("days").max().alias("max_days"),
            ]
        )
        .sort("max_err", descending=True)
    )
    print("\n=== Top satellites by max error ===")
    for r in per_sat.head(10).to_dicts():
        print(
            f"  sat={r['satellite_number']} max_err={r['max_err']:.0f}km span={r['max_days']:.0f}d"
        )

    print("\n=== Distribution by altitude band ===")
    for label, lo, hi in [
        ("LEO", 0, 2000),
        ("MEO", 2000, 20000),
        ("GEO", 20000, 50000),
    ]:
        sub = df.filter((pl.col("alt_km") >= lo) & (pl.col("alt_km") < hi))
        if len(sub) > 0:
            print(
                f"  {label}: {len(sub)} pairs, median={sub['err_km'].median():.0f} km"
            )


# ===================================================================
# debug-single-sat: Step-by-step propagation for one satellite
# ===================================================================


def cmd_debug_single_sat(args):
    from models.physics_propagator import propagate_fixed_rk4, GM, R_EARTH

    sat_id = args.sat_id
    max_steps = args.max_steps

    df = pl.read_parquet(
        ML_READY,
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
        ],
    )
    data = df.filter(pl.col("satellite_number") == sat_id).sort("epoch_dt")
    print(
        f"Satellite {sat_id}: {len(data)} rows, alt={data['mean_altitude_km'].mean():.0f} km"
    )

    epochs_s = np.array([e.timestamp() for e in data["epoch_dt"]], dtype=float)
    pos_x = np.array(data["j2k_pos_x"], dtype=float)
    pos_y = np.array(data["j2k_pos_y"], dtype=float)
    pos_z = np.array(data["j2k_pos_z"], dtype=float)
    vel_x = np.array(data["j2k_vel_x"], dtype=float)
    vel_y = np.array(data["j2k_vel_y"], dtype=float)
    vel_z = np.array(data["j2k_vel_z"], dtype=float)

    state = np.array(
        [pos_x[0], pos_y[0], pos_z[0], vel_x[0], vel_y[0], vel_z[0]], dtype=float
    )

    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:])
    E = v**2 / 2 - GM / r
    a = -GM / (2 * E)
    T = 2 * np.pi * np.sqrt(a**3 / GM)
    print(f"  a={a:.1f} km, T={T / 60:.1f} min, alt={r - R_EARTH:.1f} km")
    print(f"  Epochs: {len(data)} over {(epochs_s[-1] - epochs_s[0]) / 3600:.1f} hours")

    print(
        f"\n{'Step':>5s} {'Time(h)':>8s} {'Delta(km)':>10s} {'R(km)':>8s} {'A(km)':>8s}"
    )
    print("-" * 55)

    last_s = epochs_s[0]
    for i in range(1, min(len(epochs_s), max_steps + 1)):
        delta_s = epochs_s[i] - last_s
        state = propagate_fixed_rk4(state, delta_s, dt=120.0)
        true_pos = np.array([pos_x[i], pos_y[i], pos_z[i]])
        err = np.linalg.norm(state[:3] - true_pos)
        r_pred = np.linalg.norm(state[:3])
        v_pred = np.linalg.norm(state[3:])
        E_pred = v_pred**2 / 2 - GM / r_pred
        a_pred = -GM / (2 * E_pred)
        hours = (epochs_s[i] - epochs_s[0]) / 3600
        print(f"{i:>5d} {hours:>8.3f} {err:>10.1f} {r_pred:>8.1f} {a_pred:>8.1f}")
        last_s = epochs_s[i]
        if i >= max_steps:
            print(f"  ... (+ {len(epochs_s) - 1 - max_steps} more epochs)")
            break


# ===================================================================
# debug-satellite: Inspect satellite metadata, epochs, gaps
# ===================================================================


def cmd_debug_satellite(args):
    from models.physics_propagator import GM

    sat_id = args.sat_id
    df = pl.read_parquet(
        ML_READY,
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
        ],
    )
    data = df.filter(pl.col("satellite_number") == sat_id).sort("epoch_dt")
    print(f"=== Satellite {sat_id} ===")
    print(f"  Rows: {len(data)}, Alt: {data['mean_altitude_km'].mean():.1f} km")
    epochs_s = np.array([e.timestamp() for e in data["epoch_dt"]], dtype=float)
    gaps = np.diff(epochs_s)
    print(
        f"  Gaps (hours): min={gaps.min() / 3600:.2f}, mean={gaps.mean() / 3600:.2f}, max={gaps.max() / 3600:.2f}"
    )
    print(f"  Span: {(epochs_s[-1] - epochs_s[0]) / 3600:.1f} hours")

    r0 = np.array([data["j2k_pos_x"][0], data["j2k_pos_y"][0], data["j2k_pos_z"][0]])
    T = 2 * np.pi * np.sqrt(np.linalg.norm(r0) ** 3 / GM)
    print(f"  Orbital period: {T / 3600:.2f} hours")

    epoch0_s = epochs_s[0]
    for target_name, target_h in [("1d", 86400), ("3d", 259200), ("7d", 604800)]:
        target_s = epoch0_s + target_h
        deltas = np.abs(epochs_s - target_s)
        nearest_idx = np.argmin(deltas)
        print(
            f"  Target {target_name}: nearest obs idx={nearest_idx}/{len(epochs_s)}, "
            f"delta={deltas[nearest_idx] / 3600:.2f}h"
        )


# ===================================================================
# debug-propagation: Test one-orbit and multi-day propagation
# ===================================================================


def cmd_debug_propagation(args):
    sat_id = args.sat_id
    df = pl.read_parquet(
        ML_READY,
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
        ],
    )
    sat = df.filter(
        (pl.col("split") == "test") & (pl.col("satellite_number") == sat_id)
    )
    print(f"Sat {sat_id}: {len(sat)} rows")

    first = sat.head(1)
    last = sat.tail(1)
    r0 = np.array([first["j2k_pos_x"][0], first["j2k_pos_y"][0], first["j2k_pos_z"][0]])
    v0 = np.array([first["j2k_vel_x"][0], first["j2k_vel_y"][0], first["j2k_vel_z"][0]])
    r1 = np.array([last["j2k_pos_x"][0], last["j2k_pos_y"][0], last["j2k_pos_z"][0]])

    r_mag = np.linalg.norm(r0)
    v_mag = np.linalg.norm(v0)
    GM = 398600.4418
    specific_energy = v_mag**2 / 2 - GM / r_mag
    a = -GM / (2 * specific_energy)
    print(f"  Alt: {r_mag - 6378.137:.1f} km, a: {a:.1f} km")

    from models.physics_propagator import propagate_fixed_rk4

    state0 = np.hstack([r0, v0])
    orbital_period = 2 * np.pi * np.sqrt(a**3 / GM)

    pred_1orbit = propagate_fixed_rk4(state0, orbital_period, dt=60.0)
    err_1orbit = np.linalg.norm(pred_1orbit[:3] - r0)
    print(f"  1-orbit closure error: {err_1orbit:.3f} km")

    # Also try propagating to final time
    time_span_s = (last["epoch_dt"][0] - first["epoch_dt"][0]).total_seconds()
    if time_span_s > 0:
        pred_final = propagate_fixed_rk4(state0, time_span_s, dt=300.0)
        err_final = np.linalg.norm(pred_final[:3] - r1)
        print(f"  Final error ({time_span_s / 86400:.1f} days): {err_final:.1f} km")


# ===================================================================
# debug-time-spans: Check time spans for propagation test data
# ===================================================================


def cmd_debug_time_spans(args):
    df = pl.read_parquet(ML_READY, columns=["satellite_number", "epoch_dt", "split"])
    test = df.filter(pl.col("split") == "test")
    sats = test.group_by("satellite_number").agg(
        [
            pl.col("epoch_dt").first().alias("e0"),
            pl.col("epoch_dt").last().alias("e1"),
        ]
    )
    sats = sats.with_columns(
        [
            (pl.col("e1") - pl.col("e0")).dt.total_seconds().alias("span_s"),
        ]
    )
    print("First 10 satellites:")
    for row in sats.head(10).iter_rows(named=True):
        print(f"  Sat {row['satellite_number']}: span={row['span_s'] / 86400:.2f} days")
    print(
        f"\nSpan range: [{sats['span_s'].min():.1f}, {sats['span_s'].max():.1f}] seconds"
    )


# ===================================================================
# t3a4-diagnose: Diagnose TCN failure to learn
# ===================================================================


def cmd_t3a4_diagnose(args):
    EXCLUDE = {"object_class", "collision_risk_proxy", "debris_cloud_membership"}

    feat = pl.read_parquet(DATA_DIR / "vcm_ml_ready_25_50cm.parquet")
    feat = feat.sort(["satellite_number", "epoch_dt"])
    feat = feat.with_columns(
        [pl.int_range(0, pl.len()).over("satellite_number").alias("epoch_idx")]
    )
    tgt = pl.read_parquet(DATA_DIR / "targets_25_50cm.parquet")

    feat_cols = [c for c in feat.columns if c not in EXCLUDE]
    joined = tgt.join(
        feat.select(feat_cols),
        left_on=["satellite_number", "window_end_idx"],
        right_on=["satellite_number", "epoch_idx"],
        how="inner",
    )
    print(f"Joined: {len(joined)} windows")

    num_cols = [
        c
        for c in joined.columns
        if joined[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
    ]
    num_cols = [
        c
        for c in num_cols
        if c
        not in [
            "satellite_number",
            "window_id",
            "window_start_idx",
            "window_end_idx",
            "epoch_idx",
            "split",
            "epoch_dt",
        ]
    ]
    risk = joined["collision_risk"].to_numpy()

    print("\n=== Top 20 features by |corr| with collision_risk ===")
    corrs = []
    for c in num_cols:
        vals = joined[c].to_numpy()
        if np.std(vals) > 1e-10:
            corr = float(np.corrcoef(vals, risk)[0, 1])
            corrs.append((c, abs(corr), corr))
    corrs.sort(key=lambda x: -x[1])
    for c, abs_c, corr in corrs[:20]:
        print(f"  {c:35s} |r|={abs_c:.4f}  r={corr:.4f}")

    print(f"\nTotal features: {len(num_cols)}")
    print(f"Features with |r| > 0.4: {sum(1 for _, abs_c, _ in corrs if abs_c > 0.4)}")
    print("\nKey insight: collision_risk = threshold on log_uncertainty_volume.")
    print(
        "RF AUC=1.0 finds this threshold in 1 split. TCN has optimization difficulty."
    )


# ===================================================================
# Dispatcher
# ===================================================================

COMMANDS = {
    "quick-test": cmd_quick_test,
    "test-join-where": cmd_test_join_where,
    "final-error-sat": cmd_final_error_sat,
    "trace-bad-sat": cmd_trace_bad_sat,
    "analyze-dist": cmd_analyze_dist,
    "debug-single-sat": cmd_debug_single_sat,
    "debug-satellite": cmd_debug_satellite,
    "debug-propagation": cmd_debug_propagation,
    "debug-time-spans": cmd_debug_time_spans,
    "t3a4-diagnose": cmd_t3a4_diagnose,
}


def build_parser():
    parser = argparse.ArgumentParser(description="Unified debug & diagnostics")
    parser.add_argument(
        "command",
        choices=list(COMMANDS.keys()) + ["list"],
        help="Debug command to run (use 'list' to show all)",
    )
    parser.add_argument(
        "--sat-id", type=int, default=54899, help="Satellite NORAD ID (default: 54899)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Max steps for single-sat debug (default: 20)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="reports/physics_baseline_errors.csv",
        help="CSV file path for analyze-dist (default: reports/physics_baseline_errors.csv)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        print("Available debug commands:")
        for name in sorted(COMMANDS):
            print(f"  {name}")
        return

    sys.path.insert(0, ".")
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
