"""Shared VCM (Vector Covariance Message) parsing logic."""

import re
from pathlib import Path

import pandas as pd

SEP_RE = re.compile(r"Z{10,}")


def clean_line(line: str) -> str:
    line = line.rstrip()
    if line.startswith("<>"):
        line = line[2:].strip()
    return line


def parse_triplet(line: str, label: str):
    pattern = rf"^{re.escape(label)}:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)$"
    m = re.match(pattern, line)
    if not m:
        return None
    return tuple(float(x) for x in m.groups())


def parse_single_message(block: str) -> dict:
    """Parse one VCM message block into a flat dict."""
    row = {}

    for raw_line in block.splitlines():
        line = clean_line(raw_line)
        if not line:
            continue

        if line.startswith("SP VECTOR/COVARIANCE MESSAGE"):
            row["message_type"] = line
            continue

        m = re.match(r"^MESSAGE TIME \(UTC\):\s*(.*?)\s+CENTER:\s*(.*)$", line)
        if m:
            row["message_time_utc"] = m.group(1).strip()
            row["center"] = m.group(2).strip()
            continue

        m = re.match(r"^SATELLITE NUMBER:\s*(\d+)\s+INT\. DES\.:\s*(.*)$", line)
        if m:
            row["satellite_number"] = int(m.group(1))
            row["int_des"] = m.group(2).strip()
            continue

        m = re.match(r"^COMMON NAME:\s*(.*)$", line)
        if m:
            row["common_name"] = m.group(1).strip()
            continue

        m = re.match(r"^EPOCH TIME \(UTC\):\s*(.*?)\s+EPOCH REV:\s*(\d+)$", line)
        if m:
            row["epoch_time_utc"] = m.group(1).strip()
            row["epoch_rev"] = int(m.group(2))
            continue

        matched_vector = False
        for label, prefix in [
            ("J2K POS (KM)", "j2k_pos"),
            ("J2K VEL (KM/S)", "j2k_vel"),
            ("ECI POS (KM)", "eci_pos"),
            ("ECI VEL (KM/S)", "eci_vel"),
            ("EFG POS (KM)", "efg_pos"),
            ("EFG VEL (KM/S)", "efg_vel"),
        ]:
            vals = parse_triplet(line, label)
            if vals is not None:
                row[f"{prefix}_x"] = vals[0]
                row[f"{prefix}_y"] = vals[1]
                row[f"{prefix}_z"] = vals[2]
                matched_vector = True
                break

        if matched_vector:
            continue

        patterns = [
            (
                r"^GEOPOTENTIAL:\s*(.*?)\s+DRAG:\s*(.*?)\s+LUNAR/SOLAR:\s*(.*)$",
                ["geopotential", "drag_model", "lunar_solar"],
                [str, str, str],
            ),
            (
                r"^SOLAR RAD PRESS:\s*(.*?)\s+SOLID EARTH TIDES:\s*(.*?)\s+IN-TRACK THRUST:\s*(.*)$",
                ["solar_rad_press", "solid_earth_tides", "in_track_thrust"],
                [str, str, str],
            ),
            (
                r"^BALLISTIC COEF \(M2/KG\):\s*([-\d.E+]+)\s+BDOT \(M2/KG-S\):\s*([-\d.E+]+)$",
                ["ballistic_coef_m2kg", "bdot_m2kg_s"],
                [float, float],
            ),
            (
                r"^SOLAR RAD PRESS COEFF \(M2/KG\):\s*([-\d.E+]+)\s+EDR\(W/KG\):\s*([-\d.E+]+)$",
                ["solar_rad_press_coeff_m2kg", "edr_wkg"],
                [float, float],
            ),
            (
                r"^THRUST ACCEL \(M/S2\):\s*([-\d.E+]+)\s+C\.M\. OFFSET \(M\):\s*([-\d.E+]+)$",
                ["thrust_accel_ms2", "cm_offset_m"],
                [float, float],
            ),
            (
                r"^SOLAR FLUX:\s*F10:\s*([-\d.E+]+)\s+AVERAGE F10:\s*([-\d.E+]+)\s+AVERAGE AP:\s*([-\d.E+]+)$",
                ["solar_flux_f10", "average_f10", "average_ap"],
                [float, float, float],
            ),
            (
                r"^TAI-UTC \(S\):\s*([-\d.E+]+)\s+UT1-UTC \(S\):\s*([-\d.E+]+)\s+UT1 RATE \(MS/DAY\):\s*([-\d.E+]+)$",
                ["tai_utc_s", "ut1_utc_s", "ut1_rate_ms_day"],
                [float, float, float],
            ),
            (
                r"^POLAR MOT X,Y \(ARCSEC\):\s*([-\d.E+]+)\s+([-\d.E+]+)\s+IAU 1980 NUTAT:\s*(.*)$",
                ["polar_mot_x_arcsec", "polar_mot_y_arcsec", "iau_1980_nutat"],
                [float, float, str],
            ),
            (
                r"^TIME CONST LEAP SECOND TIME \(UTC\):\s*(.*)$",
                ["time_const_leap_second_time_utc"],
                [str],
            ),
            (
                r"^INTEGRATOR MODE:\s*(.*?)\s+COORD SYS:\s*(.*?)\s+PARTIALS:\s*(.*)$",
                ["integrator_mode", "coord_sys", "partials"],
                [str, str, str],
            ),
            (
                r"^STEP MODE:\s*(.*?)\s+FIXED STEP:\s*(.*?)\s+STEP SIZE SELECTION:\s*(.*)$",
                ["step_mode", "fixed_step", "step_size_selection"],
                [str, str, str],
            ),
            (
                r"^INITIAL STEP SIZE \(S\):\s*([-\d.E+]+)\s+ERROR CONTROL:\s*([-\d.E+]+)$",
                ["initial_step_size_s", "error_control"],
                [float, float],
            ),
            (
                r"^VECTOR U,V,W SIGMAS \(KM\):\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)$",
                ["vector_u_sigma_km", "vector_v_sigma_km", "vector_w_sigma_km"],
                [float, float, float],
            ),
            (
                r"^VECTOR UD,VD,WD SIGMAS \(KM/S\):\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)$",
                ["vector_ud_sigma_kms", "vector_vd_sigma_kms", "vector_wd_sigma_kms"],
                [float, float, float],
            ),
        ]

        for pattern, keys, casters in patterns:
            m = re.match(pattern, line)
            if m:
                for key, caster, val in zip(keys, casters, m.groups()):
                    try:
                        row[key] = caster(val.strip())
                    except (ValueError, AttributeError):
                        row[key] = None
                break

    return row


def parse_vcm_file(path: Path) -> pd.DataFrame:
    """Parse a single .vcm file into a DataFrame."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = [b.strip() for b in SEP_RE.split(text) if b.strip()]
    rows = [parse_single_message(b) for b in blocks]
    return pd.DataFrame(rows)


def discover_subfolders(vcm_root: Path) -> list[Path]:
    """Return all numeric subfolders (00, 01, ...) sorted."""
    folders = sorted(
        [p for p in vcm_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    return folders
