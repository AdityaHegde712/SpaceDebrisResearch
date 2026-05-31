"""Astrodynamic and project-wide constants — single source of truth."""

# Earth gravitational parameter (km³/s²)
GM = 398600.4418

# Earth equatorial radius (km, WGS-84)
R_EARTH = 6378.137

# J2 harmonic coefficient (dimensionless)
J2 = 1.08263e-3

# Drag coefficient (dimensionless)
CD = 2.2

# WGS-84 ellipsoid parameters
class WGS84:
    A = 6378.137               # semi-major axis (km)
    F = 1.0 / 298.257223563    # flattening
    B = A * (1.0 - F)          # semi-minor axis (km)
    E2 = 1.0 - (B / A) ** 2    # first eccentricity²
    EP2 = (A / B) ** 2 - 1.0   # second eccentricity²

# Time conversions
SECONDS_PER_DAY = 86400.0
MINUTES_PER_DAY = 1440.0

# Ballistic coefficient thresholds for size ranges
SIZE_25CM_BC = 0.00244
SIZE_10CM_BC = 0.0061
SIZE_5CM_BC = 0.0122

# Default sequence length for sliding windows
SEQ_LEN = 20

# ML feature exclusion set (unusable or target-leaking)
EXCLUDE_FEATURES = {"object_class", "collision_risk_proxy", "debris_cloud_membership"}

# Categorical feature columns
CAT_COLS = ["size_range", "altitude_band", "orbit_regime"]

# Pipeline split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Minimum epochs to include a satellite in sequence creation
MIN_EPOCHS = 10
