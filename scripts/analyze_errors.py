import polars as pl
import sys

sys.path.insert(0, ".")

df = pl.read_csv("reports/physics_baseline_errors.csv")

# Filter out nulls for sorting
df_valid = df.filter(pl.col("1d_err_km").is_not_null())

# Top 5 smallest errors
print("=== 5 best satellites ===")
small = df_valid.sort("1d_err_km").head(5)
for r in small.to_dicts():
    print(
        f"  sat={r['satellite_number']} alt={r['alt_km']:.0f}km  1d={r['1d_err_km']:.1f}km  3d={r['3d_err_km']:.1f}km  7d={r['7d_err_km']:.1f}km"
    )

# Top 5 largest errors
print("\n=== 5 worst satellites ===")
large = df_valid.sort("1d_err_km", descending=True).head(5)
for r in large.to_dicts():
    print(
        f"  sat={r['satellite_number']} alt={r['alt_km']:.0f}km  1d={r['1d_err_km']:.1f}km  3d={r['3d_err_km']:.1f}km  7d={r['7d_err_km']:.1f}km"
    )

# Group by altitude bins
bins = [
    ("LEO (<1k)", pl.col("alt_km") < 1000),
    ("MEO (1-10k)", (pl.col("alt_km") >= 1000) & (pl.col("alt_km") < 10000)),
    ("HEO (10-30k)", (pl.col("alt_km") >= 10000) & (pl.col("alt_km") < 30000)),
    ("GEO (>30k)", pl.col("alt_km") >= 30000),
]

print("\n=== By altitude band ===")
for label, cond in bins:
    sub = df.filter(cond)
    if len(sub) > 0:
        median = sub["1d_err_km"].median()
        mx = sub["1d_err_km"].max()
        print(f"  {label}: {len(sub)} sats, 1d median={median:.0f}km max={mx:.0f}km")
