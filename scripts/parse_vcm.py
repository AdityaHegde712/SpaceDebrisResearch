"""
parse_vcm.py
------------
VCM parser CLI. Reads .vcm files from numbered subfolders under a root directory
and writes a combined Parquet file.

Usage:
    python scripts/parse_vcm.py --vcm-root data/vcm --out-dir data
    python scripts/parse_vcm.py                          # uses defaults
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.vcm_parser import parse_vcm_file, discover_subfolders


def parse_all_folders(vcm_root: Path, out_dir: Path) -> None:
    subfolders = discover_subfolders(vcm_root)

    if not subfolders:
        print(f"[ERROR] No numeric subfolders found under '{vcm_root}'.")
        sys.exit(1)

    print("=" * 60)
    print("  VCM Parser")
    print("=" * 60)
    print(f"\nFound {len(subfolders)} subfolder(s) under '{vcm_root}':")
    for sf in subfolders:
        vcm_files = list(sf.glob("*.vcm"))
        print(f"  {sf.name}/  =>  {len(vcm_files):,} .vcm files")

    print(f"\nOutput directory : {out_dir.resolve()}")
    out_name = f"vcm_output_{len(subfolders)}_folders.parquet"
    out_path = out_dir / out_name
    print(f"Output filename  : {out_name}")
    print()

    input(">>> Press ENTER to begin parsing, or Ctrl+C to abort ...\n")

    all_frames: list[pd.DataFrame] = []
    t0 = time.time()

    for sf in subfolders:
        vcm_files = sorted(sf.glob("*.vcm"))
        print(f"\n[Folder {sf.name}]  parsing {len(vcm_files):,} files ...")
        folder_frames: list[pd.DataFrame] = []

        for vcm_file in tqdm(vcm_files, desc=f"  {sf.name}", unit="file"):
            df = parse_vcm_file(vcm_file)
            df["source_file"] = vcm_file.name
            df["source_folder"] = sf.name
            folder_frames.append(df)

        folder_df = pd.concat(folder_frames, ignore_index=True)
        print(f"  => {len(folder_df):,} rows from folder {sf.name}")
        all_frames.append(folder_df)

    print("\nConcatenating all folders ...")
    final_df = pd.concat(all_frames, ignore_index=True)
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("  PARSE SUMMARY")
    print("=" * 60)
    print(f"  Total rows      : {len(final_df):,}")
    print(f"  Unique sats     : {final_df['satellite_number'].nunique():,}")
    print(f"  Columns         : {len(final_df.columns)}")
    print(f"  Parse time      : {elapsed:.1f}s")

    print("\nOptimising dtypes for Parquet ...")
    for col in final_df.select_dtypes("object").columns:
        n_unique = final_df[col].nunique(dropna=False)
        if n_unique < len(final_df) / 100 and n_unique < 500:
            final_df[col] = final_df[col].astype("category")

    print(f"Writing {out_path} ...")
    final_df.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"Done.  File size: {size_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="VCM Folder Parser")
    parser.add_argument(
        "--vcm-root",
        default="data/vcm",
        help="Root directory containing numbered VCM subfolders (default: data/vcm)",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to write the output parquet file (default: current dir)",
    )
    args = parser.parse_args()

    vcm_root = Path(args.vcm_root)
    out_dir = Path(args.out_dir)

    if not vcm_root.exists():
        print(f"[ERROR] VCM root directory not found: '{vcm_root}'")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    parse_all_folders(vcm_root, out_dir)


if __name__ == "__main__":
    main()
