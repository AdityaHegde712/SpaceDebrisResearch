#!/usr/bin/env python3
"""
Temporal Structure Exploration for VCM Data
Analyzes the temporal characteristics of the VCM dataset to determine viability of time-series approach.
"""

import polars as pl
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime


def parse_utc_time(time_str):
    """Parse the UTC time string format: 'YYYY DDD (DD Mon) HH:MM:SS.sss'"""
    try:
        # Remove the day of year and date in parentheses part for parsing
        # Format: '2022 244 (01 SEP) 01:38:58.810'
        # We need: '%Y %j (%d %b) %H:%M:%S.%f'
        return datetime.strptime(time_str.strip(), "%Y %j (%d %b) %H:%M:%S.%f")
    except ValueError:
        # Fallback for any format variations
        try:
            # Try without microseconds
            return datetime.strptime(time_str.strip(), "%Y %j (%d %b) %H:%M:%S")
        except ValueError:
            # If still fails, return None and we'll filter later
            return None


def main():
    # Paths
    data_path = r"C:\Users\hifia\Projects\SpaceDebrisResearch\data\vcm_output_51_folders.parquet"
    output_dir = r"C:\Users\hifia\Projects\SpaceDebrisResearch\data"
    # script_dir = r"C:\Users\hifia\Projects\SpaceDebrisResearch\scripts"

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading VCM data...")
    # Load the parquet file
    df = pl.read_parquet(data_path)

    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns}")

    # 1. Time Range - Parse timestamps first
    print("\n1. Parsing timestamps and calculating time range...")
    # Convert epoch_time_utc to datetime objects
    df_with_time = df.with_columns(
        pl.col("epoch_time_utc")
        .map_elements(parse_utc_time, return_dtype=pl.Datetime)
        .alias("parsed_time")
    )

    # Filter out any rows where parsing failed
    valid_times_df = df_with_time.filter(pl.col("parsed_time").is_not_null())

    if valid_times_df.height == 0:
        raise ValueError("Failed to parse any timestamps")

    print(f"Successfully parsed {valid_times_df.height} out of {df.height} rows")

    time_range = valid_times_df.select(
        [
            pl.col("parsed_time").min().alias("min_time"),
            pl.col("parsed_time").max().alias("max_time"),
        ]
    )
    min_time = time_range.item(0, 0)
    max_time = time_range.item(0, 1)
    print(f"Earliest epoch: {min_time}")
    print(f"Latest epoch: {max_time}")
    total_duration = (max_time - min_time).total_seconds()
    print(
        f"Total duration: {total_duration} seconds ({total_duration / (24 * 3600):.2f} days)"
    )

    # 2. Epochs per Satellite
    print("\n2. Computing epochs per satellite...")
    epochs_per_sat = valid_times_df.group_by("satellite_number").agg(
        pl.count().alias("epoch_count")
    )

    # Assign each satellite to a bin using when-otherwise
    epochs_per_sat = epochs_per_sat.with_columns(
        pl.when(pl.col("epoch_count") == 1)
        .then(pl.lit("1"))
        .when(pl.col("epoch_count") < 5)
        .then(pl.lit("2-5"))
        .when(pl.col("epoch_count") < 10)
        .then(pl.lit("5-10"))
        .when(pl.col("epoch_count") < 20)
        .then(pl.lit("10-20"))
        .when(pl.col("epoch_count") < 50)
        .then(pl.lit("20-50"))
        .otherwise(pl.lit("50+"))
        .alias("epoch_bin")
    )

    # Histogram counts
    hist_counts = (
        epochs_per_sat.group_by("epoch_bin")
        .agg(pl.count().alias("satellite_count"))
        .sort("epoch_bin")
    )

    total_satellites = epochs_per_sat.height
    print(f"Total unique satellites: {total_satellites}")
    print("\nEpoch distribution per satellite:")
    for row in hist_counts.iter_rows():
        label, count = row
        pct = (count / total_satellites) * 100
        print(f"  {label:>4} epochs: {count:>5} satellites ({pct:5.2f}%)")

    # 3. Viable Sequence Length
    print("\n3. Viable sequence length analysis...")
    thresholds = [10, 20, 50]
    for thresh in thresholds:
        count_above = epochs_per_sat.filter(pl.col("epoch_count") >= thresh).height
        pct_above = (count_above / total_satellites) * 100
        print(
            f"  Satellites with >= {thresh} epochs: {count_above:>5} ({pct_above:5.2f}%)"
        )

    # 4. Time Gaps
    print("\n4. Computing time gaps between consecutive observations...")
    # Sort by satellite and parsed time
    df_sorted = valid_times_df.sort(["satellite_number", "parsed_time"])

    # Compute time differences within each satellite (in seconds)
    df_with_gaps = df_sorted.with_columns(
        (pl.col("parsed_time").cast(pl.Datetime).diff().over("satellite_number"))
        .dt.total_seconds()
        .alias("time_gap")
    )

    # Remove null gaps (first observation of each satellite)
    gaps_df = df_with_gaps.filter(pl.col("time_gap").is_not_null())

    if gaps_df.height > 0:
        gap_stats = gaps_df.select(
            [
                pl.col("time_gap").mean().alias("mean_gap"),
                pl.col("time_gap").median().alias("median_gap"),
                pl.col("time_gap").std().alias("std_gap"),
                pl.col("time_gap").max().alias("max_gap"),
                pl.col("time_gap").min().alias("min_gap"),
            ]
        )

        mean_gap = gap_stats.item(0, 0)
        median_gap = gap_stats.item(0, 1)
        std_gap = gap_stats.item(0, 2)
        max_gap = gap_stats.item(0, 3)
        min_gap = gap_stats.item(0, 4)

        print("  Time gap statistics (seconds):")
        print(f"    Mean: {mean_gap:.2f}")
        print(f"    Median: {median_gap:.2f}")
        print(f"    Std: {std_gap:.2f}")
        print(f"    Min: {min_gap:.2f}")
        print(f"    Max: {max_gap:.2f}")
        print(f"    Max gap in days: {max_gap / (24 * 3600):.2f}")
    else:
        print("  No gaps computed (only one observation per satellite?)")
        mean_gap = median_gap = std_gap = max_gap = min_gap = 0

    # 5. Irregular Sampling
    print("\n5. Checking for regular sampling...")
    # For each satellite, check if all gaps are equal (within a small tolerance)
    # We'll consider a satellite regular if the coefficient of variation of gaps is < 1e-6
    gap_cv = (
        gaps_df.group_by("satellite_number")
        .agg((pl.col("time_gap").std() / pl.col("time_gap").mean()).alias("cv_gap"))
        .filter(pl.col("cv_gap").is_not_null())
    )

    regular_satellites = gap_cv.filter(pl.col("cv_gap") < 1e-6).height
    total_satellites_with_gaps = gap_cv.height
    irregular_satellites = total_satellites_with_gaps - regular_satellites

    print(f"  Satellites with multiple observations: {total_satellites_with_gaps}")
    print(
        f"  Regular sampling (CV < 1e-6): {regular_satellites} ({regular_satellites / total_satellites_with_gaps * 100:.2f}%)"
    )
    print(
        f"  Irregular sampling: {irregular_satellites} ({irregular_satellites / total_satellites_with_gaps * 100:.2f}%)"
    )

    # 6. BC/SR Distribution by Epoch Count
    print("\n6. Analyzing ballistic coefficient by epoch count...")
    # Merge epoch count back to valid data to get BC distribution
    df_with_epoch_count = valid_times_df.join(
        epochs_per_sat.select(["satellite_number", "epoch_count"]),
        on="satellite_number",
    )

    # Compute average BC per epoch count group (using same binning as before)
    df_with_epoch_count = df_with_epoch_count.with_columns(
        pl.when(pl.col("epoch_count") == 1)
        .then(pl.lit("1"))
        .when(pl.col("epoch_count") < 5)
        .then(pl.lit("2-5"))
        .when(pl.col("epoch_count") < 10)
        .then(pl.lit("5-10"))
        .when(pl.col("epoch_count") < 20)
        .then(pl.lit("10-20"))
        .when(pl.col("epoch_count") < 50)
        .then(pl.lit("20-50"))
        .otherwise(pl.lit("50+"))
        .alias("epoch_bin")
    )

    bc_stats = (
        df_with_epoch_count.group_by("epoch_bin")
        .agg(
            pl.col("ballistic_coef_m2kg").mean().alias("mean_bc"),
            pl.col("ballistic_coef_m2kg").median().alias("median_bc"),
            pl.col("ballistic_coef_m2kg").std().alias("std_bc"),
        )
        .sort("epoch_bin")
    )

    print("  Ballistic coefficient statistics by epoch bin:")
    print("  Epoch Bin | Mean BC (m²/kg) | Median BC | Std BC")
    print("  --------- | --------------- | --------- | ------")
    for row in bc_stats.iter_rows():
        bin_label, mean_bc, median_bc, std_bc = row
        print(f"  {bin_label:>9} | {mean_bc:15.4f} | {median_bc:9.4f} | {std_bc:6.4f}")

    # 7. Recommended sequence length
    print("\n7. Determining recommended sequence length...")
    # Based on the percentage of satellites with sufficient epochs and the time gap characteristics
    # We'll recommend a sequence length that captures a reasonable fraction of satellites
    # while ensuring we have enough data for meaningful time-series modeling.

    # Let's look at the cumulative distribution of epoch counts using Polars
    sorted_epochs_df = epochs_per_sat.select("epoch_count").sort("epoch_count")
    sorted_epochs = sorted_epochs_df.to_series().to_list()
    total_sats = len(sorted_epochs)

    # Calculate cumulative percentages
    cum_pct = [(i + 1) / total_sats * 100 for i in range(total_sats)]

    # Find the epoch count at which we have 50%, 75%, 90% of satellites
    percentiles = [50, 75, 90]
    recommended_lengths = []
    for p in percentiles:
        # Find first index where cumulative percentage >= p
        idx = next((i for i, cp in enumerate(cum_pct) if cp >= p), total_sats - 1)
        rec_epochs = sorted_epochs[idx]
        recommended_lengths.append((p, rec_epochs))

    print("  Cumulative percentage of satellites by epoch count:")
    for p, epochs in recommended_lengths:
        actual_pct = (
            (sorted_epochs.index(epochs) + 1) / total_sats * 100
            if epochs in sorted_epochs
            else 0
        )
        print(
            f"    {p}% of satellites have <= {epochs} epochs (actual: {actual_pct:.1f}%)"
        )

    # We'll recommend the 75th percentile as a balance between coverage and sequence length
    recommended_seq_length = recommended_lengths[1][1]  # 75th percentile
    print(f"\n  Recommended sequence length: {recommended_seq_length} epochs")
    print(
        "    Justification: This captures ~75 percent of satellites while ensuring sufficient temporal depth."
    )
    print(
        f"    Alternative: Consider {recommended_lengths[2][1]} epochs to capture ~90 percent of satellites."
    )

    # Generate Plots
    print("\n8. Generating diagnostic plots...")

    # Plot 1: Epochs per Satellite Histogram
    plt.figure(figsize=(10, 6))
    # We need to convert the histogram data to a format suitable for bar plot
    bin_labels_list = hist_counts.select("epoch_bin").to_series().to_list()
    counts_list = hist_counts.select("satellite_count").to_series().to_list()
    plt.bar(bin_labels_list, counts_list)
    plt.title("Distribution of Epoch Count per Satellite")
    plt.xlabel("Number of Epochs per Satellite")
    plt.ylabel("Number of Satellites")
    plt.grid(axis="y", alpha=0.5)
    plt.tight_layout()
    epochs_hist_path = os.path.join(output_dir, "epochs_per_satellite_histogram.png")
    plt.savefig(epochs_hist_path, dpi=150)
    plt.close()
    print(f"  Saved epochs histogram to: {epochs_hist_path}")

    # Plot 2: Time Gap Distribution
    if gaps_df.height > 0:
        plt.figure(figsize=(10, 6))
        # Use a log scale for gaps due to potential wide distribution
        gap_values = gaps_df.select("time_gap").to_series().to_list()
        plt.hist(gap_values, bins=50, alpha=0.7, edgecolor="black")
        plt.title("Distribution of Time Gaps Between Consecutive Observations")
        plt.xlabel("Time Gap (seconds)")
        plt.ylabel("Frequency")
        plt.yscale("log")  # Log scale for better visualization
        plt.grid(axis="y", alpha=0.5)
        plt.tight_layout()
        gap_hist_path = os.path.join(output_dir, "time_gap_distribution.png")
        plt.savefig(gap_hist_path, dpi=150)
        plt.close()
        print(f"  Saved time gap distribution to: {gap_hist_path}")
    else:
        print("  Skipping time gap plot (no gaps available)")

    # Prepare summary for return
    summary = {
        "time_range": f"{min_time} to {max_time} ({total_duration / (24 * 3600):.2f} days)",
        "pct_10_plus_epochs": f"{(epochs_per_sat.filter(pl.col('epoch_count') >= 10).height / total_satellites * 100):.2f}%",
        "recommended_sequence_length": int(recommended_seq_length),
        "red_flags": [],
    }

    # Check for red flags
    if regular_satellites / total_satellites_with_gaps < 0.1:  # Less than 10% regular
        summary["red_flags"].append("Over 90% of satellites show irregular sampling")

    if max_gap > 30 * 24 * 3600:  # More than 30 days
        summary["red_flags"].append(
            f"Maximum time gap exceeds 30 days ({max_gap / (24 * 3600):.1f} days)"
        )

    if (
        epochs_per_sat.filter(pl.col("epoch_count") == 1).height
        > total_satellites * 0.5
    ):
        summary["red_flags"].append(
            "More than 50% of satellites have only a single observation"
        )

    print("\n=== SUMMARY ===")
    print(f"Time Range: {summary['time_range']}")
    print(f"% Satellites with 10+ epochs: {summary['pct_10_plus_epochs']}")
    print(
        f"Recommended sequence length: {summary['recommended_sequence_length']} epochs"
    )
    if summary["red_flags"]:
        print("Red Flags:")
        for flag in summary["red_flags"]:
            print(f"  - {flag}")
    else:
        print("No major red flags detected.")

    return summary


if __name__ == "__main__":
    main()
