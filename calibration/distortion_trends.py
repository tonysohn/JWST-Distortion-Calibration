import glob
import os
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii


def extract_metadata_and_metrics(file_path):
    """Extracts metrics, filter, and date from a master coefficient file."""
    # 1. Parse filename to extract filter and date
    basename = os.path.basename(file_path).replace(".txt", "")
    parts = basename.split("_")

    try:
        date_str = parts[-1]
        filt = parts[-2].upper()
        obs_date = datetime.strptime(date_str, "%Y%m%d")
    except (IndexError, ValueError) as e:
        print(f"Skipping {basename}: Could not parse date/filter from filename.")
        return None

    # 2. Calculate Physical Metrics
    data = ascii.read(file_path, format="csv", comment="#")

    c10_x, c10_y = data[1]["Sci2IdlX"], data[1]["Sci2IdlY"]
    c01_x, c01_y = data[2]["Sci2IdlX"], data[2]["Sci2IdlY"]

    scale_x = np.sqrt(c10_x**2 + c10_y**2)
    scale_y = np.sqrt(c01_x**2 + c01_y**2)
    rotation = np.degrees(np.arctan2(c10_y, c10_x))
    skew = np.degrees(np.arctan2(-c01_x, c01_y)) - rotation
    ho_power = np.sqrt(np.sum(data[3:]["Sci2IdlX"] ** 2 + data[3:]["Sci2IdlY"] ** 2))

    return {
        "date": obs_date,
        "label": filt,
        # Convert arcsec to milliarcsec (mas)
        "scale": ((scale_x + scale_y) / 2.0) * 1000.0,
        "rotation": rotation,
        "skew": skew,
        # Convert arcsec to microarcsec (uas)
        "ho_strength": ho_power * 1e6,
    }


def write_trend_summary(group_name, metrics_list, output_dir):
    """Generates an ASCII summary table with percent changes."""
    out_name = os.path.join(output_dir, f"trends_{group_name.lower()}_summary.txt")

    # Reference metrics are the first chronological epoch
    ref = metrics_list[0]

    def calc_pct_change(val, ref_val):
        if ref_val == 0:
            return 0.0
        return ((val - ref_val) / abs(ref_val)) * 100.0

    with open(out_name, "w") as f:
        f.write(f"Distortion Stability Summary: {group_name}\n")
        f.write("=" * 115 + "\n")

        headers = (
            f"{'Date':<12} | {'Scale (mas)':<12} | {'Scale %Chg':<10} | "
            f"{'Rot (deg)':<12} | {'Rot %Chg':<10} | {'Skew (deg)':<12} | "
            f"{'Skew %Chg':<10} | {'HO RMS (uas)':<12} | {'HO %Chg':<10}"
        )
        f.write(headers + "\n")
        f.write("-" * 115 + "\n")

        for m in metrics_list:
            date_str = m["date"].strftime("%Y-%m-%d")

            s_val, s_pct = m["scale"], calc_pct_change(m["scale"], ref["scale"])
            r_val, r_pct = (
                m["rotation"],
                calc_pct_change(m["rotation"], ref["rotation"]),
            )
            k_val, k_pct = m["skew"], calc_pct_change(m["skew"], ref["skew"])
            h_val, h_pct = (
                m["ho_strength"],
                calc_pct_change(m["ho_strength"], ref["ho_strength"]),
            )

            line = (
                f"{date_str:<12} | {s_val:<12.5f} | {s_pct:>9.4f}% | "
                f"{r_val:<12.6f} | {r_pct:>8.4f}% | {k_val:<12.6f} | "
                f"{k_pct:>8.4f}% | {h_val:<12.3f} | {h_pct:>8.4f}%"
            )
            f.write(line + "\n")

    print(f"  -> Generated summary file: {os.path.basename(out_name)}")


def plot_group_trends(group_name, metrics_list, output_dir):
    """Generates a trend plot for a specific instrument/filter group."""
    # Sort chronologically by datetime object
    metrics_list = sorted(metrics_list, key=lambda x: x["date"])
    dates = [m["date"] for m in metrics_list]

    # Write the text summary before plotting
    write_trend_summary(group_name, metrics_list, output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Distortion Stability Trends: {group_name}", fontsize=16, fontweight="bold"
    )

    def plot_sub(ax, key, title, ylabel, color):
        vals = [m[key] for m in metrics_list]

        # Plot with markers
        ax.plot(dates, vals, "o-", color=color, lw=2, markersize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        # Formatting Y-axis: Force plain numbers
        ax.ticklabel_format(useOffset=False, style="plain", axis="y")

        # Formatting X-axis as Dates
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Updated labels to reflect milliarcseconds and microarcseconds
    plot_sub(axes[0, 0], "scale", "Average Pixel Scale", "mas/pix", "royalblue")
    plot_sub(axes[0, 1], "rotation", "Detector Rotation", "degrees", "forestgreen")
    plot_sub(axes[1, 0], "skew", "Pixel Skew", "degrees", "crimson")
    plot_sub(
        axes[1, 1],
        "ho_strength",
        "Higher-Order Distortion Power",
        r"RMS ($\mu$as)",
        "purple",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save to the same directory
    out_name = os.path.join(output_dir, f"trends_{group_name.lower()}.png")
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    print(f"  -> Generated trend plot: {os.path.basename(out_name)}")
    plt.close(fig)


def main(data_dir):
    print(f"\nScanning directory: {data_dir}")
    files = sorted(glob.glob(os.path.join(data_dir, "*siaf_distortion*.txt")))

    if not files:
        print("No distortion coefficient files found.")
        return

    print(f"Found {len(files)} files. Grouping by filter...\n")

    groups = {}
    for f in files:
        m = extract_metadata_and_metrics(f)
        if m:
            if m["label"] not in groups:
                groups[m["label"]] = []
            groups[m["label"]].append(m)

    for label, metrics in groups.items():
        print(f"Processing {label} ({len(metrics)} epochs)...")
        if len(metrics) > 1:
            plot_group_trends(label, metrics, data_dir)
        else:
            print(f"  -> Skipping {label}: Need at least 2 data points for a trend.")

    print("\nTrend analysis complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate trend plots and ASCII summaries from combined SIAF files."
    )
    parser.add_argument("data_dir", help="Directory containing the master txt files")
    args = parser.parse_args()

    main(args.data_dir)
