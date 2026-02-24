import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from astropy.time import Time

def extract_metadata_and_metrics(file_path):
    """Extracts metrics, filter, and date from a master coefficient file."""
    data = ascii.read(file_path, format='csv', comment='#')
    
    # 1. Extract metadata from header comments
    obs_date = None
    filt = "unknown"
    aper = "unknown"
    with open(file_path, 'r') as f:
        for line in f:
            if "Observation Date:" in line:
                obs_date = line.split(":")[-1].strip()
            if "Filter/Pupil:" in line:
                filt = line.split(":")[-1].strip()
            if "Aperture:" in line:
                aper = line.split(":")[-1].strip()

    if not obs_date or obs_date == "unknown":
        return None

    # 2. Calculate Physical Metrics
    c10_x, c10_y = data[1]['Sci2IdlX'], data[1]['Sci2IdlY']
    c01_x, c01_y = data[2]['Sci2IdlX'], data[2]['Sci2IdlY']
    
    scale_x = np.sqrt(c10_x**2 + c10_y**2)
    scale_y = np.sqrt(c01_x**2 + c01_y**2)
    rotation = np.degrees(np.arctan2(c10_y, c10_x))
    skew = np.degrees(np.arctan2(-c01_x, c01_y)) - rotation
    ho_power = np.sqrt(np.sum(data[3:]['Sci2IdlX']**2 + data[3:]['Sci2IdlY']**2))
    
    return {
        "epoch": Time(obs_date).decimalyear,
        "label": f"{aper}_{filt}",
        "scale": (scale_x + scale_y) / 2.0,
        "rotation": rotation,
        "skew": skew,
        "ho_strength": ho_power
    }

def plot_group_trends(group_name, metrics_list):
    """Generates a trend plot for a specific instrument/filter group."""
    # Sort by epoch
    metrics_list = sorted(metrics_list, key=lambda x: x['epoch'])
    epochs = [m['epoch'] for m in metrics_list]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Distortion Stability Trends: {group_name}", fontsize=16)

    def plot_sub(ax, key, title, ylabel, color):
        vals = [m[key] for m in metrics_list]
        ax.plot(epochs, vals, 'o-', color=color, lw=2, markersize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        # Handle small scale variations
        ax.ticklabel_format(useOffset=False, style='plain' if "scale" not in key else 'sci')

    plot_sub(axes[0,0], "scale", "Average Pixel Scale", "arcsec/pix", "royalblue")
    plot_sub(axes[0,1], "rotation", "Detector Rotation", "degrees", "forestgreen")
    plot_sub(axes[1,0], "skew", "Pixel Skew", "degrees", "crimson")
    plot_sub(axes[1,1], "ho_strength", "Higher-Order Distortion Power", "RMS (arcsec)", "purple")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_name = f"trends_{group_name.replace(' ', '_')}.png"
    plt.savefig(out_name, dpi=150)
    print(f"Generated trend plot: {out_name}")
    plt.close(fig)

def main(search_dir):
    # Find all master files recursively
    files = glob.glob(os.path.join(search_dir, "**/results/*siaf_distortion_*.txt"), recursive=True)
    
    groups = {}
    for f in files:
        m = extract_metadata_and_metrics(f)
        if m:
            if m['label'] not in groups:
                groups[m['label']] = []
            groups[m['label']].append(m)

    for label, metrics in groups.items():
        if len(metrics) > 1:
            plot_group_trends(label, metrics)
        else:
            print(f"Skipping {label}: Only 1 data point found.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Root directory to search for master results")
    args = parser.parse_args()
    main(args.data_dir)
