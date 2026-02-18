"""
JWST Distortion Combination Module
Usage:
    python distortion_combine.py [--input_dir DIR] [--output_dir DIR] [--sigma 2.5]

Description:
    1. Scans for *_distortion_coeffs.txt files.
    2. Calculates a sigma-clipped mean (robust average).
    3. Generates a stability plot.
    4. Writes a master solution file with standardized naming:
       <instr>_siaf_distortion_<aper>_<filter>.txt
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii, fits
from astropy.stats import sigma_clip

# --- TRY IMPORTING DEFAULTS FROM RUN_CALIBRATION ---
try:
    import run_calibration

    # run_calibration.OUTPUT_DIR usually points to the base cal folder (e.g. ./niriss_calibration)
    # The results are inside /results subdirectory.
    DEFAULT_BASE_DIR = run_calibration.OUTPUT_DIR
    DEFAULT_DATA_DIR = run_calibration.DATA_DIR
except ImportError:
    DEFAULT_BASE_DIR = "./niriss_calibration"
    DEFAULT_DATA_DIR = "../data"

DEFAULT_INPUT_DIR = os.path.join(DEFAULT_BASE_DIR, "results")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_BASE_DIR, "results")
FILE_PATTERN = "*_distortion_coeffs.txt"


def get_metadata_from_fits(data_dir):
    """
    Scans the data directory for a FITS file to extract Instrument, Aperture, and Filter.
    Returns lowercase strings: (instr, aper, filt)
    """
    search_pattern = os.path.join(data_dir, "*.fits")
    files = sorted(glob.glob(search_pattern))

    if not files:
        print(
            f"Warning: No FITS files found in {data_dir}. Cannot determine filter name."
        )
        return "unknown", "unknown", "unknown"

    # Read the first file
    try:
        with fits.open(files[0]) as hdul:
            header = hdul[0].header
            instr = header.get("INSTRUME", "unknown").strip().lower()
            aper = (
                header.get("APERNAME", header.get("PPS_APER", "unknown"))
                .strip()
                .lower()
            )
            filt = header.get("FILTER", "unknown").strip().lower()
            return instr, aper, filt
    except Exception as e:
        print(f"Warning: Could not read header from {files[0]}: {e}")
        return "unknown", "unknown", "unknown"


def read_coefficients(file_list):
    """
    Reads all coefficient files into a 3D array.
    Shape: [N_Files, N_Coeffs, 4_Columns]
    """
    data_cube = []
    meta_data = None
    header = None

    print(f"Reading {len(file_list)} coefficient files...")

    valid_files = []
    for f in file_list:
        try:
            tab = ascii.read(f, format="csv", comment="#")

            # Columns: [Aper, siaf_index, exp_x, exp_y, S2IX, S2IY, I2SX, I2SY]
            # Indices: 0,    1,          2,     3,     4,    5,    6,    7

            row_data = []
            for row in tab:
                row_data.append([row[4], row[5], row[6], row[7]])

            data_cube.append(row_data)
            valid_files.append(f)

            if meta_data is None:
                meta_data = []
                header = tab.colnames
                for row in tab:
                    # Store metadata columns
                    meta_data.append([row[0], row[1], row[2], row[3]])

        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    return np.array(data_cube), meta_data, header


def compute_robust_mean(data_cube, sigma=2.5):
    """
    Performs sigma-clipping along the file axis (axis 0).
    """
    print(f"Computing robust mean (sigma={sigma})...")

    # Sigma clip along axis 0 (files)
    filtered_data = sigma_clip(
        data_cube, sigma=sigma, axis=0, maxiters=3, cenfunc="median", stdfunc="std"
    )

    # Compute stats
    robust_mean = np.ma.mean(filtered_data, axis=0)
    robust_std = np.ma.std(filtered_data, axis=0)
    n_surviving = np.ma.count(filtered_data, axis=0)
    std_error = robust_std / np.sqrt(n_surviving)

    return robust_mean, std_error


def plot_stability(data_cube, robust_mean, output_dir, label):
    """
    Plots the stability of coefficients.
    """
    deviations = data_cube - robust_mean

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Histogram of deviations
    dev_flat = deviations.ravel()
    # Robust range for histogram
    p1, p99 = np.percentile(dev_flat, [1, 99])

    axes[0].hist(dev_flat, bins=50, range=(p1, p99), color="steelblue", alpha=0.7)
    axes[0].set_title(f"Stability: {label}")
    axes[0].set_xlabel("Deviation from Mean (Raw Value)")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    # 2. RMS per coefficient index
    rms_per_coeff = np.sqrt(np.mean(deviations**2, axis=(0, 2)))

    axes[1].plot(rms_per_coeff, "o-", color="darkred")
    axes[1].set_title("RMS Stability by Polynomial Term")
    axes[1].set_xlabel("Coefficient Index")
    axes[1].set_ylabel("RMS Scatter")
    axes[1].grid(True, alpha=0.3)

    out_file = os.path.join(output_dir, f"{label}_stability.png")
    plt.savefig(out_file)
    plt.close(fig)
    print(f"Saved stability plot: {out_file}")


def write_master_file(mean_data, meta_data, header, output_path):
    """
    Writes the master file with fixed siaf_index formatting (01, 02...).
    """
    print(f"Writing Master Solution to {output_path}...")

    with open(output_path, "w") as f:
        f.write(f"# MASTER DISTORTION SOLUTION\n")
        f.write(f"# Generated by distortion_combine.py\n")
        f.write(f"# Sigma-clipped mean of {len(meta_data)} coefficients\n")
        f.write("#\n")

        # Write Header
        header_str = " , ".join(header)
        f.write(f"{header_str}\n")

        for i, row_meta in enumerate(meta_data):
            # meta: [Aper, idx, ex, ey]
            aper = row_meta[0]

            # Ensure siaf_index is 2 digits with leading zero
            try:
                siaf_idx = f"{int(row_meta[1]):02d}"
            except:
                siaf_idx = str(row_meta[1])  # Fallback if not integer-like

            ex = row_meta[2]
            ey = row_meta[3]

            line = (
                f" {aper:7s} , {siaf_idx:<3s} , {str(ex):>10s} , {str(ey):>10s} , "
                f"{mean_data[i, 0]:23.12e} , {mean_data[i, 1]:23.12e} , "
                f"{mean_data[i, 2]:23.12e} , {mean_data[i, 3]:23.12e}\n"
            )
            f.write(line)


def main():
    parser = argparse.ArgumentParser(
        description="Combine JWST distortion coefficients."
    )
    parser.add_argument(
        "--input_dir", default=DEFAULT_INPUT_DIR, help="Dir with coef files"
    )
    parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
        help="Dir with FITS files (for header info)",
    )
    parser.add_argument(
        "--output_dir", default=DEFAULT_OUTPUT_DIR, help="Dir for master output"
    )
    parser.add_argument("--sigma", type=float, default=2.5, help="Sigma clip threshold")

    args = parser.parse_args()

    # 1. Find Files
    search_path = os.path.join(args.input_dir, FILE_PATTERN)
    files = sorted(glob.glob(search_path))
    # Filter out existing master files
    files = [f for f in files if "siaf_distortion" not in f and "MASTER" not in f]

    if not files:
        print(f"No coefficient files found in {args.input_dir}")
        return

    # 2. Determine Master Filename
    instr, aper, filt = get_metadata_from_fits(args.data_dir)

    if "fgs" in instr:
        # FGS Format: fgs_siaf_distortion_<aper>.txt
        master_name = f"{instr}_siaf_distortion_{aper}.txt"
    else:
        # NIRISS Format: niriss_siaf_distortion_<aper>_<filt>.txt
        master_name = f"{instr}_siaf_distortion_{aper}_{filt}.txt"

    # 3. Process
    data_cube, meta_data, header = read_coefficients(files)
    if data_cube.size == 0:
        return

    robust_mean, std_error = compute_robust_mean(data_cube, sigma=args.sigma)

    # 4. Save
    output_path = os.path.join(args.output_dir, master_name)
    write_master_file(robust_mean, meta_data, header, output_path)

    # 5. Plot
    plot_label = master_name.replace(".txt", "")
    plot_stability(data_cube, robust_mean, args.output_dir, plot_label)

    print("\nCombination Complete.")


if __name__ == "__main__":
    main()
