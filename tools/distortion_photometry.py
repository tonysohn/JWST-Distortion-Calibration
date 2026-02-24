"""
JWST Distortion Photometry Module

Provides source detection and centroiding using photutils.
Used primarily for FGS where jwst1pass xymq files are not available.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import simple_norm
from photutils.aperture import CircularAperture
from photutils.background import MADStdBackgroundRMS, MMMBackground
from photutils.centroids import centroid_2dg, centroid_sources
from photutils.detection import IRAFStarFinder


def measure_sources_photutils(
    fits_file: str,
    instrument: str = "FGS",
    plot_dir: str = None,
    save_plot: bool = True,
) -> Table:
    """
    Detect and centroid sources in a FITS image using photutils.

    Parameters
    ----------
    fits_file : str
        Path to the FITS file (typically _cal.fits)
    instrument : str
        Instrument name ('FGS', 'NIRISS', 'NIRCAM')
    plot_dir : str
        Directory to save diagnostic plot. If None, uses FITS file directory.

    Returns
    -------
    catalog : Table
        Astropy table with columns ['x', 'y', 'm', 'q', 'flux', 'snr']
    """
    print(f"  Performing source extraction on: {os.path.basename(fits_file)}")

    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        header = hdul[1].header

        # Convert MJy/sr to DN/s (CPS)
        photmjsr = header.get("PHOTMJSR", 1.0)
        data_cps = data / photmjsr

    # --- Set Detection Parameters ---
    inst = instrument.upper()

    if "FGS" in inst:
        sigma_factor = 3.0
        fwhm = 1.55
        sharp_lo, sharp_hi = 0.7, 1.4
        round_lo, round_hi = 0.0, 0.4
        minsep_fwhm = 7.0
    elif "NIRISS" in inst:
        sigma_factor = 3.0
        fwhm = 1.5
        sharp_lo, sharp_hi = 0.7, 1.3
        round_lo, round_hi = 0.0, 0.4
        minsep_fwhm = 5.0
    else:
        sigma_factor = 3.0
        fwhm = 1.75
        sharp_lo, sharp_hi = 0.7, 1.3
        round_lo, round_hi = 0.0, 0.4
        minsep_fwhm = 5.0

    print(f"    Params: FWHM={fwhm}, Sigma={sigma_factor}")

    # --- Background Estimation ---
    bkg_rms_estimator = MADStdBackgroundRMS()
    bkg_estimator = MMMBackground()

    bkg_rms = bkg_rms_estimator(data_cps)
    bkg_level = bkg_estimator(data_cps)
    threshold = bkg_level + (sigma_factor * bkg_rms)

    # --- Detection ---
    finder = IRAFStarFinder(
        threshold=threshold,
        fwhm=fwhm,
        minsep_fwhm=minsep_fwhm,
        roundlo=round_lo,
        roundhi=round_hi,
        sharplo=sharp_lo,
        sharphi=sharp_hi,
    )

    sources = finder(data_cps)

    if sources is None or len(sources) == 0:
        print("    WARNING: No sources detected!")
        return Table(names=("x", "y", "m", "q", "flux", "snr"))

    # --- Refine Centroids ---
    x_init = sources["xcentroid"]
    y_init = sources["ycentroid"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_new, y_new = centroid_sources(
            data_cps,
            x_init,
            y_init,
            box_size=int(fwhm * 4 + 1),
            centroid_func=centroid_2dg,
        )

    # --- Format Output ---
    flux = sources["flux"]

    # Filter negative flux
    valid = flux > 0
    flux = flux[valid]
    x_new = x_new[valid]
    y_new = y_new[valid]

    # Instrumental Mag
    mag_inst = -2.5 * np.log10(flux)

    # Dummy Quality 'q'
    q_dummy = np.full_like(x_new, 0.01)

    # Approx SNR
    snr = np.sqrt(flux)

    catalog = Table()
    catalog["x"] = x_new + 1.0  # 1-based
    catalog["y"] = y_new + 1.0  # 1-based
    catalog["m"] = mag_inst
    catalog["q"] = q_dummy
    catalog["flux"] = flux
    catalog["snr"] = snr

    print(f"    Extracted {len(catalog)} sources.")

    # --- Diagnostic Plot ---
    if save_plot:
        if plot_dir is None:
            plot_dir = os.path.dirname(fits_file)

        plot_name = os.path.basename(fits_file).replace(
            ".fits", "_detected_sources.pdf"
        )
        plot_path = os.path.join(plot_dir, plot_name)

        plt.ioff()  # Turn off interactive mode
        fig, ax = plt.subplots(figsize=(12, 12))

        norm = simple_norm(data_cps, "sqrt", percent=99.0)
        ax.imshow(data_cps, norm=norm, cmap="Greys", origin="lower")

        # Overlay circles
        # Photutils apertures are 0-based, so subtract 1 from our 1-based catalog
        positions = np.transpose((catalog["x"] - 1, catalog["y"] - 1))
        apertures = CircularAperture(positions, r=10)  # Large circles to see them
        apertures.plot(color="blue", lw=1.5, alpha=0.5, axes=ax)

        ax.set_title(f"Detected Sources: {len(catalog)}")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"    Diagnostic plot saved: {plot_path}")

    return catalog


def save_xymq_file(catalog: Table, filename: str):
    """Save catalog to XYMQ format (x y m q)."""
    with open(filename, "w") as f:
        f.write("# x y m q\n")
        for row in catalog:
            f.write(f"{row['x']:.3f}  {row['y']:.3f}  {row['m']:.4f}  {row['q']:.3f}\n")
    print(f"    Saved XYMQ to: {filename}")


if __name__ == "__main__":
    import argparse
    import glob

    from astropy.io import fits

    parser = argparse.ArgumentParser(
        description="Run source extraction on a directory of FITS files."
    )
    parser.add_argument(
        "data_dir", type=str, help="Directory containing FITS files to process."
    )
    args = parser.parse_args()

    search_pattern = os.path.join(args.data_dir, "*.fits")
    fits_files = sorted(glob.glob(search_pattern))

    if not fits_files:
        print(f"No FITS files found in {args.data_dir}")
    else:
        print(f"Found {len(fits_files)} FITS files to process in {args.data_dir}.")
        for f in fits_files:
            try:
                # Automatically determine instrument from the FITS header
                with fits.open(f) as hdul:
                    detected_inst = (
                        hdul[0].header.get("INSTRUME", "FGS").strip().upper()
                    )

                print(f"  Detected Instrument: {detected_inst}")

                # Run the extraction
                catalog = measure_sources_photutils(
                    f, instrument=detected_inst, save_plot=True
                )

                # Save the results to an .xymq file
                if len(catalog) > 0:
                    out_file = f.replace(".fits", ".xymq")
                    save_xymq_file(catalog, out_file)
            except Exception as e:
                print(f"Error processing {f}: {e}")
