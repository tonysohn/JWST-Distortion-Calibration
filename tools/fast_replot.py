import glob
import os
import sys

import numpy as np
import pysiaf
from astropy.io import ascii, fits

# This automatically adds your JWST-Distortion-Calibration root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from calibration.distortion_core import PolynomialDistortion
from calibration.distortion_data import prepare_obs_catalog, prepare_ref_catalog
from calibration.distortion_matching_smart import match_in_ideal_frame
from calibration.distortion_plotting import plot_residuals, plot_trends

# ==========================================
# PATH CONFIGURATION
# ==========================================
REF_CAT = os.path.expanduser(
    "~/JWST/JWST-Distortion-Calibration/reference_catalogs/lmc_calibration_field_hst_2017p38_jwstmags.fits"
)

CURRENT_DIR = os.getcwd()
CALIB_DIR = os.path.join(CURRENT_DIR, "calibration")
RESULTS_DIR = os.path.join(CALIB_DIR, "results")
PLOTS_DIR = os.path.join(CALIB_DIR, "plots")
# ==========================================


def find_source_file(filename):
    if os.path.exists(os.path.join(CURRENT_DIR, filename)):
        return os.path.join(CURRENT_DIR, filename)
    elif os.path.exists(os.path.join(CURRENT_DIR, "..", filename)):
        return os.path.abspath(os.path.join(CURRENT_DIR, "..", filename))
    return None


def fast_replot():
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Could not find '{RESULTS_DIR}'")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    coeff_files = glob.glob(os.path.join(RESULTS_DIR, "*_distortion_coeffs.txt"))

    if not coeff_files:
        print(f"No coefficient files found in {RESULTS_DIR}")
        return

    for coeff_file in coeff_files:
        base_name = os.path.basename(coeff_file).replace("_distortion_coeffs.txt", "")

        if base_name.endswith("_NIS_CEN"):
            file_root, inst_name, aper_name, poly_deg = (
                base_name.replace("_NIS_CEN", ""),
                "NIRISS",
                "NIS_CEN",
                5,
            )
        elif base_name.endswith("_FGS1_FULL"):
            file_root, inst_name, aper_name, poly_deg = (
                base_name.replace("_FGS1_FULL", ""),
                "FGS",
                "FGS1_FULL",
                4,
            )
        elif base_name.endswith("_FGS2_FULL"):
            file_root, inst_name, aper_name, poly_deg = (
                base_name.replace("_FGS2_FULL", ""),
                "FGS",
                "FGS2_FULL",
                4,
            )
        else:
            continue

        poly = PolynomialDistortion(degree=poly_deg)
        siaf = pysiaf.Siaf(inst_name)
        aperture = siaf[aper_name]
        xref, yref = aperture.XSciRef, aperture.YSciRef

        fits_file = find_source_file(f"{file_root}.fits") or find_source_file(
            f"{file_root}_cal.fits"
        )
        xymq_file = find_source_file(f"{file_root}.xymq") or (
            fits_file.replace(".fits", ".xymq") if fits_file else None
        )

        if not fits_file or not xymq_file or not os.path.exists(xymq_file):
            print(f"Skipping {file_root}: Missing FITS or .xymq")
            continue

        print(f"Fast replotting: {file_root} ({aper_name})")

        tab = ascii.read(coeff_file, format="basic", comment="#")
        cx, cy = tab["Sci2IdlX"].data, tab["Sci2IdlY"].data

        obs_cat = prepare_obs_catalog(xymq_file, fits_file)
        ref_cat = prepare_ref_catalog(REF_CAT, obs_cat)

        x_sci_cen = obs_cat["x_SCI"] - xref
        y_sci_cen = obs_cat["y_SCI"] - yref
        x_idl_obs = poly.evaluate(cx, x_sci_cen, y_sci_cen)
        y_idl_obs = poly.evaluate(cy, x_sci_cen, y_sci_cen)

        with fits.open(fits_file) as hdul:
            hdr = hdul[1].header
            ra_ref = hdr.get("CRVAL1", np.mean(obs_cat["ra"]))
            dec_ref = hdr.get("CRVAL2", np.mean(obs_cat["dec"]))
            roll_ref = hdr.get("ROLL_REF", hdr.get("PA_V3", 0.0))

        att = pysiaf.utils.rotations.attitude(
            aperture.V2Ref, aperture.V3Ref, ra_ref, dec_ref, roll_ref
        )
        v2, v3 = pysiaf.utils.rotations.getv2v3(att, ref_cat["ra"], ref_cat["dec"])
        x_ref_idl, y_ref_idl = aperture.tel_to_idl(v2, v3)
        ref_cat["x_idl"] = x_ref_idl / obs_cat.meta.get("va_scale", 1.0)
        ref_cat["y_idl"] = y_ref_idl / obs_cat.meta.get("va_scale", 1.0)

        obs_m, ref_m, _ = match_in_ideal_frame(
            obs_cat,
            ref_cat,
            x_idl_obs,
            y_idl_obs,
            pos_tolerance_arcsec=0.1,
            verbose=False,
        )

        if len(obs_m) < 10:
            print("  -> Failed to match enough stars.")
            continue

        x_sci_m = obs_m["x_SCI"] - xref
        y_sci_m = obs_m["y_SCI"] - yref
        x_model = poly.evaluate(cx, x_sci_m, y_sci_m)
        y_model = poly.evaluate(cy, x_sci_m, y_sci_m)

        # Align the Reference Catalog to the Rotated Model Frame
        A, B, Tx, Ty = poly.fit_affine_4param_weighted(
            ref_m["x_idl"], ref_m["y_idl"], x_model, y_model
        )
        x_ref_aligned = A * ref_m["x_idl"] - B * ref_m["y_idl"] + Tx
        y_ref_aligned = B * ref_m["x_idl"] + A * ref_m["y_idl"] + Ty

        res_x_mas = (x_ref_aligned - x_model) * 1000.0
        res_y_mas = (y_ref_aligned - y_model) * 1000.0

        # Populate results dictionary with required metrics for both plotting routines
        results = {
            "x_sci_used": obs_m["x_SCI"],
            "y_sci_used": obs_m["y_SCI"],
            "residuals_x_mas": res_x_mas,
            "residuals_y_mas": res_y_mas,
            "rms_x": np.std(res_x_mas),
            "rms_y": np.std(res_y_mas),
            "n_stars": len(res_x_mas),
        }

        label_out = f"{file_root}_{aper_name}"
        plot_trends(results, output_dir=PLOTS_DIR, label=label_out)
        plot_residuals(results, output_dir=PLOTS_DIR, label=label_out)
        print("  -> Generated updated trends and residuals.")


if __name__ == "__main__":
    fast_replot()
