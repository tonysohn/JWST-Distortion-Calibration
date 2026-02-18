"""
JWST Smart Catalog Matching
Updates:
- Restored 'extract_alignment_offset' helper function.
- Implements SAFE IDEAL FRAME matching (decoupled from pysiaf projection).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree


def match_with_pointing_prior(
    obs_cat: Table,
    ref_cat: Table,
    n_bright_obs: int = 200,
    ref_mag_bins: Optional[List[Tuple[float, float]]] = None,
    pos_tolerance_arcsec: float = 0.5,
    min_matches: int = 50,
    verbose: bool = True,
    **kwargs,
) -> Tuple[Table, Table, Dict]:
    """
    Stage 1: Bootstrap Match.
    Uses bright stars to find the initial RA/Dec pointing offset.
    """
    # Use N brightest observed stars
    obs_bright = obs_cat[:n_bright_obs]

    # Auto-search magnitude bins if not provided
    if ref_mag_bins is None:
        obs_mag_min = obs_bright["mag_ab"].min()
        obs_mag_max = obs_bright["mag_ab"].max()
        best_matches = 0
        best_bin = None
        best_obs_m = None
        best_ref_m = None

        # Start from faint end and search upward
        for mag_start in np.arange(obs_mag_max + 5, obs_mag_min - 2, -0.5):
            mag_bin = (mag_start, mag_start + 3.0)
            ref_in_bin = ref_cat[
                (ref_cat["mag_ref"] >= mag_bin[0]) & (ref_cat["mag_ref"] < mag_bin[1])
            ]
            if len(ref_in_bin) < min_matches:
                continue

            obs_m, ref_m = _positional_match(
                obs_bright, ref_in_bin, pos_tolerance_arcsec
            )

            if len(obs_m) > best_matches:
                best_matches = len(obs_m)
                best_bin = mag_bin
                best_obs_m = obs_m
                best_ref_m = ref_m
                if len(obs_m) >= 0.8 * n_bright_obs:
                    break

        obs_matched = best_obs_m if best_obs_m is not None else Table()
        ref_matched = best_ref_m if best_ref_m is not None else Table()
        successful_bin = best_bin
    else:
        # Use provided bins
        all_obs = []
        all_ref = []
        for mag_bin in ref_mag_bins:
            ref_in_bin = ref_cat[
                (ref_cat["mag_ref"] >= mag_bin[0]) & (ref_cat["mag_ref"] < mag_bin[1])
            ]
            o, r = _positional_match(obs_bright, ref_in_bin, pos_tolerance_arcsec)
            all_obs.append(o)
            all_ref.append(r)

        from astropy.table import vstack

        obs_matched = vstack(all_obs) if all_obs else Table()
        ref_matched = vstack(all_ref) if all_ref else Table()
        successful_bin = ref_mag_bins[0] if ref_mag_bins else None

    # Stats
    n_matches = len(obs_matched)
    info = {
        "n_matches": n_matches,
        "match_fraction": n_matches / len(obs_bright) if len(obs_bright) > 0 else 0,
        "mag_bin": successful_bin,
    }
    return obs_matched, ref_matched, info


def _positional_match(obs, ref, tol_arcsec):
    if len(obs) == 0 or len(ref) == 0:
        return obs[:0], ref[:0]
    tol_deg = tol_arcsec / 3600.0
    ref_coords = np.column_stack([ref["ra"], ref["dec"]])
    obs_coords = np.column_stack([obs["ra"], obs["dec"]])
    tree = cKDTree(ref_coords)
    dist, idx = tree.query(obs_coords, distance_upper_bound=tol_deg)
    valid = dist < tol_deg
    return obs[valid], ref[idx[valid]]


# =============================================================================
# STAGE 2: IDEAL FRAME MATCHING
# =============================================================================


def extract_alignment_offset(stage1_obs, stage1_ref, aperture):
    """
    Calculate the offset between Observed Ideal and Reference Ideal
    using the bright stars matched in Stage 1.
    """
    # Project Observed Pixels -> Ideal (using current SIAF aperture)
    # Note: stage1_obs already has x_SCI/y_SCI columns
    x_sci = stage1_obs["x_SCI"] - aperture.XSciRef
    y_sci = stage1_obs["y_SCI"] - aperture.YSciRef

    # Use standard SIAF projection for this step (it's approximate anyway)
    # We just need the bulk shift.
    x_idl_obs, y_idl_obs = aperture.sci_to_idl(stage1_obs["x_SCI"], stage1_obs["y_SCI"])

    # Reference Ideal (already computed in pipeline)
    x_idl_ref = stage1_ref["x_idl"]
    y_idl_ref = stage1_ref["y_idl"]

    # Calculate median offset (Pointing Error in Ideal Frame)
    dx = np.median(x_idl_ref - x_idl_obs)
    dy = np.median(y_idl_ref - y_idl_obs)

    return dx, dy


def match_in_ideal_frame(
    obs_cat: Table,
    ref_cat: Table,
    x_idl_obs: np.ndarray,  # Pre-projected Observed Ideal X
    y_idl_obs: np.ndarray,  # Pre-projected Observed Ideal Y
    pos_tolerance_arcsec: float = 0.1,
    verbose: bool = True,
) -> Tuple[Table, Table, Dict]:
    """
    Matches catalogs in the IDEAL coordinate frame.
    Requires pre-calculated Ideal coordinates for the observed catalog.
    """
    if verbose:
        print(f'  Full Match in IDEAL frame (Tol: {pos_tolerance_arcsec}")...')

    # Match using KDTree on Ideal Coordinates (arcsec)
    obs_coords = np.column_stack([x_idl_obs, y_idl_obs])
    ref_coords = np.column_stack([ref_cat["x_idl"], ref_cat["y_idl"]])

    tree = cKDTree(ref_coords)
    dist, idx = tree.query(obs_coords, distance_upper_bound=pos_tolerance_arcsec)

    valid = dist < pos_tolerance_arcsec

    obs_matched = obs_cat[valid]
    ref_matched = ref_cat[idx[valid]]

    # Stats
    n = len(obs_matched)
    rms = np.std(dist[valid]) if n > 0 else 0.0

    if verbose:
        print(f"    Matched {n} stars (RMS {rms * 1000:.1f} mas)")

    return obs_matched, ref_matched, {"n_matches": n}
