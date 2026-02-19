"""
JWST Distortion Core Logic (v8: 2D Sigma Rejection)
Updates:
- Final outlier rejection uses Radial Distance (R) instead of per-axis bounds.
- Keeps the "cloud" circular and statistically consistent.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pysiaf
from astropy.stats import mad_std, sigma_clip


class PolynomialDistortion:
    def __init__(self, degree: int):
        self.degree = degree
        self.n_coeffs = (degree + 1) * (degree + 2) // 2

    def build_design_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_points = len(x)
        A = np.zeros((n_points, self.n_coeffs))
        idx = 0
        for i in range(self.degree + 1):
            for j in range(i + 1):
                A[:, idx] = (x ** (i - j)) * (y**j)
                idx += 1
        return A

    def evaluate(self, coeffs: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        A = self.build_design_matrix(x, y)
        return A @ coeffs

    def fit_robust(
        self,
        x: np.ndarray,
        y: np.ndarray,
        target: np.ndarray,
        scale: float,
        sigma: float = 2.5,
        max_iters: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.ones(len(x), dtype=bool)
        coeffs = np.zeros(self.n_coeffs)
        if len(x) < self.n_coeffs + 5:
            return coeffs, mask
        x_norm = x / scale
        y_norm = y / scale
        for _ in range(max_iters):
            A = self.build_design_matrix(x_norm[mask], y_norm[mask])
            t = target[mask]
            try:
                coeffs_norm, _, _, _ = np.linalg.lstsq(A, t, rcond=None)
            except np.linalg.LinAlgError:
                break
            model = self.evaluate(coeffs_norm, x_norm, y_norm)
            residuals = target - model
            clipped = sigma_clip(
                residuals[mask],
                sigma=sigma,
                maxiters=1,
                cenfunc="median",
                stdfunc=mad_std,
            )
            new_inliers = np.where(mask)[0][~clipped.mask]
            new_mask = np.zeros_like(mask)
            new_mask[new_inliers] = True
            if np.array_equal(mask, new_mask):
                break
            mask = new_mask
        coeffs = np.zeros_like(coeffs_norm)
        idx = 0
        for i in range(self.degree + 1):
            factor = scale**i
            for j in range(i + 1):
                coeffs[idx] = coeffs_norm[idx] / factor
                idx += 1
        return coeffs, mask

    def fit_affine_4param_weighted(self, x_in, y_in, x_out, y_out, weights=None):
        n = len(x_in)
        M = np.zeros((2 * n, 4))
        target = np.zeros(2 * n)
        M[0::2, 0] = x_in
        M[0::2, 1] = -y_in
        M[0::2, 2] = 1.0
        M[1::2, 0] = y_in
        M[1::2, 1] = x_in
        M[1::2, 3] = 1.0
        target[0::2] = x_out
        target[1::2] = y_out
        w_use = np.ones(2 * n)
        if weights is not None:
            w_use = np.repeat(weights, 2)
        mask = np.ones(n, dtype=bool)
        mask_full = np.ones(2 * n, dtype=bool)
        params = np.zeros(4)
        for _ in range(3):
            mask_full[0::2] = mask
            mask_full[1::2] = mask
            W_sqrt = np.sqrt(w_use[mask_full])
            M_w = M[mask_full] * W_sqrt[:, None]
            t_w = target[mask_full] * W_sqrt
            try:
                params, _, _, _ = np.linalg.lstsq(M_w, t_w, rcond=None)
            except np.linalg.LinAlgError:
                break
            model = M @ params
            res_sq = (target - model) ** 2
            res_mag = np.sqrt(res_sq[0::2] + res_sq[1::2])
            sig = mad_std(res_mag[mask])
            new_mask = res_mag < 3.0 * sig
            if np.sum(new_mask) < 10:
                break
            mask = new_mask
        return params

    def combine_linear_and_poly(self, lin_coeffs, poly_coeffs_x, poly_coeffs_y):
        combined = np.zeros_like(poly_coeffs_x)
        combined[0] += lin_coeffs[0]
        combined[1] += lin_coeffs[1]
        combined[2] += lin_coeffs[2]
        combined += lin_coeffs[1] * poly_coeffs_x
        combined += lin_coeffs[2] * poly_coeffs_y
        return combined

    def compute_inverse_grid(
        self, fwd_coeffs_x, fwd_coeffs_y, xlim, ylim, scale, grid_density=50
    ):
        x = np.linspace(xlim[0], xlim[1], grid_density)
        y = np.linspace(ylim[0], ylim[1], grid_density)
        xx, yy = np.meshgrid(x, y)
        x_flat, y_flat = xx.ravel(), yy.ravel()
        x_idl = self.evaluate(fwd_coeffs_x, x_flat, y_flat)
        y_idl = self.evaluate(fwd_coeffs_y, x_flat, y_flat)
        A, B, Tx, Ty = self.fit_affine_4param_weighted(x_idl, y_idl, x_flat, y_flat)
        x_std = A * x_idl - B * y_idl + Tx
        y_std = B * x_idl + A * y_idl + Ty
        dx = x_flat - x_std
        dy = y_flat - y_std
        px, _ = self.fit_robust(x_idl, y_idl, dx, scale=scale, sigma=10.0, max_iters=2)
        py, _ = self.fit_robust(x_idl, y_idl, dy, scale=scale, sigma=10.0, max_iters=2)

        inv_coeffs_x = px.copy()
        inv_coeffs_x[0] += Tx
        inv_coeffs_x[1] += A
        inv_coeffs_x[2] -= B

        inv_coeffs_y = py.copy()
        inv_coeffs_y[0] += Ty
        inv_coeffs_y[1] += B
        inv_coeffs_y[2] += A

        # Force origin intercept to 0.0 per standard SIAF conventions
        inv_coeffs_x[0] = 0.0
        inv_coeffs_y[0] = 0.0

        return inv_coeffs_x, inv_coeffs_y


class DistortionFitter:
    def __init__(self, poly_degree: int, sigma_fit: float):
        self.poly_degree = poly_degree
        self.sigma_fit = sigma_fit
        self.poly = PolynomialDistortion(poly_degree)

    def fit_distortion_step(
        self,
        x_sci: np.ndarray,
        y_sci: np.ndarray,
        x_idl: np.ndarray,
        y_idl: np.ndarray,
        aperture_params: Optional[Dict] = None,
        prior_coeffs: Optional[Dict[str, np.ndarray]] = None,
        weights: Optional[np.ndarray] = None,
        damping_factor: float = 0.75,
        use_grid: bool = False,
        grid_size: int = 20,
    ) -> Dict:

        if aperture_params is None:
            aperture_params = {}
        xlim = aperture_params.get("xlim", [0, 2048])
        ylim = aperture_params.get("ylim", [0, 2048])
        width = xlim[1] - xlim[0]
        fit_scale = max(width, ylim[1] - ylim[0]) if width > 0 else 2048.0

        # 1. Align (Bright Stars)
        N_ALIGN = 400
        if weights is not None and len(weights) > N_ALIGN:
            align_idx = np.argsort(weights)[-N_ALIGN:]
        else:
            align_idx = np.arange(len(x_sci))
        A, B, Tx, Ty = self.poly.fit_affine_4param_weighted(
            x_idl[align_idx],
            y_idl[align_idx],
            x_sci[align_idx],
            y_sci[align_idx],
            weights=weights[align_idx] if weights is not None else None,
        )

        # 2. Residuals (All Stars)
        x_std = A * x_idl - B * y_idl + Tx
        y_std = B * x_idl + A * y_idl + Ty
        dx_dist = x_std - x_sci
        dy_dist = y_std - y_sci

        # 3. Fit Poly
        px, mask_fit_x = self.poly.fit_robust(
            x_sci, y_sci, dx_dist, scale=fit_scale, sigma=self.sigma_fit
        )
        py, mask_fit_y = self.poly.fit_robust(
            x_sci, y_sci, dy_dist, scale=fit_scale, sigma=self.sigma_fit
        )

        # 4. Combine
        det = A**2 + B**2
        inv_A = A / det
        inv_B = B / det
        lx0 = -inv_A * Tx - inv_B * Ty
        lx1 = inv_A
        lx2 = inv_B
        ly0 = inv_B * Tx - inv_A * Ty
        ly1 = -inv_B
        ly2 = inv_A
        L_inv_x = np.array([lx0, lx1, lx2])
        L_inv_y = np.array([ly0, ly1, ly2])
        cx = self.poly.combine_linear_and_poly(L_inv_x, px, py)
        cy = self.poly.combine_linear_and_poly(L_inv_y, px, py)

        if prior_coeffs is not None and damping_factor < 1.0:
            prev_cx = prior_coeffs.get("Sci2IdlX_Raw")
            prev_cy = prior_coeffs.get("Sci2IdlY_Raw")
            if prev_cx is not None:
                cx = prev_cx + damping_factor * (cx - prev_cx)
                cy = prev_cy + damping_factor * (cy - prev_cy)

        cx[0] = 0.0
        cy[0] = 0.0
        cx_clean, cy_clean = self._clean_siaf_coeffs(cx, cy)
        inv_cx_clean, inv_cy_clean = self.poly.compute_inverse_grid(
            cx_clean, cy_clean, xlim=xlim, ylim=ylim, scale=fit_scale
        )

        # 5. Clean Stats with 2D REJECTION
        x_model = self.poly.evaluate(cx, x_sci, y_sci)
        y_model = self.poly.evaluate(cy, x_sci, y_sci)

        res_x = (x_idl - x_model) * 1000.0
        res_y = (y_idl - y_model) * 1000.0
        res_x -= np.median(res_x)
        res_y -= np.median(res_y)

        # Calculate Radial Residual
        res_r = np.sqrt(res_x**2 + res_y**2)

        # Robust Radial Sigma (approximated from MAD)
        sigma_r = mad_std(res_r)

        # 2D Outlier Rejection (Keep R < 3.5 * sigma)
        final_mask = res_r < (3.5 * sigma_r)

        # Re-calc stats on clean set
        rms_x = np.std(res_x[final_mask])
        rms_y = np.std(res_y[final_mask])

        return {
            "Sci2IdlX": cx_clean,
            "Sci2IdlY": cy_clean,
            "Sci2IdlX_Raw": cx,
            "Sci2IdlY_Raw": cy,
            "Idl2SciX": inv_cx_clean,
            "Idl2SciY": inv_cy_clean,
            "rms_x": rms_x,
            "rms_y": rms_y,
            "n_stars": np.sum(final_mask),
            "mask": final_mask,
            # Filtered Output for Plotting
            "residuals_x_mas": res_x[final_mask],
            "residuals_y_mas": res_y[final_mask],
            "x_sci_used": x_sci[final_mask],
            "y_sci_used": y_sci[final_mask],
            "grid_debug": {
                "x": x_sci,
                "y": y_sci,
                "dx": dx_dist,
                "dy": dy_dist,
                "x_edges": [],
                "y_edges": [],
            },
        }

    def _clean_siaf_coeffs(
        self, cx: np.ndarray, cy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        cx_wk, cy_wk = cx.copy(), cy.copy()
        cx_wk[0], cy_wk[0] = 0.0, 0.0
        try:
            temp_ap = pysiaf.Aperture()
            temp_ap._polynomial_degree = self.poly_degree
            idx = 0
            for i in range(self.poly_degree + 1):
                for j in range(i + 1):
                    setattr(temp_ap, f"Sci2IdlX{i - j}{j}", cx_wk[idx])
                    setattr(temp_ap, f"Sci2IdlY{i - j}{j}", cy_wk[idx])
                    idx += 1
            rotation_deg = temp_ap.get_polynomial_linear_parameters()["rotation_y"]
        except Exception:
            rotation_deg = 0.0
            if abs(cy_wk[2]) > 1e-10:
                rotation_deg = np.degrees(np.arctan2(cy_wk[1], cy_wk[2]))
        if abs(rotation_deg) > 1e-5:
            cx_clean, cy_clean = pysiaf.polynomial.add_rotation(
                cx_wk, cy_wk, -rotation_deg
            )
        else:
            cx_clean, cy_clean = cx_wk, cy_wk
        cx_clean[0] = 0.0
        cy_clean[0] = 0.0
        return cx_clean, cy_clean
