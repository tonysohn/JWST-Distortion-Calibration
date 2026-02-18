"""
JWST Distortion Calibration Package
-----------------------------------
A robust tool for deriving polynomial distortion solutions for JWST instruments
(NIRISS, FGS) using reference catalogs.

Modules:
    run_calibration: Batch processing script.
    distortion_combine: Master solution generator.
    distortion_pipeline: Core pipeline controller.
"""

from .distortion_pipeline import DistortionPipeline, PipelineConfig
from .distortion_core import DistortionFitter, PolynomialDistortion
from .distortion_data import prepare_obs_catalog, prepare_ref_catalog

__version__ = "1.0.0"
__author__ = "S. T. Sohn"
