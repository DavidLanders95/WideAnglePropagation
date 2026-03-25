"""
Wide-Angle Propagation Package

This package provides implementations of various wave propagation methods
for electron microscopy simulations, including Fresnel, Angular Spectrum,
and Wave Propagation Method (WPM) approaches.
"""

from .propagation import *
from .fdtd_solver import *
from .bloch import (
    solve_bloch_wave_gpu,
    wavelength_to_energy_eV,
    HAS_CUPY,
)

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"