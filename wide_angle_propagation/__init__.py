"""
Wide-Angle Propagation Package

This package provides implementations of various wave propagation methods
for electron microscopy simulations, including Fresnel, Angular Spectrum,
and Wave Propagation Method (WPM) approaches.
"""

from .propagation import *
from .fdtd_solver import *
from .ptychography import (
    make_probe,
    make_gaussian_probe,
    generate_scan_positions,
    fourier_shift,
    simulate_ptychography_as,
    simulate_ptychography_wpm,
    epie_thin,
    epie_multislice_as,
    reconstruct_as,
    reconstruct_wpm,
    make_phase_object,
    make_potential_phantom,
)

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"