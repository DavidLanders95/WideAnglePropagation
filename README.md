# Wide-Angle Propagation

A Python package for simulating wave propagation in electron microscopy using various numerical methods.

## Features

- **Fresnel Propagation**: Paraxial approximation for near-field propagation
- **Angular Spectrum Method**: Full angular spectrum propagation
- **Wave Propagation Method (WPM)**: Adaptive binning for wide-angle propagation
- **FDTD Solver**: Finite-difference time-domain solver for relativistic scattering
- **CUDA Acceleration**: GPU-accelerated FDTD implementation

## Installation

### From Source

```bash
git clone https://github.com/yourusername/wide-angle-propagation.git
cd wide-angle-propagation
pip install -e .
```

### Dependencies

- numpy
- jax
- jaxlib
- abtem
- ase
- matplotlib
- tqdm
- scipy

## Usage

```python
import wide_angle_propagation as wap
import numpy as np

# Example: Fresnel propagation
potential = ...  # Your potential array
probe = ...      # Your probe wavefront
propagator = wap.fresnel_propagation_kernel(...)
exit_wave, diffraction_pattern = wap.simulate_fresnel(potential, probe, propagator, ...)
```

## Notebooks

The `wpm/` and `FDTDmethod/` directories contain Jupyter notebooks demonstrating various propagation methods and comparisons.

## License

MIT License