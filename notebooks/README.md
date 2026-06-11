# Notebooks

These are the maintained notebooks for reproducing the main examples and
figures. They are stored without execution output; run them from the repository
root after installing the package and GPU dependencies.

## Maintained Workflows

- `verification/01_axel_lubk_verification.ipynb`: Au [100] plane-wave
  verification comparing Fresnel multislice, angular-spectrum multislice, WPM,
  and the full second-order KG ODE reference.
- `cbed/02_converge_probe_si.ipynb`: Si CBED comparison and radial peak-shift
  analysis.
- `cbed/03_convergent_probe_au.ipynb`: Au [100] convergent-probe CBED
  comparison, thickness error curves, and cross-section visualization.

## Data And Outputs

- Input crystal data lives in `notebooks/data/`.
- Generated compact simulation caches and notebook figures are written under
  `notebooks/cbed/results/` and `notebooks/verification/figures/`.
- Paper-ready exports are written under `Paper/figures/` when the save flags in
  each notebook are enabled.

Older exploratory notebooks are kept under `notebooks/archive/`.
