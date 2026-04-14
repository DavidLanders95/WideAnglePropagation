# Plan: Paper Results Structure from 4 Notebooks

## TL;DR
Restructure Section 6 (Results) around four experiments of increasing complexity. Central thesis: the angular spectrum propagator — a trivial modification to Fresnel multislice — is sufficient for high-angle scattering. WPM and KG/Lanczos methods cost much more compute without meaningful accuracy gains.

## Overall Narrative Arc
1. The angular spectrum propagator is a minimal change to Fresnel multislice.
2. It captures high-angle scattering as well as methods derived from the full Klein-Gordon equation.
3. WPM and KG/Lanczos cost significantly more compute without meaningful accuracy gains.
4. This holds across z-sampling regimes, convergence angles, and frozen-phonon simulations.

## Proposed Results Structure

### Phase A: Verification (NB 01)
Subsection 6.1 — "Verification against Rother and Scheerschmidt (2009)"
- Fig 2 (exists): Beam amplitude vs thickness, Au [100], 300 keV — beams [0,0] and [0,28] (135 mrad).
- Tab 2 (new): RMSE vs digitized paper data.
- Key message: The angular spectrum method is a small change but essentially gives you the high-angle results. First indication that expensive methods offer marginal benefit.

### Phase B: Z-Sampling Convergence (NB 02)
Subsection 6.2 — "Z-sampling convergence"
- Fig 3 (new): 1x2 subplot — rel L2 error vs slices/cell + runtime vs slices/cell.
- Tab (optional): Numeric convergence data.
- Key message: WPM/Lanczos do not let you get away with less z-sampling. Diminishing returns: much more compute for less than one order of magnitude improvement. Angular spectrum converges at a similar rate for a fraction of the cost.

### Phase C: Convergence Semi-Angle (NB 03)
Subsection 6.3 — "Accuracy vs probe convergence angle"
- Fig 4 (new): Rel L2 error vs semi-angle (5-100 mrad), 4 methods at 32 sl/cell.
- Fig 5 (new): CBED pattern grid at selected angles (5, 20, 50, 100 mrad).
- Key message: No benefit to Lanczos or WPM for multislice in convergent-beam context. Angular spectrum gives you what you need.

### Phase D: Frozen Phonon (NB 04)
Subsection 6.4 — "Frozen phonon CBED — Si [111]"
- Fig 6 (new): FP-averaged CBED patterns (log scale).
- Fig 7 (new): Difference maps vs KG FWD Lanczos.
- Tab 3 (new): RMSE, rel L2, timing per method.
- Key message: Even with realistic thermal disorder, angular spectrum gives the desired output. More expensive methods provide no practical benefit.

## Figure/Table Inventory

| #     | Type   | Source        | Content                                 |
|-------|--------|---------------|-----------------------------------------|
| Tab 1 | Table  | Existing      | Helmholtz versions                      |
| Fig 1 | Figure | Existing TikZ | Method comparison schematic             |
| Fig 2 | Figure | NB 01         | Beam amplitudes vs thickness, Au [100] |
| Tab 2 | Table  | NB 01         | RMSE vs paper reference                 |
| Fig 3 | Figure | NB 02         | Rel L2 + runtime vs slices/cell         |
| Fig 4 | Figure | NB 03         | Rel L2 vs convergence semi-angle        |
| Fig 5 | Figure | NB 03         | CBED patterns at selected angles        |
| Fig 6 | Figure | NB 04         | FP-averaged CBED patterns               |
| Fig 7 | Figure | NB 04         | Difference maps vs KG reference         |
| Tab 3 | Table  | NB 04         | Error metrics + timing                  |

## Relevant Files
- Paper/main.tex — restructure Section 6.
- Paper/figures/ — destination for exported PDFs.
- notebooks/verification/01_axel_lubk_verification.ipynb — Fig 2, Tab 2.
- notebooks/convergence/02_z_sampling_convergence_au.ipynb — Fig 3.
- notebooks/cbed/03_convergent_probe_au.ipynb — Figs 4-5.
- notebooks/frozen_phonon/04_frozen_phonon_cbed_si.ipynb — Figs 6-7, Tab 3.

## Decisions
- Ground truth: KG ODE (fine potential) for NB 02-03; KG FWD Lanczos (same z) for NB 04 — justify in text.
- Current Sec 6.2 placeholder replaced by this new structure.
- Unused TikZ diagrams stay in Methods section only.
- Paper framing (intro/abstract) may need revision: contribution is the systematic comparison showing angular spectrum suffices, not advocating WPM adoption.

## Further Considerations
1. Fig 5 sizing: 4x5 grid may be too large; consider 2 representative angles.
2. Timing consolidation: combine NB 02 + NB 04 timing into a single subsection or table.
3. Supplementary: full numeric tables from NB 02/03 in supplement, plots only in main text.
4. Intro reframing: current intro motivates WPM; revised conclusions mean intro should frame this as a comparison study.
