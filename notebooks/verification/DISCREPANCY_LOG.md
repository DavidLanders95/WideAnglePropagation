# Discrepancy Log: Au [110] Simulation vs Paper

**Reference**: Rother & Scheerschmidt 2009, doi:10.1016/j.ultramic.2008.08.008, Figure 3  
**Material**: Au [110], 300 keV, beam amplitudes [0,0] and [0,28] vs thickness  
**Goal**: Identify why all five propagation methods systematically deviate from paper curves

---

## Status Summary

| # | Hypothesis | Status | Impact |
|---|-----------|--------|--------|
| 1 | Infinite vs finite projection | **Explained** — finite uses wrong V(r) | See H13 |
| 2 | Refractive index vs sigma phase | **Ruled out** | Negligible (~1.5e-4 MAE) |
| 3 | Lobato vs Weickenmeier-Kohl | **Confirmed** WK needed | Large |
| 4 | 128 slices/cell convergence | **Verified** converged | None |
| 5 | WK coefficient accuracy | **Verified correct** vs WK 1991 Table 1 | None |
| 6 | Mott-Bethe conversion factor | **Verified correct** (47.878 V·Å²) | None |
| 7 | Lattice parameter 4.076 vs 4.08 Å | **Confirmed mismatch** | Small (~0.3%) |
| 8 | Debye-Waller / thermal factors | **Ruled out** — paper uses M=0 | None |
| 9 | Absorptive potential (V=0.4) | **Ruled out** — V is fitting param, abs=0 when M=0 | None |
| 10 | Zone axis orientation / cell setup | **Verified correct** [100] via cubic FCC | None |
| 11 | Relativistic wavelength formula | **Verified correct** (0.01969 Å) | None |
| 12 | Real-space cutoff / aliasing | **Not tested** | Moderate |
| 13 | WK real-space potential formula BUG | **CONFIRMED** | **Critical** |

---

## Completed Investigations

### H1: Infinite vs Finite Projection — NOW EXPLAINED BY H13
- **Test**: Compared WK infinite vs WK finite projection over 25-cell Fresnel sweep
- **Result**: Infinite projection gives MAE ~2.88e-3 vs paper [0,0]; finite is slightly worse
- **Conclusion**: **Explained.** The finite projection uses an incorrect real-space potential formula (see H13). Infinite projection bypasses this by working in reciprocal space.

### H2: Refractive Index vs Sigma Phase Representation
- **Test**: Compared exact n(r) = sqrt((E-V)E₀/(E(E+2E₀))) vs Taylor n ≈ 1 + σV
- **Result**: Difference in MAE: ~1.46e-4; first-slice phase diff: ~1.20e-2 rad
- **Conclusion**: **Ruled out.** Effect is negligible compared to the observed discrepancy.

### H3: Lobato vs Weickenmeier-Kohl Parametrization
- **Test**: Full sweep with both parametrizations
- **Result**: WK required for <5% error on AS multislice [0,0] beam
- **Conclusion**: **Confirmed.** WK is necessary but not sufficient — residual error remains.

### H4: Z-Sampling Convergence (128 slices/cell)
- **Test**: ODE solver convergence check
- **Result**: 128 slices/cell is converged for all methods
- **Conclusion**: **Verified.** Not a source of error.

### H5: WK Coefficient Accuracy — VERIFIED
- **Test**: Cross-checked B coefficients and A formula against Table 1 of Weickenmeier & Kohl 1991 (Acta Cryst. A47, 590-597), page 5.
- **Result**: **Exact match.**
  - B = [5.493E-01, 1.728E+00, 6.720E+00, 2.637E-02, 7.253E-02, 3.546E+01] ✓
  - V = 0.4 fitting parameter ✓
  - A₁=A₂=A₃ = 0.02395×Z/(3(1+V)) = 0.450488 Å⁻¹ ✓ (WK eqs 12-14)
  - A₄=A₅=A₆ = V×A₁ = 0.180195 Å⁻¹ ✓
  - Sum(A) = 0.02395×79 = 1.892050 ✓ (constraint eq 11)
- **Conclusion**: **Verified correct.** Code coefficients match the published table exactly.

### H6: Mott-Bethe Conversion Factor — VERIFIED
- **Test**: Computed h²/(2πm_e e) using CODATA constants from ASE.
- **Result**: 47.87765 V·Å² vs code value 47.87801. Difference < 0.001%.
- **Conclusion**: **Verified correct.**

### H7: Lattice Parameter — CONFIRMED MISMATCH
- **Test**: Paper (Rother 2009, page 5) states Au (Fm3m; a = 4.08 Å). Code uses 4.076 Å.
- **Result**: 
  - Difference: 0.004 Å (0.098%)
  - V_200 changes by -0.217% (code potential is 0.3% stronger due to smaller cell)
- **Conclusion**: **Confirmed minor mismatch.** Should use a = 4.08 Å to match paper, but this alone explains only a tiny fraction of the discrepancy.

### H8: Debye-Waller Factor — RULED OUT
- **Test**: Paper explicitly states (page 5): "no other contributions, e.g. those of the microscope or thermal motion of the atoms, have been incorporated."
- **Result**: Paper uses zero thermal motion, matching code's thermal_sigma = 0.
- **Conclusion**: **Ruled out.**

### H9: Absorptive Potential — RULED OUT
- **Test**: Analyzed WK 1991 eq (6) for the absorptive form factor f'(g).
- **Result**: The V = 0.4 parameter is a **fitting parameter** for the elastic scattering amplitude shape (WK eqs 12-14), NOT an absorptive ratio. The absorptive form factor f'(g) from WK eq (6) is **identically zero when M = 0** (no thermal motion), because the two DW exponentials cancel: exp(-Mg²) - exp(-M[q²+(q-g)²]) = 1 - 1 = 0.
- **Conclusion**: **Ruled out.** No absorption when thermal motion is zero.

### H10: Zone Axis Orientation — VERIFIED CORRECT
- **Test**: Paper (page 5) states Au in "[100] zone axis orientation." `bulk("Au", "fcc", cubic=True)` creates a cubic cell with beam along z = [001], which is equivalent to [100] by cubic symmetry.
- **Result**: Zone axis is correct. However, the notebook title says "[110]" which is a **labeling error** in the notebook (not in the physics).
- **Note**: The [0,28] beam gives a scattering angle of 135.1 mrad, matching the paper's stated "135 mrad" exactly.
- **Conclusion**: **Verified correct.**

### H11: Relativistic Wavelength — VERIFIED CORRECT
- **Test**: Computed wavelength from `energy2wavelength(300e3)`.
- **Result**: λ = 0.019687 Å (expected ~0.01969 Å).
- **Conclusion**: **Verified correct.**

### H13: WK Real-Space Potential Formula BUG — **CRITICAL FINDING**
- **Test**: Computed the correct 3D spherical Fourier transform of the WK scattering factor f_e(s) and compared against the code's `weickenmeier_kohl_potential` function.
- **Derivation**:

  The WK scattering factor is: f_e(s) = s⁻² Σ Aᵢ [1 - exp(-Bᵢ s²)]

  The real-space potential is V(r) = 47.87801 × (2/r) × ∫₀^∞ f_e(s) sin(2πrs) s ds

  For each term: ∫₀^∞ Aᵢ(1-e^(-Bᵢs²))/s × sin(2πrs) ds = Aᵢ (π/2) erfc(πr/√Bᵢ)

  (Using the identity: ∫₀^∞ (1-e^(-ax²))/x sin(bx)dx = (π/2)erfc(b/(2√a)))

  **Correct formula**: V(r) = 47.87801 × (π/r) × Σ Aᵢ erfc(πr/√Bᵢ)

  **Code's formula**: V(r) = 47.87801 × (4π/r) × Σ Aᵢ erfc(2πr/√Bᵢ)

- **Result**: Two errors in the code:
  1. Overall prefactor: **4π instead of π** (4× too large)
  2. erfc argument: **2πr/√B instead of πr/√B** (2× too large)

  These partially cancel but leave large residual errors:
  | r (Å) | Correct V(r) | Code V(r) | Code/Correct |
  |-------|-------------|-----------|--------------|
  | 0.1   | 1739 eV     | 4918 eV   | 2.83×        |
  | 0.5   | 104.2 eV    | 146.1 eV  | 1.40×        |
  | 1.0   | 18.3 eV     | 14.9 eV   | 0.81×        |
  | 2.0   | 1.86 eV     | 0.15 eV   | 0.08×        |

- **Impact**: The `weickenmeier_kohl_potential` function is ONLY used by abTEM's **finite projection** mode. The reciprocal-space functions (`elastic`, `projected_scattering_factor`) are correct and used by **infinite projection**. This explains why infinite projection gives better results.
- **Conclusion**: **CONFIRMED BUG.** The WK real-space potential formula is wrong. This affects all simulations using WK parametrization with `projection="finite"`. **Use `projection="infinite"` as a workaround**, or fix the potential function.
- **Fix**: Change `weickenmeier_kohl_potential` to:
  ```python
  def weickenmeier_kohl_potential(r, parameters):
      A, B = parameters
      r = np.asarray(r, dtype=np.float64)
      result = np.zeros_like(r)
      for i in range(len(A)):
          result += A[i] * scipy_erfc(np.pi * r / np.sqrt(B[i]))
      with np.errstate(divide='ignore', invalid='ignore'):
          V = 47.87801 * np.pi * result / r
      V = np.where(r < 1e-14, 1e30, V)
      return V
  ```

---

## Pending Investigations

### H12: Real-Space Cutoff and Aliasing  
**Priority: LOW** (may be resolved by switching to infinite projection)

WK cutoff = 20 Å on 128×128 grid with ~4 Å cell may cause wrapping artifacts in finite projection.

---

## Action Items

1. **Fix the lattice parameter**: Change from a = 4.076 Å to a = 4.08 Å to match Rother 2009.
2. **Fix the notebook title**: Change "[110]" to "[100]" to match the actual zone axis.
3. **Use infinite projection for WK** (workaround): The `projected_scattering_factor` is correct in reciprocal space.
4. **Fix `weickenmeier_kohl_potential`** (proper fix): Change prefactor from 4π to π, and erfc argument from 2πr/√B to πr/√B. This restores correct behavior for finite projection.
5. **Re-run all simulations** after fixes to see how much the discrepancy is reduced.

## Investigation Order (Revised)

1. ~~H10~~ ✅ Zone axis correct
2. ~~H5~~ ✅ WK coefficients verified
3. ~~H9~~ ✅ Absorption ruled out (M=0)
4. ~~H8~~ ✅ Debye-Waller ruled out (paper = no thermal motion)
5. ~~H6~~ ✅ Mott-Bethe verified
6. ~~H7~~ ✅ Lattice parameter: 4.076 should be 4.08
7. ~~H11~~ ✅ Wavelength verified
8. **H13** ✅ **CRITICAL BUG FOUND**: WK real-space potential formula
9. H12 — Cutoff / aliasing (low priority, may be moot after fix)
