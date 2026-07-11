# Ptychography robustness roadmap

This document records the evidence and acceptance gates for turning the current
matched synthetic reconstruction into a method that can be evaluated for
experimental use. A low diffraction residual is necessary, but it is not a
structural validation criterion by itself.

## Current evidence

The saved 750-update notebook result reached a validation amplitude loss of
approximately `1.47e-4`, while the synthetic-truth metrics remained poor:

- vacancy F1: `0.353`;
- displacement RMSE: `0.201 A`;
- potential NRMSE in the mutable region: `0.494`;
- eight displacement controls at their `+/-0.5 A` bounds;
- best validation checkpoint at the final update.

The result is therefore a low-residual, structurally non-identifiable solution,
not evidence that the optimizer merely needed more updates.

## 1. Interaction volume and atomic templates

Implemented:

- derive the forward interaction volume from probe coordinates, angle, waist,
  an excluded Gaussian-power budget, and geometry uncertainty;
- propagate beam-position and angle uncertainty as a spatially local envelope
  of bounded rays rather than applying the maximum path-length margin to every
  specimen pixel; reject angle intervals that reach a surface-parallel ray;
- require nominal overlap from training scans for the active-site volume, so
  validation and held-out geometry cannot create trainable specimen values;
- distinguish the uncertainty-expanded forward volume, pixel support,
  active-site support, and rendered lattice-potential influence halo;
- classify every catalogued lattice site with a digest-bound material-support
  contract. A site center in the nominal training-only update mask is a
  reportable `TARGET`; a non-target padded atomic patch that intersects the
  uncertainty-expanded forward mask is an optimized, non-reportable
  `NUISANCE`; explicitly fixed sites require provenance; below-budget sites
  are labelled as such; unresolved forward-relevant sites fail strict
  preparation;
- bind all-site coordinates, center indices, padded footprints, masks, roles,
  provenance, budgets, and exact parameter counts into the support-contract
  SHA-256 and prepared reconstruction problem ID. Results and non-pickled
  archives retain the modeled TARGET/NUISANCE partition, bind the typed
  fully-parameterized-material flag into version-2 support evidence, and
  reject inconsistent fields or metadata on load. Legacy version-1 metadata
  cannot promote that flag and is loaded fail-closed;
- treat a fixed-material provenance string as an assertion record, not proof.
  Only a scope with no unverified `FIXED_KNOWN` sites removes exterior material
  from the observability missing-scope list or satisfies the ensemble material
  trust flag;
- keep nuisance vacancies and the shared smooth displacement field in the
  forward fit while default metrics, structural plots, consensus calls, and
  reconstruction GIFs expose only TARGET sites. The full fitted potential is
  retained as a labelled forward-model diagnostic;
- require an exact support-contract ID and ordered role match before any
  TARGET-labelled plot or GIF. Current global-alignment candidates rebuild the
  complete slab but do not yet carry candidate-specific material contracts,
  so they remain available as untrusted forward-model diagnostics and are not
  animated as recovered TARGET structure;
- render the finite reference slab from explicit sites while retaining atomic
  potential tails across the material/vacuum boundary;
- select or reject the Si template cutoff against a larger common-grid
  reference, then compare the complete finite slab for the selected defect
  case and both signs of the maximum allowed displacement; the acceptance
  test uses the worst individual scan rather than only an aggregate error;
- directly tensor-integrate the Kirkland Si radial potential over specimen
  pixels and the finite projection width without abTEM's potential builder,
  projection integrals, or image interpolation. Compare it with the production
  Lobato template using immutable provenance/content digests, then propagate a
  selected Kirkland-template alternative specimen and report aggregate and
  worst-scan amplitude NRMSE. This has no acceptance threshold, is excluded
  from cutoff certification, and remains structurally untrusted because the
  IAM parameterization is not experimental evidence and accumulation,
  displacement rendering, and propagation are still shared;
- compute dose-scaled local Poisson-Fisher blocks with stochastic error checks
  as a necessary site-sensitivity screen;
- compute a dense SVD calculation of the ideal-model local expected
  Poisson-Fisher approximation for small reference problems, including
  explicit row-space estimability tests;
- apply the same gauge-free parameterization through a scan-batched,
  matrix-free JVP/VJP Fisher operator for at most 32 explicitly selected sites,
  with exact low-rank nuisance projection, audited zero-start PCG, and an
  exhaustive dense-SVD oracle for small problems. This phase is a bounded
  exact-follow-up foundation;
- screen every physical vacancy/displacement output with factorized Gaussian
  detector probes and separate Gaussian null probes. The prepared adapter uses
  the same Jacobian and nuisance projector, reports simultaneous chi-square
  marginal bounds conditional on accepted PCG solves, preserves every solver
  diagnostic, enforces hard resource budgets, and can only nominate sites for
  the exact selected-site method;
- bind matrix-free ideal-Poisson information to the exact prepared Poisson
  dose, fixed signal scale, dark level, numerical floor, validity mask, and
  calibration identifier. Gaussian/read-noise objectives, nonconstant dose or
  dark fields that the scalar operator cannot represent, and conflicting
  caller count models fail closed. Legacy amplitude problems are labelled as
  hypothetical count analyses;
- verify that the stored potential, vacancy fractions, displacement controls,
  rigid displacement, and displaced site coordinates describe one renderer-
  consistent reconstruction before computing matrix-free information.

Still required:

- perform an explicit direct-quadrature order-convergence sweep on the
  production sampling and validate atomic potentials against independent
  experimental or first-principles evidence. The Kirkland/Lobato comparison
  detects numerical/parameterization sensitivity but is not an independent
  physical validation;
- validate or expand the assumed Si site inventory. Interstitials,
  substitutions, adatoms, steps, amorphous material, and unknown chemistry are
  not represented by the lattice support contract and must remain rejected or
  structurally untrusted;
- benchmark the prepared all-site stochastic screen on the maintained geometry,
  add compact digest-bound persistence, and define a conservative nomination
  policy from its covariance/null intervals. A stochastic result never directly
  establishes observability or structural trust; nominated sites still require
  the exact selected-site method, which deliberately rejects requests above 32;
- extend the package-owned calibrated nuisance constructor beyond its current
  common scan-origin, probe shift/tilt/width, and detector
  frequency/gain/dark directions to per-scan geometry, partial coherence,
  detector nonlinearity/point spread, and out-of-lattice illuminated exterior
  material. Neither the generated nor arbitrary low-rank tangent can establish
  that the nuisance scope is complete;
- extend the local ray envelope to uncertain surface height/topography rather
  than treating the surface plane as exactly known.

Acceptance gates:

- every appreciably illuminated atom is known fixed, active, nuisance, or
  explicitly rejected;
- potential values outside the influence halo are invariant for all allowed
  site parameters;
- template potential and worst-scan amplitude errors pass configured budgets
  for surface, bulk, boundary, subpixel, and maximum-displacement cases.

On the maintained geometry, the reportable structure contains 3,607 target
vacancies plus 936 shared displacement controls: 4,543 structural quantities,
or 56.8 times fewer than the 257,909 mutable pixels. The complete safe inverse
problem also profiles 1,497 nuisance vacancies, giving 6,040 optimized
parameters and a 42.7-times total reduction. The earlier 50-times *total*
target is therefore not met once every forward-relevant exterior site receives
an independent guard occupancy. Those nuisance variables must not be omitted
from the honest optimization count; recovering a 50-times total reduction
would require a separately validated grouped/sparse nuisance model rather than
silent truncation or a pristine assertion.

## 2. Optimization and stopping

Implemented:

- distinguish numerical convergence from exhaustion of the update budget;
- target-loss and joint loss-plateau/parameter-step stopping;
- non-finite diagnostic failures;
- gradient, normalized-step, and active-bound histories;
- render the specimen once per validation checkpoint rather than once per
  evaluation batch;
- compile the complete gradient, Adam, and projection training step;
- remove the constant-control gauge by separating the equal-site-mean active
  translation from zero-mean residual displacement controls;
- use separate learning-rate scales and translation, vacancy, residual, and
  joint optimization stages;
- reserve distributed geometry-only audit blocks, optional neighboring guard
  scans, and exclude both from every reconstruction method;
- select multistart basins using validation loss before examining audit loss;
- retain compact site-parameter checkpoints and animate every update beside
  like-for-like truth only inside contract-bound TARGET influence support;
  prefer the streaming FFmpeg writer for the 501-frame notebook animation and
  expose stride, DPI, and writer controls, with a portable but memory-heavier
  Pillow fallback;
- summarize all statistically equivalent low-loss starts with equal-weight
  intervals, a real medoid representative, ambiguity calls, and conservative
  trust flags;
- prevent a provenance-free Boolean sensitivity mask from unlocking structural
  trust; the positive gate requires typed marginalized-observability reports
  for every accepted optimizer basin.
- prepare and eagerly compile one fixed-shape reconstruction problem, then
  reuse those executables with fresh optimizer and random states across
  deterministic multistart runs.

Still required:

- implement true global specimen/scan registration; the present translation
  moves active sites relative to the fixed exterior and is deliberately labeled
  with that limited scope;
- add a full-batch polishing stage and starts that span the still-missing
  global specimen/scan registration parameters;
- projected-gradient/KKT stopping and momentum handling at active bounds.

Acceptance gates:

- an improving run that reaches its update budget reports `converged=False`;
- plateau stopping requires stable physical parameters as well as stable loss;
- multiple starts agree within stated uncertainty on identifiable benchmarks;
- ambiguous low-loss solutions are reported as ambiguous, not selected as a
  trusted structure.

## 3. Realistic initialization

The reconstruction must not receive the generating registration or strain.
The intended initialization workflow is a bounded coarse search over surface
height, lattice origin, orientation, and lattice scale, followed by continuous
registration refinement. Vacancy fractions and zero-mean residual displacement then
start from zero. Independent high-frequency random strain is no longer used by
the notebook default. A bounded global registration search remains required.

The first truth-isolated global initializer is implemented. It generates a
deterministic, termination-balanced Sobol catalog over canonical axial phase,
in-section rotation, and lattice scale; copies only a geometry-stratified
training screen and complete validation rows into the selection boundary; and
reports paired-validation equivalence rather than breaking a tie with audit
data. Surface height, common probe shift, scan-origin error, and axial lattice
phase are recognized as one gauge combination in this geometry and must not be
fitted independently. A differentiable control-space projection can remove
translation, in-section rotation, and isotropic dilation from residual strain.

Every termination/phase/rotation/scale candidate rebuilds all fixed and
variable atoms, the finite pristine reference, site patches, controls, and
influence support. Candidates are ranked on training data, only a deterministic
shortlist reaches full validation, and the selected model can be passed directly
to a prepared local reconstruction with the similarity gauge enforced. The
notebook exposes this path behind `RUN_GLOBAL_ALIGNMENT` because the complete-
slab catalog is intentionally more expensive than the controlled matched-model
benchmark.

The current initializer still fixes the calibrated probe, angle, and scan
geometry. It now follows the coarse Sobol level with deterministic bounded
phase/rotation/log-scale stencils around the training-ranked frontier; only the
final shortlist reaches validation. Alignment evidence can be saved atomically
in a non-pickled, SHA-256-bound archive. Loading requires the exact raw scan and
forward-problem IDs, rebuilds every shortlisted complete slab, and recomputes
its training and validation losses. Continuous probe tilt/waist refinement and
reconstruction across every validation-equivalent global model remain required.
Alignment summaries are therefore explicitly marked structurally untrusted.

## 4. Experimental trust ladder

Implemented foundations:

- propagate an explicit Boolean detector-validity mask through pixel and
  lattice objectives, minibatches, validation, held-out audit evaluation,
  prepared-problem hashes, and non-pickled result archives;
- replace invalid values before evaluating square roots, so masked saturation
  sentinels or non-finite values have zero loss and gradient contribution;
- fail closed when a fitted or assessed scan has no valid detector pixels;
- retain the normalized-amplitude objective for legacy non-negative intensity
  data and label it explicitly as neither a Poisson nor a read-noise
  likelihood;
- provide a truth-free calibrated measurement container that carries
  dark-subtracted signal, total electron-equivalent observations, validity,
  calibrated dark and read-noise arrays, and a calibration identifier;
- convert forward-model FFT intensities to expected signal electrons using the
  declared per-pattern dose and incident-probe norm, with a fixed relative
  signal scale that is never fitted during reconstruction;
- support an ideal-Poisson deviance for non-negative total electron-equivalent
  observations and a heteroscedastic Gaussian approximation for calibrated
  dark-subtracted data with declared read noise. The latter is not described
  as the exact Poisson--Gaussian convolution;
- bind the objective, dose, calibration arrays, validity mask, and calibration
  identifier into the prepared-problem digest and non-pickled result archive;
- adapt synthetic detector-benchmark output through a narrow boundary that
  deliberately omits truth, random seeds, raw ADU values, saturation causes,
  and generating detector parameters.

Required benchmark levels are:

1. matched noiseless unit tests;
2. truth generated by an independent forward implementation;
3. Poisson dose sweeps and detector gain, dark, saturation, masking, background,
   and read-noise perturbations;
4. probe, scan-position, registration, angle, lattice, potential-model, and
   slab mismatch sweeps;
5. out-of-model substitutions, interstitials, adatoms, steps, and exterior
   defects;
6. pristine negative controls, multiple starts, and blocked spatial test data;
7. repeated calibration and specimen measurements.

A result is structurally trustworthy only when numerical convergence,
observability, independent-start agreement, residual calibration, and the
relevant mismatch benchmark all pass. The currently implemented dense report
holds probe, scan, detector, and exterior-material nuisance parameters fixed.
The matrix-free phase can profile either an explicit detector-space tangent or
a package-generated, calibration-bound tangent for common scan-origin, probe,
and detector directions. The generated profile is differentiated from the
exact prepared model in the same whitened count observable, but it still cannot
prove that the experimental nuisance scope is complete. Package-produced
reports therefore deliberately cannot unlock the structural-trust gate yet. A
declared calibration Boolean and identifier are retained as provenance but do
not set the calibrated-noise trust flag without typed calibration evidence.

The calibrated objectives are an implementation boundary, not a validation
claim. In particular, the present Poisson deviance accepts calibrated
electron-equivalent totals and does not enforce a raw integer-count contract;
the Gaussian read-noise objective is an approximation; detector gain and
background are fixed calibration inputs rather than fitted nuisances; and the
pixel reconstruction remains on the legacy amplitude objective. Structural
trust therefore remains false until independent calibration, dose/noise
coverage, residual calibration, and mismatch benchmarks pass. A future exact
count path must bind the raw-count acquisition contract, while a future
read-noise path must either validate the approximation over the operating
range or use a numerically stable Poisson--Gaussian convolution.

## 5. Package design

The target package separates measured data, geometry, specimen priors,
rendering, forward simulation, objectives, optimization, diagnostics,
synthetic benchmarks, I/O, and plotting. Experimental datasets must not require
truth fields. Saved results need a schema version, model and input hashes,
software provenance, device, precision, and enough specimen information to
rerender checkpoints independently.

## 6. Performance

Correctness benchmarks are frozen before performance changes. GPU profiling
must report compilation time, steady-state updates per second, validation time,
serialization time, and peak device memory with explicit synchronization.
Next priorities are compact delta rendering, cached reference transmission,
on-device batch generation, fixed-shape compiled evaluation, and explicit
precision/device policies. CPU correctness tests and scheduled GPU science and
performance tests are both required.

The synchronized harness runs the current workflow and prepared API directly:

```bash
python scripts/benchmark_ptychography_1d.py \
  --device gpu --precision float64 --updates 500 --starts 5 \
  --output benchmark_notebook_gpu.json
```

Its JSON distinguishes one-time build, simulation, and eager preparation from
per-start run and optimization times, records the exact scope of each
updates/s value, and reports JAX allocator memory statistics where the backend
provides them. `--quick --device cpu --updates 2 --starts 1` is available as a
non-comparable harness smoke test.

A synchronized float64 run on an NVIDIA A100 80 GB PCIe on 2026-07-11 used the
exact default notebook geometry, two starts of 20 updates, and a validation
interval of 10. It measured 20.08 s for experiment construction, 69.52 s for
the synthetic dataset and cutoff stress checks, 13.01 s for eager preparation,
and 26.33 s and 21.94 s for the two starts. Aggregate optimization-phase
throughput was 0.838 updates/s and the process-cumulative JAX allocator peak
was 4.87 GB. This is a short, evaluation-heavy end-to-end reference, not an
extrapolated 500-update completion time; in particular, it must not be compared
with an isolated renderer or train-step kernel rate.

After restricting non-authoritative training-loss history to 32 fixed,
geometry-stratified scans, the identical synchronized protocol measured 19.33 s
for construction, 68.23 s for simulation, 12.84 s for preparation, and 19.39 s
and 13.01 s for its two starts. Aggregate optimization-phase throughput was
1.257 updates/s, 50.1% above the 0.838 updates/s full-training-diagnostic
reference, with the same 4.87 GB process-cumulative allocator peak. Each run
used 96 training-diagnostic scan evaluations instead of 729, while complete
validation still selected checkpoints and the final full 243-scan training loss
was recomputed. The first start includes residual one-time execution costs, so
both per-start phase timings and the aggregate are retained in the JSON.

The complete support-contract implementation increases the maintained problem
from 3,607 TARGET sites to 5,104 modeled sites: 3,607 reportable TARGET sites
and 1,497 profiled NUISANCE sites. A subsequent 20-update, two-start float64
run measured 20.59 s for construction, 81.50 s for simulation, 19.76 s for
preparation, and 19.13 s and 13.18 s for its two starts. Its synchronized
optimization-phase throughput was 1.270 updates/s and its process-cumulative
JAX allocator peak was 6.55 GB. Thus the optimizer rate was effectively
unchanged while the modeled-site count grew by 41.5%; memory increased by
34.5%, and the larger finite specimen made build and simulation more
expensive. This comparison is indicative rather than controlled: the two runs
used different JAX environments, the A100 was shared, both worktrees were
dirty, and allocator peaks are process-cumulative. The JSON report and exact
environment metadata remain the authoritative record for any controlled
regression claim.

Next priorities are compact delta rendering, cached reference transmission,
on-device batch generation, and scheduled GPU science/performance regression
runs.

The prepared runner now supports a fixed geometry-stratified training-loss
diagnostic subset. The maintained workflow requests 32 scans when a validation
split exists; complete validation remains authoritative for checkpoint
selection, stopping, and basin comparison, and an exact full-training loss is
recomputed from the selected final prediction. Coarse synchronized phase
timings and scan/batch evaluation counts are included in result metadata and
the benchmark JSON. This reduces diagnostic forward work without changing the
optimizer updates or validation trajectory; a no-validation problem
automatically falls back to complete training evaluation.
