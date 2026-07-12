# Sparse atomistic-edit ptychography

This document is the specification and trust checklist for the physics-based
side-view reconstruction. There is one user-facing specimen model: a known,
deformable reference crystal plus sparse atomic removals and positive
off-crystal scattering centres. The reference-crystal renderer remains an
internal building block, not a second reconstruction method.

The governing principle is:

> Find the smallest data-supported change to the known host, keep every atom
> inside a broad physics-defined admissibility envelope, and report ambiguity
> whenever the acquisition cannot localize the change.

## 1. Is this the right model?

It is a good primary model when all of the following are credible before
reconstruction:

- one crystalline host species dominates the illuminated volume;
- approximate pose, surface location, sampling, probe and detector calibration
  are independently known;
- departures from the host are sparse relative to the illuminated crystal;
- the desired result is defect support, displacement and host-equivalent added
  scattering, not automatic chemical identification.

The known host is a strong prior, but it is not the defect answer. Starting
from a pristine host is fair only when registration and material identity come
from calibration or a truth-free search. Defect locations, counts and shapes
must never cross the reconstruction boundary.

This is not an open-world structure solver. It becomes inappropriate for an
unknown phase, a mostly amorphous specimen, large unmodelled charge transfer,
dense reconstruction-wide disorder, or an unknown host orientation. Those
cases need a broader specimen model rather than more aggressive optimization.

Lennard--Jones is not used. It is a configurational-energy model and is not an
electron-scattering potential; it is also a poor default for covalent silicon.
The forward model instead uses validated atomic electrostatic kernels. A
material energy model may later act only as a one-sided exclusion envelope.

## 2. Specimen representation

The specimen is

\[
X=X_0(\mathbf u)\ominus X_-\oplus X_+,
\]

with projected potential

\[
V_X(\mathbf r)=
\sum_{i=1}^{N_h}(1-b_i)
v_h\!\left(\mathbf r-\mathbf R_i^0-\mathbf u(\mathbf R_i^0)\right)
+\sum_{j=1}^{K_+}a_jv_{\rm eff}(\mathbf r-\mathbf x_j).
\]

Here:

- \(X_0\) is the declared finite host crystal;
- \(b_i\in[0,1]\) removes scattering from an active host site;
- \(\mathbf u\) is a smooth, bounded host displacement field;
- \(\mathbf x_j\) is a continuous side-view addition position;
- \(a_j\in[0,a_{\max}]\) is positive host-equivalent integrated scattering.

The first implementation deliberately does not infer element labels for added
centres. A substitution is a host removal plus a nearby positive addition. A
vacancy, adatom, interstitial, missing row and irregular cluster all use the
same representation; no object class, radius, centre, boundary or phase label
is supplied.

For the maintained two-dimensional problem, the active structural count is

\[
P_{\rm structure}=P_{\rm deformation}+K_-+3K_+.
\]

Fixed-capacity arrays are compilation resources. Reported parameter counts use
active edits, not capacity.

## 3. Interaction and discovery volume

The user does not draw an update rectangle. Geometry constructs the support
from scan coordinates, probe waist, glancing angle, declared position/angle
uncertainty, surface envelope and an excluded Gaussian-power budget.

The support contract distinguishes:

- **TARGET**: nominal training illumination supports reconstruction and public
  structural reporting;
- **NUISANCE**: uncertainty-expanded or held-out illumination can scatter from
  the location, so it is fitted but never reported as recovered structure;
- **fixed/below budget**: material is retained only with explicit provenance or
  a declared excluded-power approximation;
- **unresolved**: a forward-relevant location without a valid role; preparation
  fails closed.

Training scans alone define TARGET support. Validation and audit geometry never
create trainable specimen values. Atomic influence halos are included in the
forward model even where the atom centre lies outside the reportable mask.

The discovery boundary is hard. An active continuous centre is accepted only
when its complete interpolation cell lies in TARGET/NUISANCE support and its
continuous transverse coordinate remains inside the surface envelope. This
constraint is enforced for every ablation, including the edit-only arm.

Atomic templates require complete finite-grid containment. The current direct
Kirkland truth renderer uses a factorized finite-voxel integral: Gaussian terms
are analytic error-function products and Yukawa terms reduce to one adaptive
vector quadrature. Tensor quadrature remains a numerical diagnostic, not the
accepted near-core truth path.

## 4. Objective and physical prior

The default Level-1 objective is

\[
\mathcal J_0=
D_{\rm Poisson}\!\left[Y\middle\|\mathcal F_{\rm MS}(V_X,\eta)\right]
+\lambda_{\rm edit}\left(\sum_i b_i+\sum_j a_j\right)
+\frac{1}{2\sigma_\epsilon^2}\mathbf u^\mathsf TL_{\rm el}\mathbf u
+R_{\rm hc}(X).
\]

Each term has one role:

- the calibrated Poisson count likelihood and multislice propagator contain the
  measurement physics;
- atomic kernels map coordinates to electrostatic potential;
- one edit-mass penalty asks for the smallest host-equivalent change;
- weak symmetric strain discourages unsupported high-frequency deformation;
- a steep hard-core barrier excludes atomic overlap.

Hard-core pair weights vanish with occupancy. Extra--extra terms use the
normalized product of their masses; host--extra terms use \((1-b_i)a_j\).
Dormant centres therefore exert no force, and a fully removed host may be
replaced at the same site.

The edit penalty is the only statistical structural regularization strength.
Its decreasing path must be frozen before inspecting recovered structure,
using pristine controls or held-out count prediction.

### Optional energy envelope

A chemistry-specific interatomic potential is not part of Level 1. It may be
tested only when its surfaces, defects, strain and cross-species environments
are validated. The allowed form is one-sided:

\[
R_E(X)=\sum_i\operatorname{softplus}\!\left(
\frac{e_i-e_{i,\rm allow}}{\Delta e_i}
\right)^2.
\]

Below the allowed energy it should be nearly flat. It must be rejected if it
erases a count-supported metastable defect or improves its own energy while
worsening held-out prediction.

## 5. Reconstruction algorithm

The active-set solver is intentionally understandable:

1. start from the deformed host with an empty edit set;
2. differentiate the full-training count objective with respect to potential;
3. correlate that adjoint with the atomic kernel over the discovery grid and
   score host removals and paired replacements;
4. birth the largest penalized KKT violation;
5. jointly refine active masses, continuous positions, removals and host
   deformation through the full multislice model;
6. re-anchor continuous centres, prune zero edits and merge only numerical
   duplicates inside one declared resolution element;
7. stop when proposal-grid dormant directions and active projected gradients
   satisfy their declared tolerances;
8. select a point on the frozen decreasing-penalty path using validation counts;
9. freeze support/positions and debias amplitudes without the edit penalty;
10. inspect audit counts only after selection.

The KKT certificate currently covers the declared proposal grid, not all
continuous birth positions. Capacity exhaustion, unresolved duplicates,
non-finite values and incomplete debiasing fail closed.

TQDM reports active-set progress. An optional truth-free callback emits
immutable states at meaningful structural events: initialization, refinement,
birth, polish, completed penalty level and debias. These events drive the
reconstruction GIF; they are not mislabelled as every Adam sub-update.

Training derivatives are accumulated deterministically over bounded scan
batches. Each batch contributes its unnormalized Poisson deviance and the
solver divides by the valid-pixel count over the complete training split; the
prior is added exactly once. This is an exact full-training gradient with lower
peak memory, not stochastic minibatch optimization.

Multiple starts vary only bounded host controls and begin with empty edits.
Validation selects the candidate before audit evaluation. Disagreement among
validation-equivalent starts is reported as structural ambiguity.

## 6. Runtime contract and easy-use API

The material-specific run configuration may contain only:

- removal/addition capacities;
- one-centre scattering bound;
- minimum admissible separation;
- expected RMS host strain;
- an explicit frozen edit-penalty path;
- vacuum discovery depth, exact scan-batch size and solver resource limits.

It must not contain object existence, count, position, radius, shape, phase,
composition or synthetic truth.

The normal entry point is
`reconstruct_silicon_atomistic_edits_1d(experiment, measurement, objective,
config=...)`. `SiliconAtomisticEditConfig1D` holds the policy above and requires
the penalty path explicitly. `plot_silicon_atomistic_edit_run_1d` keeps the
TARGET display boundary, while `save_silicon_atomistic_edit_run_1d` and
`load_silicon_atomistic_edit_run_1d` provide authenticated non-pickled replay.
`summarize_silicon_atomistic_edit_run_1d` reports active removals, additions and
stopping evidence without exposing the low-level assembly API. The returned
`SiliconAtomisticEditRun1D` still contains the prepared problem and the
selected/debiased result for specialist inspection.

The low-level state, proposal, certificate, truth-generator and benchmark
types remain available from their specialist modules. They are not the normal
user interface.

## 7. Initialization and nuisance separation

The inverse method starts with zero removals, zero additions and zero residual
strain. It must not receive generating registration or strain. A real workflow
requires a bounded truth-free search over surface height, crystal origin,
orientation and scale before local refinement.

Probe, scan, detector and coherence mismatch must not be absorbed as atoms.
Only bounded, calibrated nuisance parameters are admissible; no free nuisance
image or per-scan correction field is allowed. The next required physical
extension is a common calibrated probe phase/tilt nuisance, because the current
nuisance-only blind case otherwise fails closed without attribution evidence.

## 8. Evidence and trust ladder

A small residual is not structural validation. A result is trustworthy only
when all relevant levels pass:

1. renderer identity, normalization, support and gradient tests;
2. matched noiseless recovery;
3. independent numerical truth generation;
4. Poisson dose and detector perturbations;
5. probe/scan/registration/potential/slab mismatch;
6. pristine controls, blocked spatial audit and multiple starts;
7. acquisition-bound observability and depth response;
8. repeated calibration and experimental measurements.

Saved evidence must reproduce the model, acquisition, partition, path states,
objective terms, KKT/capacity status and debiased specimen without pickle.
Software/device/precision fields are provenance, not trust flags.

## 9. Required blind cases

One immutable public reconstruction schema and one selection rule are used for:

1. pristine host;
2. one vacancy;
3. one off-crystal addition;
4. one substitution generated with a different atomic kernel;
5. one irregular finite cluster without object metadata;
6. one strained/metastable defect;
7. nuisance-only probe/scan/coherence mismatch;
8. one depth-unresolved addition.

Run edit-only and Level-1 arms for every case. The energy-envelope arm remains
blocked until chemistry is justified. Private truth and audit counts are opened
only after every selection callback returns.

The current case generators validate schema isolation and synthetic physics,
but the complete real-solver matrix has not yet established all recovery gates.
In particular, nuisance attribution and acquisition-derived depth intervals are
still missing; asserted uncertainty scalars do not count as observability.

## 10. Current implementation evidence

As of 2026-07-12:

- zero edits reproduce the finite host exactly;
- unit vacancies/additions, permutation/dormancy, support and finite-difference
  gradients are tested;
- the continuous discovery boundary is enforced during validation and
  backtracking in both solver arms;
- objective components, births, pruning, merging, proposal-grid KKT, frozen
  path selection and fixed-support debias are implemented;
- fixed-shape compiled objectives are reused across topology and penalty changes;
- exact deterministic scan batching keeps the maintained side-view gradient
  below accelerator memory limits without changing its full-training value;
  proposal/KKT and archive replay use the same recorded batching contract;
- the renderer transpose is factorized into a batched potential adjoint,
  bounded local atomic-patch contractions, an analytic control-grid transpose
  and local continuous-addition derivatives; dormant-grid hard-core checks use
  bounded-radius spatial queries rather than candidate-by-host tensors;
- authenticated AE-1 and complete AE-2 archives replay without pickle;
- direct Si/Ge finite-voxel cubature passes the declared \(10^{-4}\) production
  convergence budgets, and an independent 3-D cubature audit agreed within
  \(2.93\times10^{-8}\) over tested voxels;
- the maintained side-view notebook executed end to end on an A100 with exact
  16-scan accumulation: all eleven code cells completed, seven truth-free
  structural events produced the TARGET-view GIF, and the 54 MiB archive
  reloaded with matching problem/model identities;
- that deliberately tiny bounded smoke stopped honestly with
  `regularization_path_incomplete`, no defect calls and no promotion claim; it
  reports the fitted TARGET displacement and strain field rather than treating
  a low count residual as defect recovery;
- the final CPU regression passed 421 tests, with two optional
  Weickenmeier--Kohl paper comparisons skipped because their local
  parametrization helper is absent; static checks, focused Ruff checks and
  whitespace validation also passed.

The numerical atomic integration is converged, but this is not experimental or
first-principles validation. Both numerical paths use independent-atom
Kirkland parameters, and downstream propagation is shared.

## 11. Promotion gates

Promote a result only when:

- pristine data yield a stable empty edit set;
- nuisance-only mismatch does not become atoms or vacancies;
- isolated edits localize only to acquisition-supported resolution;
- irregular added mass is stable without object metadata;
- supported metastable strain survives the prior;
- Level 1 reduces overlaps/rough strain without degrading held-out counts;
- validation-equivalent starts agree or ambiguity is explicit;
- no depth feature is narrower than its measured response interval;
- active count is below a dense free-potential representation;
- every archive rerenders the specimen and reproduces objective components.

These gates are not all passed. The method is implemented and suitable for
controlled synthetic development, but not yet certified for experimental
structural claims.

## 12. Non-goals

Do not add an object-specific nanoparticle/inclusion class, shape or radius
prior, phase field, total variation, free residual potential, learned denoiser,
large species dictionary, molecular-dynamics relaxation, large per-scan
nuisance field, or atom-count claim below information-supported resolution
unless a frozen failed benchmark identifies that exact missing ingredient.

## 13. Notebooks

- `nine_atom_atomistic_edit_ptychography_1d.ipynb` is the executable,
  specialist-internals teaching example: nine synthetic known sites,
  empty-host identity, one vacancy, one off-grid addition, weak strain,
  objective terms, count-only boundary and a tiny active-set reconstruction.
  Normal silicon runs use the compact material-specific facade.
- `sideview_glancing_ptychography_1d.ipynb` is the maintained geometry workflow:
  automatic interaction support, calibrated AE configuration, TQDM, stopping
  evidence, TARGET-only reconstruction/evolution, and authenticated archive.

The side-view notebook labels bounded smoke runs as mechanics demonstrations;
they are not acceptance evidence for the complex truth case.
