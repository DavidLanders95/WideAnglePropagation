# Full-slab sparse crystal-edit ptychography

This document defines the maintained glancing-incidence reconstruction and its
acceptance criteria. The implementation addresses one deliberately constrained
question: given a known diamond-silicon host, fixed probe, fixed scan geometry,
and fixed latent depth, can diffraction recover a smooth projected displacement
field together with a small number of removals and additions?

The complete 1000 Å specimen is retained in every forward calculation. There
is no cropped 250 Å reconstruction, permanent pixel correction, continuous
site weight, or separate registration solver.

## Geometry and supplied information

The experiment reproduces the side-view viewer geometry:

- 30 keV incident energy;
- 2° glancing incidence;
- a 15 mrad circular-aperture probe;
- a 1000 Å propagation length and 50 Å silicon depth; and
- 21 surface landings from 400 Å to 600 Å.

The inverse problem is supplied with the silicon species, complete diamond
host, approximate lattice scale, exterior crystal, fixed probe, scan
coordinates, and a fixed latent coordinate for added atoms. It fits four
global registration parameters, projected host displacements, discrete host
removals, and at most four added silicon atoms. Added atoms have continuous
projected coordinates and fixed latent coordinate \(y=a/4\).

The host is built independently of the abTEM forward specimen. Its latent
\(y\) coordinates are preserved, and nearest-neighbour construction uses the
periodic minimum image in \(y\). The renderer deposits complete physical sites
on the \((s,u)\) grid and convolves them with one independently generated
Lobato silicon template. A removed host contributes no template; an active
added atom contributes one. Temporary residual pixels are never passed to this
renderer.

## Training-defined mutable wedge

Only the fifteen training scans define the mutable beam-path wedge. For
wavelength \(\lambda\) and convergence semiangle \(\alpha\), its reference
radius is the first Airy zero,

\[
r_0 = 0.61\frac{\lambda}{\alpha}.
\]

Host sites within \(2.5r_0\) of a post-landing ray have full mobility. Mobility
falls smoothly to zero between \(2.5r_0\) and \(4r_0\), and all more distant
sites remain fixed. Removal and insertion proposals are restricted to the
full-mobility core. The temporary-pixel mask includes the same taper and an
atomic-template halo so that a physical atom centred at the wedge boundary can
still be proposed without truncating its diagnostic signature.

There is no target-versus-nuisance hierarchy inside the wedge. Any host in the
core can be proposed for removal, and additions can be proposed throughout the
core, subject to the hard-core constraint and fixed capacities.

## Three separate reconstruction operations

The workflow deliberately separates data fitting, mechanics, and topology
proposal.

### Diffraction coordinate updates

For a fixed topology, each physical cycle performs ten scaled Adam updates of
the diffraction objective and a three-dimensional 1.8 Å hard-core penalty.
Only projected \((s,u)\) host displacements are fitted. Their support and
maximum excursion taper with host mobility. Active added atoms move in
\((s,u)\), while their latent \(y\) coordinate remains fixed.

The detector objective balances the complete detector and the reflected
0--80 mrad band:

\[
L_{\rm data}
= \tfrac12 L_{\rm all}
+ \tfrac12 L_{0\text{--}80\,{\rm mrad}},
\qquad
L_\Omega
= \frac{\sum_\Omega(\sqrt{I_{\rm pred}}-\sqrt{I_{\rm meas}})^2}
        {\sum_\Omega I_{\rm meas}}.
\]

### Weak Keating proximal update

One mechanics step follows the ten data updates. The silicon host uses sparse
linearized bond-stretch and bond-angle operators, including terms that cross
from mobile atoms into the fixed exterior. Terms touching a removed host are
masked. Added atoms receive no Keating bonds. The stretch-to-bend ratio follows
the standard silicon parameterization of the
[Keating model](https://journals.aps.org/pr/abstract/10.1103/PhysRev.145.637).

The proximal subproblem is applied matrix-free with eight conjugate-gradient
iterations, \(\sigma_K=0.15\) Å, and initial strength 0.1. A trial is rejected
if training NRMSE increases by more than 0.5%. The strength is halved after a
rejection, and the mechanics operation is skipped after three failed trials.
This makes the elastic model a weak regularizer rather than an alternative fit
to the diffraction data.

The initial host receives six physical cycles. An accepted edit receives four,
and the retained topology receives eight final cycles. Candidate screening and
pruning use two cycles with only the local 6 Å neighbourhood free.

### Discarded signed-pixel proposal

Before each topology decision, the physical state is frozen and a scratch
residual is initialized to zero. Five Adam updates fit training diffraction
inside the tapered scratch mask. The residual is signed and clipped to
\(\pm1.25\) times the silicon-template peak.

Correlation with the silicon template turns the scratch field into proposals:

- negative correlation at an active host ranks possible removals; and
- positive, spatially non-maximal peaks rank possible additions.

At most two removals and two additions are screened. The scratch field is
stored only as a downsampled diagnostic frame and is then discarded. It is not
a reconstruction variable, is not propagated into candidate fitting, and is
absent from the final potential.

The valid candidate with lowest selection NRMSE is accepted only if it improves
that value by at least \(10^{-5}\). Search stops when the calibrated target is
reached, no valid edit improves selection loss, or the capacities of four
removals and four additions are exhausted. Once the target is reached, each
accepted edit is removed in turn and discarded if the reduced topology still
meets the target.

## Registration, partitions, and mismatch calibration

Registration is the first stage of `reconstruct_crystal_1d`. A sequential
phase search followed by scaled Adam fits axial phase, surface-normal offset,
in-plane rotation, and axial strain. Once the other three coordinates have
settled, a second bounded phase profile selects the best symmetry-equivalent
axial basin. Registration is then fixed during physical and topology updates.
Registration history remains available in the returned result, but users do
not coordinate a second solver.

The 21 scans are partitioned deterministically into fifteen training scans,
three topology-selection scans, and three audit scans. Training data determine
continuous coordinates and scratch proposals. Selection data decide topology
and pruning. Audit data remain unopened until the final physical state has
been selected.

Independent forward and inverse rasterizers do not agree exactly, even at the
known pristine coordinates. The notebook therefore fixes the target before
either reconstruction gate:

\[
\epsilon
= \max\!\left(10^{-3},,1.25\,\epsilon_{\rm pristine\ oracle}\right).
\]

The oracle compares diffraction from the independently rasterized pristine
abTEM slab with diffraction from the inverse renderer at the known pristine
state. This calibration admits unavoidable renderer mismatch without using
defect truth or audit scans to tune the threshold.

## Maintained synthetic gates

The notebook evaluates two noiseless gates.

The strained-pristine gate contains global misregistration and a smooth,
wedge-supported displacement field. It must recover the field without
introducing a removal or addition.

The sparse-defect gate uses the same smooth field, removes the surface host
nearest the 500 Å landing, and adds one silicon atom 0.65 Å farther along the
axial direction and 1.9 Å above that site. A prescribed local relaxation is
added around the defect; it is not generated by the reconstruction's Keating
model. This gate is run both with the 10:1 data/mechanics schedule and with
mechanics disabled. Both variants retain the same signed-pixel proposal steps.

The maintained acceptance conditions are:

- no edits in the strained-pristine gate;
- exactly the intended vacancy and one addition, with no false edits;
- added-atom error no greater than 0.25 Å;
- host-displacement RMSE no greater than 0.15 Å in the full-mobility core;
- minimum three-dimensional separation of at least 1.8 Å;
- selection and unopened-audit NRMSE below the calibrated target; and
- lower displacement roughness with mechanics, without more than 5% audit
  degradation relative to the no-mechanics result.

Compact tests cover renderer discreteness and gradients, periodic latent-depth
neighbours, training-only wedge construction, sparse Keating actions,
backtracking, hard-core exclusion, signed residual correlation, deterministic
proposal ranking, fixed topology capacities, and pruning. The complete 1000 Å
gates remain a slow accelerator/notebook test.

## Scope and limitations

This is a matched, noiseless, single-species synthetic experiment. It assumes a
fixed coherent probe, exact scan positions, known detector geometry, known
exterior, and fixed latent depth. It does not infer chemistry, probe
aberrations, partial coherence, scan errors, experimental noise, or arbitrary
three-dimensional structure. The Keating operator is linearized and excludes
added atoms, so it is not a reactive defect potential.

Passing unopened synthetic scans tests consistency under these assumptions; it
does not establish structural uniqueness or experimental validity. Extending
the method requires new observability tests rather than adding an unreported
permanent pixel correction.
