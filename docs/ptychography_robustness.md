# Free-atom ptychography: first working experiment

This document describes the small experiment that must work before free atoms
are used in the large side-view reconstruction.

## The idea

The specimen is represented by candidate silicon atoms rather than independent
potential pixels. Each candidate has only three fitted values:

- its axial position;
- its transverse position;
- an occupancy between zero and one.

Every active candidate produces the same known Lobato silicon scattering
potential. The measured diffraction data decide which candidates become atoms
and where they move.

The reconstruction is **not** given a silicon lattice, particle shape, atom
count, surface, vacancy list, or ground-truth position. Twenty-four candidates
start on a uniform rectangular grid spanning the illuminated search region.
That grid is only a numerical starting point; it is not a crystal model.

Knowing the atomic species and the beam-observable region is still a strong
prior. The comparison with a pixel reconstruction reports that reduction in
freedom explicitly.

## The fitted loss

The initial solver uses three terms:

\[
L = L_{\rm amplitude}
  + 10^{-3}\,\mathrm{mean}(o_i)
  + 10^{-2} E_{\rm repulsion}.
\]

The first term fits the diffraction amplitudes. The occupancy term removes
unneeded weak candidates. The repulsion term prevents occupied candidates from
approaching more closely than 1.8 Å.

The first 200 updates change occupancies only. The remaining 800 updates also
move atom positions. Occupancies and positions are clipped to their allowed
ranges after every Adam update. Every fifth scan is held out, and the returned
state is the checkpoint with the lowest held-out amplitude loss.

A weak, softened Lennard--Jones-like term is tested only after repulsion works.
It is an ablation, not an accurate energy model for covalent silicon. It ramps
from zero during the final 400 updates and remains disabled unless it preserves
the deliberately missing atoms and improves held-out recovery.

## The nine-atom gate

The truth begins as a small triangular arrangement, then removes one internal
and two surface atoms and applies displacements of at most 0.20 Å. The truth is
not relaxed with the reconstruction energy.

The maintained notebook compares:

1. independent potential pixels;
2. atoms without pair interactions;
3. atoms with short-range repulsion;
4. atoms with repulsion and weak cohesion.

The repulsion result must recover exactly nine atoms at occupancy 0.5, produce
no false or missing atoms, achieve position RMSE at most 0.25 Å, keep every pair
at least 1.8 Å apart, and reach validation amplitude NRMSE below \(10^{-3}\).
Its 72 specimen parameters must also be at least 50 times fewer than the
selected pixel values.

If this gate fails, the side-view atom reconstruction is not implemented. A low
diffraction residual by itself is not a successful structural result.

## The first glancing-incidence gate

The maintained side-view experiment uses the pristine 250 Å, 30 keV, 2 degree
geometry from the viewer notebook. Silicon outside a calibrated 20 Å surface
box is fixed and known. Inside that box, sixteen uniform candidates reconstruct
eight atoms in the uppermost projected row; no axial lattice sites or atom count
are supplied.

Compact local atomic patches make this calculation scale with the template
area rather than with candidate count times the complete 737 by 512 specimen.
Each candidate is allowed to move by 0.75 Å around its uniform seed, while the
known exterior remains part of every forward simulation.

The accepted noiseless run recovers all eight atoms without false positives,
with 0.064 Å position RMSE, 2.53 Å minimum spacing, and held-out amplitude NRMSE
of \(4.04\times10^{-4}\). Its 48 specimen parameters replace 4,920 independent
potential values in the complete atomic influence halo.

This success does not extend to depth. Development runs freeing three projected
layers (23 atoms) and a wider 4.5 Å band (52 atoms) reduced diffraction loss but
did not recover reliable structures. They are treated as failed observability
tests, not promoted results. Acquisition diversity or depth-response analysis
must improve before deeper atoms are freed.

## What this does not establish

This is a matched, noiseless, single-species test with a fixed known probe and
known scan geometry. It uses the maintained two-dimensional \((s,u)\) model;
it is not a three-dimensional atom reconstruction.

Noise, unknown probes, scan errors, species inference, candidate birth/death,
save/load infrastructure, and the realistic side-view specimen were deferred by
this first gate. The crystalline-host extension below keeps its own tests and
does not weaken the original free-atom acceptance criteria.

## The pristine crystalline-host baseline

The additive ``ptychography_crystal_1d`` module keeps the diffraction model in
projected ``(s, u)`` coordinates but evaluates crystalline elasticity on latent
three-dimensional ``(s, y, u)`` diamond sites.  A fixed sparse bond/angle graph
provides occupancy-gated Keating stretch and tetrahedral-angle penalties without
forming all-pairs distance matrices. The single-species interface fixes every
site to occupied Si, fits a global lattice transform, and then releases bounded
local displacements under the Keating penalty.

The maintained large-slab notebook forward-simulates the complete 1000
Angstrom-long, 50 Angstrom-deep pristine slab, while reconstruction uses an
independently generated diamond host in the loose 100--900 Angstrom axial
volume. The fixed graph assumes the diamond topology, silicon bond length,
approximate (100) orientation and termination, and atom count implied by the
volume. It is therefore a crystallographic refinement test rather than lattice
discovery.

The coordinate start is deliberately perturbed by a 1.7 Angstrom axial phase
error, 1.2 percent axial strain, shear and transverse strain, 0.45 degree
rotation, and 0.18 Angstrom RMS local disorder.  Coordinate updates are now
restricted to the union of post-landing incident half-rays.  The support is
empty before the first 400 Angstrom landing, grows while scans land from
400--600 Angstrom, and continues downstream after 600 Angstrom.  A smooth
transverse support is one inside 2.5 input-probe waists and zero beyond four
waists; the affine transform, local displacements, occupancy/species gradients,
and Keating terms all use the same per-site weights.  Of 10,915 sites in the
loose box, 8,099 are therefore exactly frozen and all 4,089 sites before 400
Angstrom have zero update weight.  These
sites retain the assumed diamond-host prior so that an arbitrary unobservable
guess cannot contaminate the forward calculation, but they are greyed out and
are neither scored nor claimed as reconstructed.

With this post-landing observability constraint, 300 ptychographic updates
reach a held-out amplitude NRMSE of 0.01008, compared with 0.01141 for simply
cropping the exact forward potential to the loose inverse volume.  In the
interpretable illumination core (weight above 0.8), the final median host bond
is 2.3537 Angstrom with a 0.0249 Angstrom standard deviation, and the
axial-gauge-aligned median site error is 0.0616 Angstrom.  The soft support
boundary is excluded from structural scores.

The post-hoc registration is used only for structural scoring and potential
visualization, not for reconstruction or diffraction evaluation. Absolute
axial unit-cell phase is weakly observable for a long uniform slab illuminated
far from its ends, so the notebook reports that degree of freedom as a gauge
rather than as a discovered coordinate. The gauge is restricted to the axial
direction: shifting in depth would change the finite surface termination and
can incorrectly hide surface rows. With axial registration only, the median
interpretable-core site error is 0.0616 Angstrom, while the median illuminated
top surface plane lies 0.0150 Angstrom from its known position. The more general module retains
occupancy, Si/Ge substitution, and off-lattice adatom variables, with compact
isolated tests for each case, but the maintained large-slab notebook does not
activate those variables until the pristine baseline is established.
