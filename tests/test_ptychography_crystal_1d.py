"""Tests for the unified sparse crystal ptychography workflow."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation import ptychography_crystal_1d as module  # noqa: E402
from wide_angle_propagation.propagation_methods import (  # noqa: E402
    fresnel_propagation_kernel_1d,
)
from wide_angle_propagation.ptychography_1d import simulate_glancing_scan_1d  # noqa: E402
from wide_angle_propagation.ptychography_crystal_1d import (  # noqa: E402
    CrystalModel1D,
    CrystalReconstruction1D,
    CrystalState1D,
    _backtracked_proximal_keating_step,
    _hard_core_penalty_1d,
    _keating_quadratic_1d,
    _matched_filter_residual_1d,
    _proximal_keating_step,
    _rank_topology_proposals_1d,
    make_crystal_model_1d,
    reconstruct_crystal_1d,
    render_crystal_1d,
)


def _model():
    s = np.arange(17) * 0.5
    u = (np.arange(17) - 8) * 0.5
    grid_s, grid_u = np.meshgrid(np.arange(-1, 2) * 0.5, np.arange(-1, 2) * 0.5, indexing="ij")
    template = np.exp(-0.5 * (grid_s**2 + grid_u**2) / 0.28**2)
    host = np.asarray(
        [
            [2.0, 0.2, 0.0],
            [2.0, 3.8, 2.0],
            [3.5, 1.5, -0.5],
            [5.0, 3.0, 0.0],
            [6.5, 0.3, -0.6],
        ]
    )
    return make_crystal_model_1d(
        s,
        u,
        template,
        host,
        scan_coordinates_A=np.asarray([1.0, 2.0, 3.0]),
        training_indices=[0],
        beam_tilt_rad=0.0,
        airy_first_zero_A=0.8,
        slab_bounds_A=(-4.0, 3.0),
        axial_period_A=4.0,
        latent_period_A=4.0,
        insertion_grid_spacing_A=1.0,
        insertion_vacuum_A=1.0,
        bond_cutoff_A=2.1,
        max_host_removals=2,
        max_extra_atoms=2,
    )


def _state(model, *, removed=(), extra=None, displacement=None):
    removed_mask = np.zeros(len(model.reference_positions_3d), dtype=bool)
    removed_mask[list(removed)] = True
    extras = np.zeros((model.max_extra_atoms, 3), dtype=float)
    extras[:, 1] = model.latent_period_A / 4.0
    active = np.zeros(model.max_extra_atoms, dtype=bool)
    if extra is not None:
        extras[0] = extra
        active[0] = True
    if displacement is None:
        displacement = np.zeros((len(model.reference_positions_3d), 2))
    return CrystalState1D(
        registration=jnp.zeros(4),
        host_displacements=jnp.asarray(displacement),
        removed_host_mask=jnp.asarray(removed_mask),
        extra_positions_3d=jnp.asarray(extras),
        extra_active_mask=jnp.asarray(active),
    )


def test_public_api_is_the_single_crystal_workflow():
    assert set(module.__all__) == {
        "CrystalModel1D",
        "CrystalState1D",
        "CrystalReconstruction1D",
        "make_crystal_model_1d",
        "make_si_atom_template_1d",
        "render_crystal_1d",
        "reconstruct_crystal_1d",
    }
    for removed in (
        "CrystallineHostModel1D",
        "CrystallineRegistrationResult1D",
        "register_crystalline_host_1d",
        "reconstruct_crystalline_defects_1d",
    ):
        assert not hasattr(module, removed)


def test_model_builds_training_only_wedge_and_periodic_latent_bonds():
    model = _model()
    assert isinstance(model, CrystalModel1D)
    assert model.host_mobility.shape == (5,)
    assert model.scratch_mask.shape == (17, 17)
    assert np.min(np.asarray(model.scratch_mask)) == 0.0
    assert np.max(np.asarray(model.scratch_mask)) == pytest.approx(1.0)
    assert np.any(
        (np.asarray(model.scratch_mask) > 0.0)
        & (np.asarray(model.scratch_mask) < 1.0)
    )
    assert np.any(np.asarray(model.full_mobility_mask))
    assert len(model.insertion_anchors_3d) > 0
    pairs = {tuple(pair) for pair in np.asarray(model.bond_indices)}
    assert (0, 1) in pairs  # Their y separation is short only by minimum image.
    assert model.angle_vectors_3d.shape[1:] == (2, 3)


def test_renderer_has_exact_discrete_add_remove_behavior_and_finite_gradients():
    model = _model()
    pristine = _state(model)
    removed = _state(model, removed=(2,))
    added = _state(model, extra=np.asarray([4.0, 1.0, 1.8]))
    pristine_potential = render_crystal_1d(model, pristine)
    removed_potential = render_crystal_1d(model, removed)
    added_potential = render_crystal_1d(model, added)
    assert np.linalg.norm(np.asarray(pristine_potential - removed_potential)) > 0.0
    assert np.linalg.norm(np.asarray(added_potential - pristine_potential)) > 0.0

    def objective(displacements):
        state = CrystalState1D(
            registration=pristine.registration,
            host_displacements=displacements,
            removed_host_mask=pristine.removed_host_mask,
            extra_positions_3d=pristine.extra_positions_3d,
            extra_active_mask=pristine.extra_active_mask,
        )
        return jnp.sum(render_crystal_1d(model, state) * jnp.linspace(0.2, 1.0, 17)[:, None])

    gradient = jax.jit(jax.grad(objective))(pristine.host_displacements)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.linalg.norm(np.asarray(gradient)) > 0.0


def test_sparse_keating_is_positive_and_masks_removed_terms():
    model = _model()
    displacement = np.zeros((5, 2))
    displacement[2] = [0.18, -0.11]
    pristine_energy = _keating_quadratic_1d(model, jnp.asarray(displacement), jnp.zeros(5, dtype=bool))
    removed_energy = _keating_quadratic_1d(
        model,
        jnp.asarray(displacement),
        jnp.asarray([False, False, True, False, False]),
    )
    assert float(pristine_energy) > 0.0
    assert float(removed_energy) <= float(pristine_energy)
    assert float(_keating_quadratic_1d(model, jnp.zeros((5, 2)), jnp.zeros(5, dtype=bool))) == 0.0


def test_proximal_mechanics_reduces_keating_energy():
    model = _model()
    displacement = jnp.asarray(
        [[0.0, 0.0], [0.08, -0.03], [0.2, -0.12], [-0.09, 0.04], [0.0, 0.0]]
    )
    removed = jnp.zeros(5, dtype=bool)
    before = _keating_quadratic_1d(model, displacement, removed)
    after_displacement = _proximal_keating_step(
        model,
        displacement,
        removed,
        sigma_A=0.15,
        strength=0.1,
        cg_iterations=8,
    )
    after = _keating_quadratic_1d(model, after_displacement, removed)
    assert float(after) <= float(before)


def test_proximal_mechanics_halves_strength_and_can_skip(monkeypatch):
    model = _model()
    displacement = jnp.zeros((5, 2))

    def fake_step(_model, values, _removed, *, strength, **_):
        return values + strength

    monkeypatch.setattr(module, "_proximal_keating_step", fake_step)
    accepted, accepted_loss, strength, trials = _backtracked_proximal_keating_step(
        model,
        displacement,
        jnp.zeros(5, dtype=bool),
        jnp.ones(5, dtype=bool),
        lambda candidate: 1.01 if float(jnp.max(candidate)) > 0.075 else 1.0,
        1.0,
        sigma_A=0.15,
        initial_strength=0.1,
        cg_iterations=8,
    )
    assert strength == pytest.approx(0.05)
    assert accepted_loss == pytest.approx(1.0)
    assert [trial[0] for trial in trials] == pytest.approx([0.1, 0.05])
    assert np.allclose(np.asarray(accepted), 0.05)

    skipped, skipped_loss, strength, trials = _backtracked_proximal_keating_step(
        model,
        displacement,
        jnp.zeros(5, dtype=bool),
        jnp.ones(5, dtype=bool),
        lambda _candidate: 1.01,
        1.0,
        sigma_A=0.15,
        initial_strength=0.1,
        cg_iterations=8,
    )
    assert strength == 0.0 and skipped_loss == pytest.approx(1.0)
    assert len(trials) == 3
    assert np.array_equal(np.asarray(skipped), np.asarray(displacement))


def test_hard_core_penalizes_overlapping_added_atom():
    model = _model()
    host = np.asarray(model.reference_positions_3d)[0]
    overlapping = _state(model, extra=host)
    separated = _state(model, extra=np.asarray([7.5, 1.0, 1.8]))
    assert float(_hard_core_penalty_1d(model, overlapping)) > 0.0
    assert float(_hard_core_penalty_1d(model, separated)) == 0.0

    inactive = _state(model)

    def penalty(extra_positions):
        candidate = CrystalState1D(
            inactive.registration,
            inactive.host_displacements,
            inactive.removed_host_mask,
            extra_positions,
            inactive.extra_active_mask,
        )
        return _hard_core_penalty_1d(model, candidate)

    gradient = jax.grad(penalty)(inactive.extra_positions_3d)
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_signed_template_residual_ranks_isolated_removal_and_addition():
    model = _model()
    state = _state(model)
    shape = np.asarray(model.scratch_mask).shape
    template = np.asarray(model.atom_template)
    half_s, half_u = np.asarray(template.shape) // 2

    def stamp(position_su, sign):
        residual = np.zeros(shape, dtype=float)
        s = int(np.rint(position_su[0] / 0.5))
        u = int(np.rint((position_su[1] + 4.0) / 0.5))
        residual[
            s - half_s : s + half_s + 1,
            u - half_u : u + half_u + 1,
        ] = sign * template
        return residual * np.asarray(model.scratch_mask)

    host_index = 2
    host_su = np.asarray(model.reference_positions_3d)[host_index, [0, 2]]
    removal_score = _matched_filter_residual_1d(model, stamp(host_su, -1.0))
    removal_proposals = _rank_topology_proposals_1d(model, state, removal_score)
    assert removal_proposals[0][:2] == ("remove", host_index)

    anchors = np.asarray(model.insertion_anchors_3d)
    host = np.asarray(model.reference_positions_3d)
    tiled_host = np.concatenate(
        [host - [0.0, model.latent_period_A, 0.0], host,
         host + [0.0, model.latent_period_A, 0.0]]
    )
    from scipy.spatial import cKDTree

    allowed = cKDTree(tiled_host).query(anchors, k=1)[0] >= 1.8
    interior = (
        (anchors[:, 0] >= 1.0) & (anchors[:, 0] <= 7.0)
        & (anchors[:, 2] >= -3.0) & (anchors[:, 2] <= 3.0)
    )
    anchor_index = int(np.flatnonzero(allowed & interior)[0])
    addition_score = _matched_filter_residual_1d(
        model, stamp(anchors[anchor_index, [0, 2]], 1.0)
    )
    addition_proposals = _rank_topology_proposals_1d(model, state, addition_score)
    assert ("add", anchor_index) in [proposal[:2] for proposal in addition_proposals]


def _scan_problem(model, truth_state):
    potential = render_crystal_1d(model, truth_state)
    u = model.transverse_coordinates
    probes = jnp.stack(
        [jnp.exp(-0.5 * ((u - center) / 1.0) ** 2) for center in (-0.5, 0.0, 0.5)]
    ).astype(jnp.complex128)
    starts = jnp.zeros(3, dtype=jnp.int32)
    kernel = fresnel_propagation_kernel_1d(len(u), 0.5, 0.5, 5e3)
    measured = simulate_glancing_scan_1d(
        potential, probes, starts, len(model.axial_coordinates), kernel, 0.5, 5e3
    )
    detector_angles = jnp.linspace(-100.0, 100.0, len(u))
    return probes, starts, kernel, measured, detector_angles


def test_tiny_pristine_workflow_stops_without_edits_and_retains_no_pixels():
    model = _model()
    problem = _scan_problem(model, _state(model))
    result = reconstruct_crystal_1d(
        model,
        problem[0],
        problem[1],
        len(model.axial_coordinates),
        problem[2],
        0.5,
        5e3,
        problem[3],
        problem[4],
        training_indices=[0],
        selection_indices=[1],
        audit_indices=[2],
        target_nrmse=1.0,
        registration_phase_points=3,
        registration_updates=1,
        initial_cycles=0,
        accepted_cycles=0,
        screening_cycles=0,
        final_cycles=1,
        data_updates_per_cycle=1,
        mechanics_strength=0.0,
        scratch_updates=0,
        max_active_iterations=1,
    )
    assert isinstance(result, CrystalReconstruction1D)
    assert result.termination_reason == "target_reached"
    assert not np.any(np.asarray(result.state.removed_host_mask))
    assert not np.any(np.asarray(result.state.extra_active_mask))
    assert result.metadata["pixel_residual_retained"] is False
    assert result.scratch_residual_history.shape[0] == 0
    assert np.allclose(
        np.asarray(result.potential), np.asarray(render_crystal_1d(model, result.state))
    )


def test_notebook_is_clean_and_uses_only_the_unified_workflow():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "sideview_glancing_ptychography_1d.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    identifiers = [cell.get("id") for cell in notebook["cells"]]
    assert all(identifiers) and len(identifiers) == len(set(identifiers))
    code = []
    markdown = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", ()))
        if cell["cell_type"] == "code":
            assert not cell.get("outputs")
            ast.parse(
                "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%")),
                filename=f"notebook cell {index}",
            )
            code.append(source)
        else:
            markdown.append(source)
    joined_code = "\n".join(code)
    joined_markdown = "\n".join(markdown).lower()
    assert "reconstruct_crystal_1d" in joined_code
    assert "make_crystal_reconstruction_viewer_1d" in joined_code
    assert "progress=True" in joined_code
    assert "jax.value_and_grad" not in joined_code
    assert "optax" not in joined_code
    assert "occupanc" not in joined_code.lower()
    assert "temporary" in joined_markdown and "discard" in joined_markdown
    assert "fixed" in joined_markdown and "latent" in joined_markdown
    assert "noise" in joined_markdown and "probe" in joined_markdown
