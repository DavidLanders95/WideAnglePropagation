"""Focused AE-2 active-set, scoring, KKT, and debiasing tests."""

from dataclasses import replace
import inspect
from types import SimpleNamespace

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax", reason="the ptychography extra is not installed")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation import (  # noqa: E402
    ptychography_atomistic_edit_solver_1d as solver_module,
)

from tests.atomistic_edit_test_helpers import (  # noqa: E402
    CompactAtomisticEditModelSpec1D,
    CompactPreparedProblemSpec1D,
    make_compact_atomistic_edit_model_1d,
    make_compact_prepared_atomistic_edit_problem_1d,
)
from wide_angle_propagation.ptychography_atomistic_edit_1d import (  # noqa: E402
    atomistic_edit_state_is_admissible_1d,
    empty_atomistic_edit_state_1d,
    render_atomistic_edit_potential_1d,
)
from wide_angle_propagation.ptychography_1d import (  # noqa: E402
    _shift_patch_axis_keys_cubic_1d,
)
from wide_angle_propagation.ptychography_atomistic_edit_solver_1d import (  # noqa: E402
    AtomisticEditSolverOptions1D,
    _clear_compiled_objective_cache,
    _compiled_objective_cache_info,
    _merge_duplicate_additions,
    _objective_value_and_gradients,
    _prune_state,
    _refine_state,
    _reference_objective_value_and_gradients,
    _scan_batches,
    atomistic_edit_objective_components_1d,
    atomistic_edit_proposal_scores_1d,
    run_prepared_atomistic_edit_multistart_reconstruction_1d,
    run_prepared_atomistic_edit_reconstruction_1d,
)


SHAPE = (13, 13)
HOST_CENTRES = np.asarray([[6, 6], [6, 9]], dtype=np.int32)
ADDITION_A = (3, 3)
ADDITION_B = (9, 3)


def _model(
    *,
    penalty_path=(0.05, 0.005),
    max_removals=2,
    max_extras=2,
    deformation_parameter_count=8,
):
    return make_compact_atomistic_edit_model_1d(
        CompactAtomisticEditModelSpec1D(
            shape=SHAPE,
            host_centres=HOST_CENTRES,
            target_discovery_centres=(ADDITION_A, ADDITION_B),
            nuisance_discovery_centres=(),
            edit_penalty_path=penalty_path,
            max_host_removals=max_removals,
            max_extra_centres=max_extras,
            deformation_parameter_count=deformation_parameter_count,
            fixture_id="ae2-solver-test",
            reference_background=0.02,
            maximum_displacement_A=0.2,
        ),
    )


def _with_extra(state, model, anchor, mass):
    anchors = np.asarray(state.extra_anchor_indices).copy()
    masses = np.asarray(state.extra_scattering_equivalents).copy()
    active = np.asarray(state.extra_active).copy()
    anchors[0] = anchor
    masses[0] = mass
    active[0] = True
    return replace(
        state,
        extra_anchor_indices=jnp.asarray(anchors),
        extra_scattering_equivalents=jnp.asarray(masses),
        extra_active=jnp.asarray(active),
    )


def _with_removal(state, site, fraction):
    indices = np.asarray(state.host_removal_indices).copy()
    fractions = np.asarray(state.host_removal_fractions).copy()
    active = np.asarray(state.host_removal_active).copy()
    indices[0] = site
    fractions[0] = fraction
    active[0] = True
    return replace(
        state,
        host_removal_indices=jnp.asarray(indices),
        host_removal_fractions=jnp.asarray(fractions),
        host_removal_active=jnp.asarray(active),
    )


def _prepared(
    model,
    truth_state=None,
    *,
    objective_kind="poisson_deviance",
    audit_count_scale=1.0,
):
    return make_compact_prepared_atomistic_edit_problem_1d(
        model,
        CompactPreparedProblemSpec1D(
            window_starts=(0, 1, 0, 1),
            window_length=12,
            probe_shifts=(-1, 0, 1, 2),
            validation_indices=(2,),
            audit_indices=(3,),
            electrons_per_pattern=200_000.0,
            fixture_id="ae2-solver-test",
            objective_kind=objective_kind,
            audit_count_scale=audit_count_scale,
        ),
        truth_state=truth_state,
    )


def _count_component(prepared, state):
    return float(
        atomistic_edit_objective_components_1d(
            prepared,
            state,
            0.0,
            ablation="edit_only",
        ).count_deviance
    )


def test_preparation_is_truth_free_poisson_only_and_counts_all_controls():
    model = _model()
    prepared = _prepared(model)
    assert prepared.metadata["truth_inputs_accepted"] is False
    assert prepared.metadata["nuisance_image_present"] is False
    assert prepared.metadata["energy_envelope_present"] is False
    assert prepared.objective.kind == "poisson_deviance"
    assert len(prepared.reconstruction_problem_id) == 64

    with pytest.raises(ValueError, match="poisson_deviance"):
        _prepared(model, objective_kind="poisson_gaussian_nll")
    with pytest.raises(ValueError, match="deformation subspace"):
        _prepared(_model(deformation_parameter_count=6))


def test_objective_reports_count_edit_elastic_and_hard_core_separately():
    model = _model()
    state = _with_extra(
        _with_removal(empty_atomistic_edit_state_1d(model), 0, 1.0),
        model,
        tuple(HOST_CENTRES[0]),
        0.7,
    )
    controls = np.zeros((2, 2, 2), dtype=float)
    controls[1, :, 0] = 0.08
    state = replace(state, host_displacement_controls=jnp.asarray(controls))
    prepared = _prepared(model, state)

    level1 = atomistic_edit_objective_components_1d(
        prepared, state, 0.05, ablation="level1_physical"
    )
    edit_only = atomistic_edit_objective_components_1d(
        prepared, state, 0.05, ablation="edit_only"
    )
    assert float(level1.edit_mass) == pytest.approx(1.7)
    assert float(level1.weighted_edit_penalty) == pytest.approx(0.085)
    assert float(level1.elastic_penalty) > 0.0
    assert float(level1.hard_core_penalty) >= 0.0
    assert float(level1.total_objective) == pytest.approx(
        float(level1.count_deviance)
        + float(level1.weighted_edit_penalty)
        + float(level1.elastic_penalty)
        + float(level1.hard_core_penalty)
    )
    assert float(edit_only.total_objective) == pytest.approx(
        float(edit_only.count_deviance)
        + float(edit_only.weighted_edit_penalty)
    )


def test_compiled_objective_reuses_one_trace_across_active_sets_and_lambda():
    model = _model()
    prepared = _prepared(model)
    empty = empty_atomistic_edit_state_1d(model)
    removal = _with_removal(empty, 0, 0.35)
    addition = _with_extra(empty, model, ADDITION_A, 0.4)
    cases = (
        (empty, 0.05),
        (removal, 0.05),
        (addition, 0.05),
        (addition, 0.005),
    )
    _clear_compiled_objective_cache()
    trace_count_after_first = None
    for case_index, (state, penalty) in enumerate(cases):
        reference_value, reference_gradients = (
            _reference_objective_value_and_gradients(
                prepared,
                state,
                penalty,
                "level1_physical",
            )
        )
        compiled_value, compiled_gradients = _objective_value_and_gradients(
            prepared,
            state,
            penalty,
            "level1_physical",
        )
        # Synchronize every leaf before inspecting the trace diagnostic.  A
        # JIT call is asynchronous on accelerators and must not make this cache
        # regression depend on dispatch timing.
        compiled_value = jax.block_until_ready(compiled_value)
        compiled_gradients = jax.tree_util.tree_map(
            jax.block_until_ready, compiled_gradients
        )
        np.testing.assert_allclose(
            compiled_value, reference_value, rtol=2e-12, atol=2e-12
        )
        for name in reference_gradients:
            np.testing.assert_allclose(
                compiled_gradients[name],
                reference_gradients[name],
                rtol=2e-11,
                atol=2e-11,
            )
        public_value = atomistic_edit_objective_components_1d(
            prepared,
            state,
            penalty,
            ablation="level1_physical",
        ).total_objective
        np.testing.assert_allclose(
            compiled_value, public_value, rtol=2e-12, atol=2e-12
        )
        info = _compiled_objective_cache_info(
            prepared, "level1_physical"
        )
        if case_index == 0:
            assert info["trace_count"] >= 1
            trace_count_after_first = info["trace_count"]
        else:
            # Active masks, integer indices/anchors, and λ all changed without
            # producing a second fixed-shape XLA trace.
            assert info["trace_count"] == trace_count_after_first

    info = _compiled_objective_cache_info(prepared, "level1_physical")
    assert info["lookup_count"] >= len(cases) + 1
    _clear_compiled_objective_cache()


@pytest.mark.parametrize("ablation", ["edit_only", "level1_physical"])
def test_exact_scan_batched_objective_and_gradients_match_full_training(ablation):
    model = _model()
    prepared = _prepared(model)
    state = _with_extra(
        empty_atomistic_edit_state_1d(model), model, ADDITION_A, 0.35
    )
    full_value, full_gradients = _objective_value_and_gradients(
        prepared, state, 0.005, ablation
    )
    batched_value, batched_gradients = _objective_value_and_gradients(
        prepared,
        state,
        0.005,
        ablation,
        training_scan_batch_size=1,
    )
    np.testing.assert_allclose(batched_value, full_value, rtol=2e-11, atol=2e-11)
    for name in full_gradients:
        np.testing.assert_allclose(
            batched_gradients[name],
            full_gradients[name],
            rtol=2e-10,
            atol=2e-10,
        )


def test_scan_batches_bound_the_compiled_scan_axis_without_reweighting():
    model = _model()
    prepared = _prepared(model)
    batches = _scan_batches(prepared.training_indices, 1)
    assert tuple(batch.shape for batch in batches) == ((1,), (1,))
    np.testing.assert_array_equal(
        np.concatenate(batches), np.asarray(prepared.training_indices)
    )
    assert max(batch.size for batch in batches) == 1


def test_host_adjoint_geometry_cache_is_immutable_reused_and_clearable():
    prepared = _prepared(_model())
    first = solver_module._host_adjoint_geometry(prepared)
    second = solver_module._host_adjoint_geometry(prepared)
    assert first is second
    assert not first.patches.flags.writeable
    assert not first.patch_starts.flags.writeable
    _clear_compiled_objective_cache()
    third = solver_module._host_adjoint_geometry(prepared)
    assert third is not first
    np.testing.assert_array_equal(third.patches, first.patches)


def test_scan_batched_gradient_factorizes_renderer_from_scan_jit(monkeypatch):
    assert not hasattr(
        solver_module, "_compiled_scan_batch_objective_value_and_gradient"
    )
    model = _model()
    prepared = _prepared(model)
    state = empty_atomistic_edit_state_1d(model)
    render_calls = 0
    original_render = solver_module.render_atomistic_edit_potential_1d

    def counted_render(*args, **kwargs):
        nonlocal render_calls
        render_calls += 1
        return original_render(*args, **kwargs)

    def combined_graph_forbidden(*args, **kwargs):
        raise AssertionError("scan batching entered the fused renderer/scan JIT")

    monkeypatch.setattr(
        solver_module, "render_atomistic_edit_potential_1d", counted_render
    )
    monkeypatch.setattr(
        solver_module,
        "_compiled_objective_value_and_gradient",
        combined_graph_forbidden,
    )
    value, gradients = _objective_value_and_gradients(
        prepared,
        state,
        0.005,
        "level1_physical",
        training_scan_batch_size=1,
    )
    assert np.isfinite(float(value))
    assert all(np.all(np.isfinite(value)) for value in gradients.values())
    # The renderer is used for the value only. Its host transpose is evaluated
    # from local cotangent patches, never by a full-grid/all-sites VJP.
    assert render_calls == 1


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("shift", [0.0, 1.0, -1.0, 0.37, -0.42])
def test_analytic_keys_shift_derivative_matches_autodiff_and_finite_difference(
    axis, shift
):
    patch = jnp.asarray(
        np.arange(30, dtype=np.float64).reshape(5, 6) / 17.0
    )
    shift_value = jnp.asarray(shift, dtype=jnp.float64)
    analytic = solver_module._shift_patch_axis_keys_cubic_numpy_1d(
        np.asarray(patch)[None, ...],
        np.asarray([shift], dtype=np.float64),
        axis=axis,
        derivative=True,
    )[0]
    _, autodiff = jax.jvp(
        lambda value: _shift_patch_axis_keys_cubic_1d(
            patch, value, axis=axis
        ),
        (shift_value,),
        (jnp.asarray(1.0, dtype=jnp.float64),),
    )
    np.testing.assert_allclose(analytic, autodiff, rtol=2e-13, atol=2e-13)
    if not float(shift).is_integer():
        step = 1e-6
        finite = (
            _shift_patch_axis_keys_cubic_1d(
                patch, shift_value + step, axis=axis
            )
            - _shift_patch_axis_keys_cubic_1d(
                patch, shift_value - step, axis=axis
            )
        ) / (2.0 * step)
        np.testing.assert_allclose(analytic, finite, rtol=2e-8, atol=2e-9)


def test_local_renderer_transpose_matches_full_vjp_off_grid():
    model = _model()
    prepared = _prepared(model)
    state = _with_extra(
        _with_removal(empty_atomistic_edit_state_1d(model), 0, 0.3),
        model,
        ADDITION_A,
        0.4,
    )
    state = replace(
        state,
        host_displacement_controls=jnp.asarray(
            np.linspace(-0.06, 0.05, 8, dtype=np.float64).reshape(2, 2, 2)
        ),
    )
    parameters = solver_module._state_parameters(state)
    structure = solver_module._state_structure(state)
    cotangent = jnp.asarray(
        np.random.default_rng(7).normal(size=SHAPE), dtype=jnp.float64
    )
    _, full_pullback = jax.vjp(
        lambda values: render_atomistic_edit_potential_1d(
            model,
            solver_module._state_from_structure_and_parameters(
                structure, values
            ),
        ),
        parameters,
    )
    reference = full_pullback(cotangent)[0]
    local = solver_module._parameter_data_gradients_from_potential_adjoint(
        prepared, parameters, structure, cotangent
    )
    for name in reference:
        np.testing.assert_allclose(
            local[name], reference[name], rtol=3e-12, atol=3e-12
        )


def test_spatial_hard_core_proposal_terms_match_dense_reference(monkeypatch):
    model = _model()
    state = _with_extra(
        _with_removal(empty_atomistic_edit_state_1d(model), 0, 0.35),
        model,
        ADDITION_A,
        0.4,
    )
    controls = np.linspace(-0.04, 0.05, 8).reshape(2, 2, 2)
    state = replace(state, host_displacement_controls=jnp.asarray(controls))
    candidates = np.argwhere(model.options.discovery_support.discovery_mask)
    paired = solver_module._paired_replacement_anchors(model)

    spatial_hard = solver_module._hard_core_directional_derivatives(
        model, state, candidates, paired
    )
    spatial_admissible = solver_module._addition_admissible_mask(
        model, state, candidates
    )
    spatial_paired_admissible = (
        solver_module._paired_addition_admissible_mask(model, state, paired)
    )

    def dense_sums(queries, sources, weights, minimum):
        queries = np.asarray(queries, dtype=float).reshape(-1, 2)
        sources = np.asarray(sources, dtype=float).reshape(-1, 2)
        if not len(sources):
            return np.zeros(len(queries), dtype=float)
        distances = np.linalg.norm(
            queries[:, None, :] - sources[None, :, :], axis=2
        )
        return np.sum(
            np.asarray(weights)[None, :]
            * solver_module._hard_core_phi_numpy(distances, minimum),
            axis=1,
        )

    def dense_admissible(queries, obstacles, minimum):
        queries = np.asarray(queries, dtype=float).reshape(-1, 2)
        obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 2)
        if not len(obstacles):
            return np.ones(len(queries), dtype=bool)
        distances = np.linalg.norm(
            queries[:, None, :] - obstacles[None, :, :], axis=2
        )
        return np.all(distances >= minimum, axis=1)

    with monkeypatch.context() as patch:
        patch.setattr(
            solver_module, "_weighted_hard_core_neighbor_sums", dense_sums
        )
        patch.setattr(
            solver_module,
            "_minimum_separation_admissible_mask",
            dense_admissible,
        )
        dense_hard = solver_module._hard_core_directional_derivatives(
            model, state, candidates, paired
        )
        dense_mask = solver_module._addition_admissible_mask(
            model, state, candidates
        )
        dense_paired_mask = np.zeros(len(paired), dtype=bool)
        for local_index, anchor in enumerate(paired):
            if np.all(anchor >= 0):
                dense_paired_mask[local_index] = (
                    solver_module._addition_admissible_mask(
                        model,
                        state,
                        anchor[None, :],
                        excluded_host_local_index=local_index,
                    )[0]
                )

    for spatial, dense in zip(spatial_hard, dense_hard, strict=True):
        np.testing.assert_allclose(spatial, dense, rtol=2e-13, atol=2e-10)
    np.testing.assert_array_equal(spatial_admissible, dense_mask)
    np.testing.assert_array_equal(
        spatial_paired_admissible, dense_paired_mask
    )


def test_fft_addition_correlation_matches_direct_reference():
    from scipy.signal import correlate2d

    model = _model()
    potential_adjoint = np.random.default_rng(11).normal(size=SHAPE)
    actual = solver_module._addition_data_derivative_grid(
        model, potential_adjoint
    )
    kernel = np.asarray(model.addition_kernel.unit_integrated_values)
    centre = np.asarray(model.addition_kernel.centre_index, dtype=float)
    start_offset = np.floor(-centre + 0.5).astype(int)
    base_shift = -(start_offset.astype(float) + centre)
    shifted = solver_module._shift_axis_numpy(
        kernel, float(base_shift[0]), axis=0
    )
    shifted = solver_module._shift_axis_numpy(
        shifted, float(base_shift[1]), axis=1
    )
    direct = correlate2d(potential_adjoint, shifted, mode="valid")
    anchors = np.argwhere(model.options.discovery_support.discovery_mask)
    starts = anchors + start_offset[None, :]
    expected = direct[starts[:, 0], starts[:, 1]] * float(
        model.addition_kernel.host_equivalent_integrated_scattering
    )
    np.testing.assert_allclose(
        actual[anchors[:, 0], anchors[:, 1]],
        expected,
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.all(np.isnan(actual[~model.options.discovery_support.discovery_mask]))


def test_proposal_spatial_queries_do_not_regress_to_dense_candidate_pairs():
    hard_source = inspect.getsource(
        solver_module._hard_core_directional_derivatives
    )
    admissible_source = inspect.getsource(
        solver_module._addition_admissible_mask
    )
    assert "[:, None, :]" not in hard_source + admissible_source
    assert "for index, position" not in hard_source
    assert "_weighted_hard_core_neighbor_sums" in hard_source
    assert "_minimum_separation_admissible_mask" in admissible_source


@pytest.mark.parametrize("ablation", ["edit_only", "level1_physical"])
def test_exact_scan_batched_proposal_scores_match_full_training(ablation):
    model = _model()
    truth = _with_extra(
        empty_atomistic_edit_state_1d(model), model, ADDITION_A, 0.8
    )
    prepared = _prepared(model, truth)
    state = empty_atomistic_edit_state_1d(model)
    full = atomistic_edit_proposal_scores_1d(
        prepared, state, 0.005, ablation=ablation
    )
    batched = atomistic_edit_proposal_scores_1d(
        prepared,
        state,
        0.005,
        ablation=ablation,
        training_scan_batch_size=1,
    )
    for name in (
        "addition_data_derivative_grid",
        "host_removal_data_derivative",
        "addition_violation_grid",
        "host_removal_violation",
        "paired_replacement_violation",
    ):
        np.testing.assert_allclose(
            getattr(batched, name),
            getattr(full, name),
            rtol=2e-10,
            atol=2e-10,
        )
    assert batched.best_kind == full.best_kind
    assert batched.best_index == full.best_index
    assert batched.best_violation == pytest.approx(
        full.best_violation, rel=2e-10, abs=2e-10
    )


def test_full_training_adjoint_scores_match_one_sided_edit_derivatives():
    model = _model()
    truth = _with_extra(
        empty_atomistic_edit_state_1d(model), model, ADDITION_A, 0.8
    )
    prepared = _prepared(model, truth)
    empty = empty_atomistic_edit_state_1d(model)
    scores = atomistic_edit_proposal_scores_1d(
        prepared, empty, 0.005, ablation="edit_only"
    )
    step = 2e-6
    addition = _with_extra(empty, model, ADDITION_A, step)
    finite_addition = (
        _count_component(prepared, addition)
        - _count_component(prepared, empty)
    ) / step
    assert scores.addition_data_derivative_grid[ADDITION_A] == pytest.approx(
        finite_addition, rel=2e-4, abs=2e-6
    )

    removal = _with_removal(empty, 0, step)
    finite_removal = (
        _count_component(prepared, removal)
        - _count_component(prepared, empty)
    ) / step
    assert scores.host_removal_data_derivative[0] == pytest.approx(
        finite_removal, rel=2e-4, abs=2e-6
    )
    np.testing.assert_array_equal(
        scores.training_indices, np.asarray(prepared.training_indices)
    )
    assert scores.score_units == (
        "objective_change_per_host_equivalent_edit_mass"
    )
    assert scores.certificate_scope == "full_training_proposal_grid_kkt:v1"


def test_level1_exposes_generic_paired_replacement_at_an_occupied_site():
    model = _model(penalty_path=(1e-4,))
    truth = _with_extra(
        _with_removal(empty_atomistic_edit_state_1d(model), 0, 1.0),
        model,
        tuple(HOST_CENTRES[0]),
        2.0,
    )
    prepared = _prepared(model, truth)
    scores = atomistic_edit_proposal_scores_1d(
        prepared,
        empty_atomistic_edit_state_1d(model),
        1e-4,
        ablation="level1_physical",
    )
    host_anchor = tuple(HOST_CENTRES[0])
    assert np.isneginf(scores.addition_violation_grid[host_anchor])
    assert tuple(scores.paired_replacement_anchor_indices[0]) == host_anchor
    assert np.isfinite(scores.paired_replacement_violation[0])
    assert scores.paired_replacement_scattering_equivalent[0] in {
        0.1,
        1.0,
        2.0,
    }


def test_frozen_path_uses_validation_only_and_debias_fixes_support_positions():
    model = _model(penalty_path=(1e6, 1e5))
    prepared = _prepared(model)
    result = run_prepared_atomistic_edit_reconstruction_1d(
        prepared,
        options=AtomisticEditSolverOptions1D(
            maximum_active_set_iterations=2,
            joint_refinement_updates=0,
            polish_updates=0,
            debias_updates=2,
            proposal_grid_kkt_tolerance=1e-8,
            active_projected_gradient_tolerance=1e3,
            validation_relative_tolerance=1.0,
            training_scan_batch_size=1,
            seed=13,
        ),
    )
    assert result.converged
    assert result.selected_path_index == 0
    assert result.selected_edit_penalty == 1e6
    assert len(result.path_points) == 2
    assert all(
        point.kkt.continuous_birth_kkt_evaluated is False
        for point in result.path_points
    )
    np.testing.assert_array_equal(
        result.penalized_state.extra_active, result.debiased_state.extra_active
    )
    np.testing.assert_array_equal(
        result.penalized_state.extra_anchor_indices,
        result.debiased_state.extra_anchor_indices,
    )
    np.testing.assert_array_equal(
        result.penalized_state.extra_position_offsets_A,
        result.debiased_state.extra_position_offsets_A,
    )
    assert result.metadata["selection_uses_validation_only"] is True
    assert result.metadata["audit_used_for_selection"] is False
    assert result.metadata["debias_rule"].startswith("support_and_position_fixed")
    assert result.metadata["training_gradient_accumulation"] == (
        "deterministic_exact_scan_batch_sum"
    )
    assert result.metadata["effective_training_scan_batch_size"] == 1
    assert result.debias_converged
    assert result.debias_projected_gradient_norm <= (
        result.debias_projected_gradient_tolerance
    )


def test_progress_callback_emits_truth_free_immutable_active_set_events():
    model = _model(penalty_path=(1e6,))
    prepared = _prepared(model)
    events = []
    result = run_prepared_atomistic_edit_reconstruction_1d(
        prepared,
        options=AtomisticEditSolverOptions1D(
            maximum_active_set_iterations=1,
            joint_refinement_updates=0,
            polish_updates=0,
            debias_updates=0,
            proposal_grid_kkt_tolerance=1e-8,
            active_projected_gradient_tolerance=1e3,
            debias_projected_gradient_tolerance=1e3,
        ),
        progress_callback=events.append,
    )

    phases = [event.phase for event in events]
    assert phases[0] == "initial"
    assert "refinement" in phases
    assert "polish" in phases
    assert "lambda_complete" in phases
    assert phases[-1] == "debias"
    assert all(event.path_index in {-1, 0} for event in events)
    assert all(not hasattr(event, "truth") for event in events)
    assert events[-1].detail == "support_and_position_fixed"
    assert events[-1].state is not result.debiased_state
    with pytest.raises(TypeError):
        events[0].state.extra_active[0] = True

    with pytest.raises(TypeError, match="progress_callback"):
        run_prepared_atomistic_edit_reconstruction_1d(
            prepared,
            options=AtomisticEditSolverOptions1D(
                maximum_active_set_iterations=1,
                joint_refinement_updates=0,
                polish_updates=0,
                debias_updates=0,
            ),
            progress_callback=object(),
        )


def test_exact_capacity_saturation_fails_closed():
    model = _model(
        penalty_path=(1e-8,), max_removals=2, max_extras=1
    )
    truth = _with_extra(
        empty_atomistic_edit_state_1d(model), model, ADDITION_B, 2.0
    )
    prepared = _prepared(model, truth)
    result = run_prepared_atomistic_edit_reconstruction_1d(
        prepared,
        options=AtomisticEditSolverOptions1D(
            ablation="edit_only",
            maximum_active_set_iterations=1,
            joint_refinement_updates=0,
            polish_updates=0,
            debias_updates=0,
            proposal_grid_kkt_tolerance=1e-12,
            active_projected_gradient_tolerance=1e9,
        ),
    )
    assert result.capacity_exhausted
    assert not result.converged
    assert result.stop_reason == "capacity_bound_fail_closed"
    assert result.path_points[0].capacity_status == (
        "saturated_resource_bound:extra_centres"
    )


def test_zero_pruning_is_an_active_set_change_before_kkt():
    model = _model(penalty_path=(1e6,), max_extras=1)
    initial = _with_extra(
        empty_atomistic_edit_state_1d(model), model, ADDITION_A, 1e-8
    )
    pruned, removal_count, extra_count = _prune_state(
        model, initial, 1e-4, ablation="level1_physical"
    )
    assert not np.any(pruned.extra_active)
    assert removal_count == 0
    assert extra_count == 1


def test_initial_state_cannot_seed_active_edits():
    model = _model(penalty_path=(1e6,))
    prepared = _prepared(model)
    active = _with_extra(
        empty_atomistic_edit_state_1d(model), model, ADDITION_A, 0.2
    )
    with pytest.raises(ValueError, match="empty edits"):
        run_prepared_atomistic_edit_reconstruction_1d(
            prepared,
            initial_state=active,
            options=AtomisticEditSolverOptions1D(
                joint_refinement_updates=0,
                polish_updates=0,
                debias_updates=0,
            ),
        )


def test_numerical_duplicate_additions_merge_within_declared_resolution():
    model = _model(penalty_path=(1e6,))
    state = empty_atomistic_edit_state_1d(model)
    anchors = np.asarray(state.extra_anchor_indices).copy()
    offsets = np.asarray(state.extra_position_offsets_A).copy()
    masses = np.asarray(state.extra_scattering_equivalents).copy()
    active = np.asarray(state.extra_active).copy()
    anchors[:] = ADDITION_A
    offsets[1, 0] = 1e-7
    masses[:] = (0.2, 0.3)
    active[:] = True
    state = replace(
        state,
        extra_anchor_indices=jnp.asarray(anchors),
        extra_position_offsets_A=jnp.asarray(offsets),
        extra_scattering_equivalents=jnp.asarray(masses),
        extra_active=jnp.asarray(active),
    )
    merged, merge_count, unresolved = _merge_duplicate_additions(
        model, state, 1e-6, ablation="edit_only"
    )
    assert not unresolved
    assert merge_count == 1
    assert np.count_nonzero(merged.extra_active) == 1
    assert np.sum(merged.extra_scattering_equivalents) == pytest.approx(0.5)


@pytest.mark.parametrize("ablation", ["edit_only", "level1_physical"])
def test_refinement_backtracks_instead_of_raising_at_continuous_support_boundary(
    monkeypatch,
    ablation,
):
    model = _model(penalty_path=(0.05,))
    prepared = _prepared(model)
    state = _with_extra(
        empty_atomistic_edit_state_1d(model),
        model,
        anchor=(4, 3),
        mass=0.2,
    )
    assert atomistic_edit_state_is_admissible_1d(model, state)

    def outward_value_and_gradient(parameters, structure, edit_penalty):
        del structure, edit_penalty
        gradients = {
            name: jnp.zeros_like(value) for name, value in parameters.items()
        }
        gradients["extra_position_offsets_A"] = gradients[
            "extra_position_offsets_A"
        ].at[0, 0].set(-1.0)
        return jnp.asarray(0.0), gradients

    monkeypatch.setattr(
        "wide_angle_propagation.ptychography_atomistic_edit_solver_1d."
        "_compiled_objective_value_and_gradient",
        lambda prepared, ablation: SimpleNamespace(
            function=outward_value_and_gradient
        ),
    )
    refined, accepted_updates = _refine_state(
        prepared,
        state,
        0.05,
        ablation=ablation,
        updates=1,
        learning_rate=0.1,
        gradient_clip=10.0,
        maximum_backtracking_steps=4,
    )
    assert accepted_updates == 0
    np.testing.assert_array_equal(
        refined.extra_position_offsets_A,
        state.extra_position_offsets_A,
    )


def test_deterministic_multistart_is_truth_free_and_reports_ambiguity():
    model = _model(penalty_path=(1e6,))
    prepared = _prepared(model)
    options = AtomisticEditSolverOptions1D(
        maximum_active_set_iterations=1,
        joint_refinement_updates=0,
        polish_updates=0,
        debias_updates=0,
        active_projected_gradient_tolerance=1e9,
        debias_projected_gradient_tolerance=1e9,
        validation_relative_tolerance=1.0,
        seed=19,
    )
    result = run_prepared_atomistic_edit_multistart_reconstruction_1d(
        prepared,
        number_of_starts=2,
        initial_host_control_std_A=0.01,
        options=options,
    )
    assert len(result.candidates) == 2
    assert result.initial_host_control_rms_A[0] == 0.0
    assert result.initial_host_control_rms_A[1] > 0.0
    assert result.start_seeds[0] != result.start_seeds[1]
    assert result.metadata["deterministic_seed_used"] is True
    assert result.metadata["truth_inputs_used"] is False
    assert result.metadata["selection_uses_validation_only"] is True
    assert result.metadata["audit_used_for_selection"] is False
    assert all(
        candidate.debiased_audit_count_deviance is None
        for candidate in result.candidates
    )
    assert result.selected_result.debiased_audit_count_deviance is not None

    changed_audit = run_prepared_atomistic_edit_multistart_reconstruction_1d(
        _prepared(model, audit_count_scale=1.2),
        number_of_starts=2,
        initial_host_control_std_A=0.01,
        options=options,
    )
    assert changed_audit.selected_start_index == result.selected_start_index
    assert changed_audit.start_seeds == result.start_seeds
    for first, second in zip(
        result.candidates, changed_audit.candidates, strict=True
    ):
        assert first.penalized_validation_count_deviance == pytest.approx(
            second.penalized_validation_count_deviance
        )
        np.testing.assert_allclose(
            first.debiased_state.host_displacement_controls,
            second.debiased_state.host_displacement_controls,
        )
    assert changed_audit.selected_result.debiased_audit_count_deviance != (
        result.selected_result.debiased_audit_count_deviance
    )
