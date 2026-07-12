"""Exact-reference and prior gates for the sparse atomistic-edit renderer."""

from dataclasses import replace

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from tests.atomistic_edit_test_helpers import (  # noqa: E402
    CompactAtomisticEditModelSpec1D,
    make_compact_atomistic_edit_model_1d,
)
from wide_angle_propagation.ptychography_atomistic_edit_1d import (  # noqa: E402
    AtomisticEditOptions1D,
    atomistic_edit_active_parameter_count_1d,
    atomistic_edit_addition_roles_1d,
    atomistic_edit_prior_components_1d,
    atomistic_edit_state_is_admissible_1d,
    atomistic_edit_state_is_within_discovery_support_1d,
    empty_atomistic_edit_state_1d,
    make_atomistic_edit_discovery_support_1d,
    make_atomistic_edit_kernel_1d,
    make_atomistic_edit_model_1d,
    render_atomistic_edit_potential_1d,
)
from wide_angle_propagation.ptychography_support_contract_1d import (  # noqa: E402
    LatticeSiteRole1D,
)


SHAPE = (13, 13)
DS_A = 1.0
DU_A = 1.0
HOST_EQUIVALENT_INTEGRAL = 5.0
HOST_CENTRES = np.asarray([[6, 6], [6, 9]], dtype=np.int32)
TARGET_ANCHOR = (3, 3)
NUISANCE_ANCHOR = (9, 8)


def _raw_kernel() -> np.ndarray:
    values = np.zeros((5, 5), dtype=np.float64)
    values[1:4, 1:4] = np.asarray(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
    )
    return values


def _kernel(
    values=None,
    *,
    axial_sampling_A=DS_A,
    transverse_sampling_A=DU_A,
    maximum_boundary_mass_fraction=1e-6,
):
    return make_atomistic_edit_kernel_1d(
        _raw_kernel() if values is None else values,
        axial_sampling_A=axial_sampling_A,
        transverse_sampling_A=transverse_sampling_A,
        host_equivalent_integrated_scattering=HOST_EQUIVALENT_INTEGRAL,
        centre_index=(2.0, 2.0),
        parameterization_id="compact-host-equivalent:v1",
        cutoff_A=2.0,
        projection_width_A=5.0,
        maximum_boundary_mass_fraction=maximum_boundary_mass_fraction,
    )


def _host_model():
    return make_compact_atomistic_edit_model_1d(
        CompactAtomisticEditModelSpec1D(
            shape=SHAPE,
            host_centres=HOST_CENTRES,
            target_discovery_centres=(TARGET_ANCHOR,),
            nuisance_discovery_centres=(NUISANCE_ANCHOR,),
            edit_penalty_path=(1.0, 0.5),
            max_host_removals=2,
            max_extra_centres=3,
            deformation_parameter_count=8,
            fixture_id="ae1-core-test",
            reference_background=0.05,
            maximum_displacement_A=0.5,
        )
    ).host_model


def _discovery(
    *,
    target_points=None,
    nuisance_points=None,
    surface_envelope_A=(0.0, 12.0),
):
    target = np.zeros(SHAPE, dtype=bool)
    nuisance = np.zeros(SHAPE, dtype=bool)
    if target_points is None:
        target_points = tuple(
            (row, column)
            for centre in (TARGET_ANCHOR, tuple(HOST_CENTRES[0]))
            for row in range(centre[0] - 1, centre[0] + 2)
            for column in range(centre[1] - 1, centre[1] + 2)
        )
    if nuisance_points is None:
        nuisance_points = tuple(
            (row, column)
            for centre in (NUISANCE_ANCHOR, tuple(HOST_CENTRES[1]))
            for row in range(centre[0] - 1, centre[0] + 2)
            for column in range(centre[1] - 1, centre[1] + 2)
        )
    for point in target_points:
        target[point] = True
    for point in nuisance_points:
        nuisance[point] = True
    return make_atomistic_edit_discovery_support_1d(
        np.arange(SHAPE[0], dtype=np.float64),
        np.arange(SHAPE[1], dtype=np.float64),
        target,
        nuisance,
        surface_envelope_A=surface_envelope_A,
        geometry_source_id="compact-valid-host-support:v1",
        excluded_probe_power=1e-6,
    )


def _model(*, discovery=None, enable_material_energy_envelope=False):
    addition_kernel = _kernel()
    host = _host_model()
    options = AtomisticEditOptions1D(
        max_host_removals=2,
        max_extra_centres=3,
        max_scattering_equivalent_per_centre=2.0,
        minimum_separation_A=2.0,
        expected_rms_host_strain=0.1,
        edit_penalty_path=(1.0, 0.5),
        discovery_support=_discovery() if discovery is None else discovery,
        enable_material_energy_envelope=enable_material_energy_envelope,
    )
    return make_atomistic_edit_model_1d(
        host,
        np.arange(SHAPE[0], dtype=np.float64),
        np.arange(SHAPE[1], dtype=np.float64),
        addition_kernel,
        options,
        deformation_parameter_count=6,
    )


def test_discovery_accepts_a_uniform_float32_grid_with_large_origin():
    axial = np.linspace(-1000.0, -900.0, 501, dtype=np.float32)
    transverse = np.linspace(-2.0, 2.0, 9, dtype=np.float32)
    target = np.zeros((axial.size, transverse.size), dtype=bool)
    nuisance = np.zeros_like(target)
    target[250, 4] = True
    nuisance[251, 4] = True
    support = make_atomistic_edit_discovery_support_1d(
        axial,
        transverse,
        target,
        nuisance,
        surface_envelope_A=(-2.0, 2.0),
        geometry_source_id="uniform-float32-large-origin:v1",
        excluded_probe_power=1e-6,
    )
    np.testing.assert_array_equal(support.axial_coordinates_A, axial)
    nonuniform = axial.copy()
    nonuniform[250] += np.float32(0.02)
    with pytest.raises(ValueError, match="uniformly increasing"):
        make_atomistic_edit_discovery_support_1d(
            nonuniform,
            transverse,
            target,
            nuisance,
            surface_envelope_A=(-2.0, 2.0),
            geometry_source_id="nonuniform-float32-large-origin:v1",
            excluded_probe_power=1e-6,
        )


@pytest.fixture(scope="module")
def compact_model():
    return _model()


def _one_extra_state(model, *, anchor=TARGET_ANCHOR, offset=(0.0, 0.0), mass=1.0):
    state = empty_atomistic_edit_state_1d(model)
    anchors = np.asarray(state.extra_anchor_indices).copy()
    anchors[0] = anchor
    offsets = np.zeros((model.options.max_extra_centres, 2), dtype=np.float64)
    offsets[0] = offset
    masses = np.zeros(model.options.max_extra_centres, dtype=np.float64)
    masses[0] = mass
    active = np.zeros(model.options.max_extra_centres, dtype=bool)
    active[0] = True
    return replace(
        state,
        extra_anchor_indices=jnp.asarray(anchors),
        extra_position_offsets_A=jnp.asarray(offsets),
        extra_scattering_equivalents=jnp.asarray(masses),
        extra_active=jnp.asarray(active),
    )


def _with_removal(state, *, site=0, fraction=1.0, active=True):
    indices = np.asarray(state.host_removal_indices).copy()
    fractions = np.asarray(state.host_removal_fractions).copy()
    active_mask = np.asarray(state.host_removal_active).copy()
    indices[0] = site
    fractions[0] = fraction
    active_mask[0] = active
    return replace(
        state,
        host_removal_indices=jnp.asarray(indices),
        host_removal_fractions=jnp.asarray(fractions),
        host_removal_active=jnp.asarray(active_mask),
    )


def _embedded_host_patch(model, site):
    result = np.zeros(SHAPE, dtype=np.float64)
    start_s, start_u = np.asarray(model.host_model.patch_starts)[site]
    patch = np.asarray(model.host_model.site_patches)[site]
    result[
        start_s : start_s + patch.shape[0],
        start_u : start_u + patch.shape[1],
    ] += patch
    return result


def test_kernel_is_unit_integrated_and_rejects_uncertified_boundary_mass():
    kernel = _kernel(axial_sampling_A=0.4, transverse_sampling_A=0.25)
    integral = np.sum(kernel.unit_integrated_values) * 0.4 * 0.25
    assert integral == pytest.approx(1.0, abs=1e-15)
    assert kernel.boundary_mass_fraction == 0.0

    values = _raw_kernel()
    values[0, 2] = 1.0
    with pytest.raises(ValueError, match="boundary mass"):
        _kernel(values, maximum_boundary_mass_fraction=0.0)


def test_discovery_anchors_with_truncated_kernel_footprints_are_rejected():
    edge_discovery = _discovery(target_points=((0, 0),), nuisance_points=())
    with pytest.raises(ValueError, match="boundary|footprint|padding"):
        _model(discovery=edge_discovery)


def test_zero_edit_is_exact_pristine_host_identity(compact_model):
    state = empty_atomistic_edit_state_1d(compact_model)
    rendered = np.asarray(render_atomistic_edit_potential_1d(compact_model, state))
    np.testing.assert_array_equal(
        rendered, np.asarray(compact_model.host_model.reference_potential)
    )


def test_unit_removal_subtracts_exactly_one_host_template(compact_model):
    state = _with_removal(
        empty_atomistic_edit_state_1d(compact_model), fraction=1.0
    )
    rendered = np.asarray(render_atomistic_edit_potential_1d(compact_model, state))
    reference = np.asarray(compact_model.host_model.reference_potential)
    np.testing.assert_allclose(
        rendered - reference,
        -_embedded_host_patch(compact_model, 0),
        rtol=0.0,
        atol=2e-16,
    )


@pytest.mark.parametrize("offset", [(0.0, 0.0), (0.23, -0.17)])
def test_unit_addition_has_one_host_equivalent_integrated_scattering(
    compact_model, offset
):
    state = _one_extra_state(compact_model, offset=offset, mass=1.0)
    rendered = np.asarray(render_atomistic_edit_potential_1d(compact_model, state))
    reference = np.asarray(compact_model.host_model.reference_potential)
    integrated_delta = np.sum(rendered - reference) * DS_A * DU_A
    assert integrated_delta == pytest.approx(HOST_EQUIVALENT_INTEGRAL, abs=2e-14)


def test_active_slot_permutations_do_not_change_render_or_prior(compact_model):
    state = empty_atomistic_edit_state_1d(compact_model)
    state = replace(
        state,
        host_removal_indices=jnp.asarray([0, 1]),
        host_removal_fractions=jnp.asarray([0.2, 0.7]),
        host_removal_active=jnp.asarray([True, True]),
        extra_anchor_indices=jnp.asarray(
            [TARGET_ANCHOR, NUISANCE_ANCHOR, TARGET_ANCHOR]
        ),
        extra_position_offsets_A=jnp.asarray(
            [[0.21, -0.12], [-0.24, 0.31], [0.0, 0.0]]
        ),
        extra_scattering_equivalents=jnp.asarray([0.4, 0.7, 0.0]),
        extra_active=jnp.asarray([True, True, False]),
    )
    permuted = replace(
        state,
        host_removal_indices=state.host_removal_indices[::-1],
        host_removal_fractions=state.host_removal_fractions[::-1],
        host_removal_active=state.host_removal_active[::-1],
        extra_anchor_indices=state.extra_anchor_indices[jnp.asarray([1, 0, 2])],
        extra_position_offsets_A=state.extra_position_offsets_A[
            jnp.asarray([1, 0, 2])
        ],
        extra_scattering_equivalents=state.extra_scattering_equivalents[
            jnp.asarray([1, 0, 2])
        ],
        extra_active=state.extra_active[jnp.asarray([1, 0, 2])],
    )
    np.testing.assert_allclose(
        render_atomistic_edit_potential_1d(compact_model, state),
        render_atomistic_edit_potential_1d(compact_model, permuted),
        rtol=0.0,
        atol=2e-15,
    )
    prior = atomistic_edit_prior_components_1d(compact_model, state, 0.5)
    permuted_prior = atomistic_edit_prior_components_1d(
        compact_model, permuted, 0.5
    )
    np.testing.assert_allclose(
        np.asarray(
            [
                prior.edit_mass,
                prior.weighted_edit_penalty,
                prior.elastic_penalty,
                prior.hard_core_penalty,
                prior.total_prior,
            ]
        ),
        np.asarray(
            [
                permuted_prior.edit_mass,
                permuted_prior.weighted_edit_penalty,
                permuted_prior.elastic_penalty,
                permuted_prior.hard_core_penalty,
                permuted_prior.total_prior,
            ]
        ),
        rtol=1e-15,
        atol=1e-15,
    )


def test_dormant_slots_have_exact_zero_value_and_jvp(compact_model):
    state = empty_atomistic_edit_state_1d(compact_model)
    reference = render_atomistic_edit_potential_1d(compact_model, state)

    def render_dormant(offsets, masses):
        return render_atomistic_edit_potential_1d(
            compact_model,
            replace(
                state,
                extra_position_offsets_A=offsets,
                extra_scattering_equivalents=masses,
            ),
        )

    value, tangent = jax.jvp(
        render_dormant,
        (state.extra_position_offsets_A, state.extra_scattering_equivalents),
        (
            jnp.full_like(state.extra_position_offsets_A, 0.37),
            jnp.full_like(state.extra_scattering_equivalents, 0.61),
        ),
    )
    np.testing.assert_array_equal(value, reference)
    np.testing.assert_array_equal(tangent, jnp.zeros_like(tangent))


def test_dormant_coincident_slots_have_finite_zero_hard_core_position_gradient(
    compact_model,
):
    state = empty_atomistic_edit_state_1d(compact_model)

    def hard_core(offsets, masses):
        candidate = replace(
            state,
            extra_position_offsets_A=offsets,
            extra_scattering_equivalents=masses,
        )
        return atomistic_edit_prior_components_1d(
            compact_model, candidate, 1.0
        ).hard_core_penalty

    offset_gradient, mass_gradient = jax.grad(
        hard_core, argnums=(0, 1)
    )(
        state.extra_position_offsets_A,
        state.extra_scattering_equivalents,
    )
    assert np.all(np.isfinite(np.asarray(offset_gradient)))
    assert np.all(np.isfinite(np.asarray(mass_gradient)))
    np.testing.assert_array_equal(
        offset_gradient, jnp.zeros_like(offset_gradient)
    )


def test_continuous_offset_and_amplitude_gradients_match_finite_differences(
    compact_model,
):
    state = _one_extra_state(
        compact_model, anchor=TARGET_ANCHOR, offset=(0.23, -0.17), mass=0.7
    )
    weights = jnp.asarray(
        np.random.default_rng(7).normal(size=SHAPE), dtype=jnp.float64
    )

    def objective(parameters):
        offsets = state.extra_position_offsets_A.at[0].set(parameters[:2])
        masses = state.extra_scattering_equivalents.at[0].set(parameters[2])
        rendered = render_atomistic_edit_potential_1d(
            compact_model,
            replace(
                state,
                extra_position_offsets_A=offsets,
                extra_scattering_equivalents=masses,
            ),
        )
        return jnp.sum(weights * rendered)

    parameters = np.asarray([0.23, -0.17, 0.7], dtype=np.float64)
    automatic = np.asarray(jax.grad(objective)(jnp.asarray(parameters)))
    step = 2e-6
    finite_difference = np.empty_like(parameters)
    for index in range(parameters.size):
        perturbation = np.zeros_like(parameters)
        perturbation[index] = step
        finite_difference[index] = (
            float(objective(jnp.asarray(parameters + perturbation)))
            - float(objective(jnp.asarray(parameters - perturbation)))
        ) / (2.0 * step)
    assert np.all(np.abs(automatic) > 1e-5)
    np.testing.assert_allclose(
        automatic, finite_difference, rtol=2e-9, atol=2e-9
    )


def test_addition_roles_keep_target_and_nuisance_separate(compact_model):
    state = empty_atomistic_edit_state_1d(compact_model)
    state = replace(
        state,
        extra_anchor_indices=jnp.asarray(
            [TARGET_ANCHOR, NUISANCE_ANCHOR, TARGET_ANCHOR]
        ),
        extra_scattering_equivalents=jnp.asarray([0.4, 0.6, 0.0]),
        extra_active=jnp.asarray([True, True, False]),
    )
    np.testing.assert_array_equal(
        atomistic_edit_addition_roles_1d(compact_model, state),
        np.asarray(
            [
                int(LatticeSiteRole1D.TARGET),
                int(LatticeSiteRole1D.NUISANCE),
                0,
            ],
            dtype=np.int8,
        ),
    )


def test_render_is_contained_by_bound_support_and_exterior_is_invariant(
    compact_model,
):
    state = _with_removal(
        _one_extra_state(
            compact_model, anchor=TARGET_ANCHOR, offset=(0.31, 0.24), mass=0.8
        ),
        fraction=0.35,
    )
    controls = np.zeros((2, 2, 2), dtype=np.float64)
    controls[1, :, 0] = 0.2
    controls[:, 1, 1] = -0.1
    state = replace(state, host_displacement_controls=jnp.asarray(controls))
    rendered = np.asarray(render_atomistic_edit_potential_1d(compact_model, state))
    reference = np.asarray(compact_model.host_model.reference_potential)
    outside = ~np.asarray(compact_model.support_contract.total_influence_mask)
    np.testing.assert_array_equal(rendered[outside], reference[outside])

    addition_state = _one_extra_state(
        compact_model, anchor=TARGET_ANCHOR, offset=(-0.2, 0.27), mass=0.6
    )
    addition_delta = np.asarray(
        render_atomistic_edit_potential_1d(compact_model, addition_state)
    ) - reference
    outside_addition = ~np.asarray(
        compact_model.support_contract.addition_influence_mask
    )
    np.testing.assert_array_equal(
        addition_delta[outside_addition],
        np.zeros(np.count_nonzero(outside_addition), dtype=addition_delta.dtype),
    )

    unsupported = _one_extra_state(compact_model, anchor=(1, 1), mass=1.0)
    with pytest.raises(ValueError, match="outside discovery support"):
        render_atomistic_edit_potential_1d(compact_model, unsupported)


@pytest.mark.parametrize(
    ("anchor", "outward_offset", "inward_offset"),
    [
        ((4, 3), (0.25, 0.0), (-0.25, 0.0)),
        ((9, 8), (0.25, 0.0), (-0.25, 0.0)),
    ],
)
def test_continuous_centres_cannot_leave_target_or_nuisance_discovery(
    anchor,
    outward_offset,
    inward_offset,
):
    discovery = _discovery(
        target_points=((3, 3), (4, 3)),
        nuisance_points=((8, 8), (9, 8)),
    )
    model = _model(discovery=discovery)
    outward = _one_extra_state(
        model,
        anchor=anchor,
        offset=outward_offset,
        mass=0.5,
    )
    assert not atomistic_edit_state_is_within_discovery_support_1d(
        model, outward
    )
    with pytest.raises(ValueError, match="leave TARGET/NUISANCE discovery support"):
        render_atomistic_edit_potential_1d(model, outward)

    inward = _one_extra_state(
        model,
        anchor=anchor,
        offset=inward_offset,
        mass=0.5,
    )
    assert atomistic_edit_state_is_within_discovery_support_1d(model, inward)
    assert atomistic_edit_state_is_admissible_1d(model, inward)
    rendered = np.asarray(render_atomistic_edit_potential_1d(model, inward))
    assert np.all(np.isfinite(rendered))


def test_continuous_centres_cannot_leave_surface_envelope():
    discovery = _discovery(
        target_points=((3, 9), (3, 10)),
        nuisance_points=(),
        surface_envelope_A=(0.0, 10.0),
    )
    model = _model(discovery=discovery)
    outward = _one_extra_state(
        model,
        anchor=(3, 10),
        offset=(0.0, 0.25),
        mass=0.5,
    )
    assert not atomistic_edit_state_is_within_discovery_support_1d(
        model, outward
    )
    with pytest.raises(ValueError, match="leave the declared surface envelope"):
        render_atomistic_edit_potential_1d(model, outward)

    inward = _one_extra_state(
        model,
        anchor=(3, 10),
        offset=(0.0, -0.25),
        mass=0.5,
    )
    assert atomistic_edit_state_is_within_discovery_support_1d(model, inward)
    assert atomistic_edit_state_is_admissible_1d(model, inward)
    rendered = np.asarray(render_atomistic_edit_potential_1d(model, inward))
    assert np.all(np.isfinite(rendered))


def test_active_parameter_count_is_pdef_plus_kminus_plus_three_kplus(
    compact_model,
):
    state = empty_atomistic_edit_state_1d(compact_model)
    assert atomistic_edit_active_parameter_count_1d(compact_model, state) == 6
    state = replace(
        state,
        host_removal_indices=jnp.asarray([0, 1]),
        host_removal_fractions=jnp.asarray([0.2, 0.8]),
        host_removal_active=jnp.asarray([True, True]),
        extra_anchor_indices=jnp.asarray(
            [TARGET_ANCHOR, NUISANCE_ANCHOR, TARGET_ANCHOR]
        ),
        extra_scattering_equivalents=jnp.asarray([0.3, 0.9, 0.0]),
        extra_active=jnp.asarray([True, True, False]),
    )
    expected = 6 + 2 + 3 * 2
    assert atomistic_edit_active_parameter_count_1d(compact_model, state) == expected


def test_full_removal_allows_a_colocated_substitution(compact_model):
    host_anchor = tuple(HOST_CENTRES[0])
    substitution = _with_removal(
        _one_extra_state(compact_model, anchor=host_anchor, mass=1.0),
        fraction=1.0,
    )
    assert atomistic_edit_state_is_admissible_1d(compact_model, substitution)
    prior = atomistic_edit_prior_components_1d(compact_model, substitution, 1.0)
    assert float(prior.hard_core_penalty) == 0.0
    np.testing.assert_allclose(
        render_atomistic_edit_potential_1d(compact_model, substitution),
        compact_model.host_model.reference_potential,
        rtol=0.0,
        atol=3e-16,
    )

    partial_removal = replace(
        substitution, host_removal_fractions=jnp.asarray([0.5, 0.0])
    )
    assert not atomistic_edit_state_is_admissible_1d(
        compact_model, partial_removal
    )


def test_hard_core_host_addition_term_is_weighted_by_host_occupancy(compact_model):
    collision = _one_extra_state(
        compact_model, anchor=tuple(HOST_CENTRES[0]), mass=1.0
    )

    def hard_core(removal_fraction):
        state = _with_removal(collision, fraction=removal_fraction)
        return float(
            atomistic_edit_prior_components_1d(
                compact_model, state, 1.0
            ).hard_core_penalty
        )

    occupied = hard_core(0.0)
    half_occupied = hard_core(0.5)
    removed = hard_core(1.0)
    assert np.isfinite(occupied) and occupied > 0.0
    assert half_occupied == pytest.approx(0.5 * occupied, rel=2e-15)
    assert removed == 0.0


def test_elastic_prior_has_translation_and_rotation_nulls_but_penalizes_strain(
    compact_model,
):
    state = empty_atomistic_edit_state_1d(compact_model)

    translation = np.zeros((2, 2, 2), dtype=np.float64)
    translation[...] = (0.2, -0.15)
    translation_prior = atomistic_edit_prior_components_1d(
        compact_model,
        replace(state, host_displacement_controls=jnp.asarray(translation)),
        1.0,
    )
    assert float(translation_prior.elastic_penalty) == 0.0

    omega = 0.02
    rotation = np.zeros((2, 2, 2), dtype=np.float64)
    control_s = np.asarray(compact_model.host_model.control_coordinates_s)
    control_u = np.asarray(compact_model.host_model.control_coordinates_u)
    rotation[..., 0] = -omega * control_u[None, :]
    rotation[..., 1] = omega * control_s[:, None]
    rotation_prior = atomistic_edit_prior_components_1d(
        compact_model,
        replace(state, host_displacement_controls=jnp.asarray(rotation)),
        1.0,
    )
    assert abs(float(rotation_prior.elastic_penalty)) < 1e-28

    strain = np.zeros((2, 2, 2), dtype=np.float64)
    strain[1, :, 0] = 0.24
    strain_prior = atomistic_edit_prior_components_1d(
        compact_model,
        replace(state, host_displacement_controls=jnp.asarray(strain)),
        1.0,
    )
    assert float(strain_prior.elastic_penalty) > 0.0


def test_material_energy_envelope_fails_closed_before_validation():
    with pytest.raises(NotImplementedError, match="blocked"):
        _model(enable_material_energy_envelope=True)
