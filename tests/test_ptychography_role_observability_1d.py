"""Focused TARGET/NUISANCE tests for prepared observability adapters."""

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax", reason="the ptychography extra is not installed")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.ptychography_1d import (  # noqa: E402
    LatticeSiteModel1D,
    prepare_lattice_site_reconstruction_1d,
    run_prepared_lattice_site_reconstruction_1d,
)
from wide_angle_propagation.ptychography_diagnostics_1d import (  # noqa: E402
    PoissonCountingModel1D,
)
from wide_angle_propagation.ptychography_observability_1d import (  # noqa: E402
    MatrixFreeObservabilityOptions1D,
    estimate_prepared_lattice_site_observability_matrix_free_1d,
    estimate_prepared_lattice_site_observability_stochastic_1d,
)
from wide_angle_propagation import (  # noqa: E402
    ptychography_stochastic_observability_1d as stochastic_observability,
)
from wide_angle_propagation.ptychography_support_contract_1d import (  # noqa: E402
    LatticeSiteRole1D,
    classify_lattice_site_support_1d,
)


ENERGY_EV = 30_000.0
POTENTIAL_SHAPE = (5, 8)
AXIAL_SAMPLING_A = 0.4
TRANSVERSE_SAMPLING_A = 0.3


def _role_aware_model():
    centers = np.asarray([[1, 2], [3, 5]], dtype=np.int64)
    starts = np.asarray([[0, 1], [2, 4]], dtype=np.int64)
    shapes = np.full((2, 2), 3, dtype=np.int64)
    coordinates = centers * np.asarray(
        [AXIAL_SAMPLING_A, TRANSVERSE_SAMPLING_A]
    )
    patch = np.asarray(
        [
            [0.0, 0.1, 0.0],
            [0.1, 0.9, 0.1],
            [0.0, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    patches = np.stack([patch, patch])
    reference = np.full(POTENTIAL_SHAPE, 0.02, dtype=np.float64)
    for start, site_patch in zip(starts, patches):
        start_s, start_u = start
        reference[
            start_s : start_s + site_patch.shape[0],
            start_u : start_u + site_patch.shape[1],
        ] += site_patch

    target = np.zeros(POTENTIAL_SHAPE, dtype=bool)
    target[tuple(centers[0])] = True
    contract = classify_lattice_site_support_1d(
        coordinates,
        centers,
        starts,
        shapes,
        target,
        np.ones(POTENTIAL_SHAPE, dtype=bool),
        known_fixed_site_mask=np.zeros(2, dtype=bool),
        excluded_probe_power=1e-5,
        atomic_template_cutoff_A=2.0,
        maximum_displacement_A=0.0,
        displacement_control_shape=(2, 2, 2),
        maximum_nuisance_sites=2,
        maximum_specimen_parameters=16,
        strict=True,
    )
    np.testing.assert_array_equal(
        contract.site_role_codes,
        [LatticeSiteRole1D.TARGET, LatticeSiteRole1D.NUISANCE],
    )
    return LatticeSiteModel1D(
        reference_potential=jnp.asarray(reference),
        site_coordinates=jnp.asarray(coordinates),
        site_patches=jnp.asarray(patches),
        patch_starts=jnp.asarray(starts),
        control_coordinates_s=jnp.asarray(
            [0.0, (POTENTIAL_SHAPE[0] - 1) * AXIAL_SAMPLING_A]
        ),
        control_coordinates_u=jnp.asarray(
            [0.0, (POTENTIAL_SHAPE[1] - 1) * TRANSVERSE_SAMPLING_A]
        ),
        axial_sampling=AXIAL_SAMPLING_A,
        transverse_sampling=TRANSVERSE_SAMPLING_A,
        maximum_displacement=0.0,
        metadata={"fixture": "role-aware-observability"},
        support_contract=contract,
    )


@pytest.fixture(scope="module")
def role_aware_prepared_result():
    model = _role_aware_model()
    transverse = (
        jnp.arange(POTENTIAL_SHAPE[1]) - 3.5
    ) * TRANSVERSE_SAMPLING_A
    probe = jnp.exp(-0.5 * (transverse / 0.6) ** 2) * jnp.exp(
        0.2j * transverse
    )
    prepared = prepare_lattice_site_reconstruction_1d(
        model,
        probe,
        jnp.asarray([0, 1]),
        4,
        jnp.ones(POTENTIAL_SHAPE[1], dtype=jnp.complex128),
        AXIAL_SAMPLING_A,
        ENERGY_EV,
        jnp.ones((2, POTENTIAL_SHAPE[1]), dtype=jnp.float64),
        validation_indices=[1],
        potential_max=4.0,
        minibatch_size=1,
        evaluation_batch_size=1,
        rematerialize=False,
        require_complete_material_scope=True,
    )
    reconstruction = run_prepared_lattice_site_reconstruction_1d(
        prepared,
        initial_vacancy_fractions=jnp.zeros(2),
        initial_displacement_controls=jnp.zeros((2, 2, 2)),
        learning_rate_start=0.01,
        learning_rate_end=0.01,
        updates=1,
        validation_interval=1,
        seed=3,
    )
    return prepared, reconstruction


def _operator_options():
    return MatrixFreeObservabilityOptions1D(
        scan_batch_size=1,
        maximum_iterations=16,
        relative_residual_tolerance=1e-6,
        stagnation_iterations=4,
        operator_check_vectors=1,
        maximum_selected_sites=2,
        exhaustive=False,
    )


def _counting_model():
    return PoissonCountingModel1D(electrons_per_pattern=10_000.0)


def test_exact_observability_reports_targets_and_profiles_nuisance_sites(
    role_aware_prepared_result,
):
    prepared, reconstruction = role_aware_prepared_result

    with pytest.raises(
        ValueError,
        match="nuisance sites cannot be selected as structural observability",
    ):
        estimate_prepared_lattice_site_observability_matrix_free_1d(
            prepared,
            reconstruction,
            _counting_model(),
            site_indices=[1],
            options=_operator_options(),
        )

    report = estimate_prepared_lattice_site_observability_matrix_free_1d(
        prepared,
        reconstruction,
        _counting_model(),
        site_indices=[0],
        options=_operator_options(),
    )

    assert report.metadata["selected_site_indices"] == [0]
    assert report.metadata["structural_reporting_site_indices"] == [0]
    assert report.metadata["profiled_nuisance_site_indices"] == [1]
    assert report.metadata["n_parameters"] == 6
    assert report.fit.metadata["n_parameters"] == 6
    assert report.fit.metadata["selected_site_indices"] == [0]

    material = report.metadata["represented_nuisance_coverage"][
        "material_support"
    ]
    assert material["target_sites"] == 1
    assert material["nuisance_sites_profiled"] == 1
    assert material["support_contract_id"] == reconstruction.support_contract_id

    assert report.metadata["material_scope_complete"] is True
    assert report.metadata["missing_nuisance_scopes"] == [
        "probe",
        "scan_geometry",
        "detector_calibration",
    ]
    assert "fixed_exterior_material" not in report.metadata[
        "missing_nuisance_scopes"
    ]


def test_stochastic_prepared_adapter_outputs_targets_only(
    role_aware_prepared_result,
):
    prepared, reconstruction = role_aware_prepared_result
    report = estimate_prepared_lattice_site_observability_stochastic_1d(
        prepared,
        reconstruction,
        _counting_model(),
        operator_options=_operator_options(),
        screening_options=stochastic_observability.StochasticFisherScreeningOptions1D(
            covariance_probe_count=2,
            null_probe_count=2,
            random_seed=7,
            maximum_iterations=8,
            relative_residual_tolerance=1e-6,
            stagnation_iterations=4,
            operator_check_vectors=1,
            maximum_pcg_solves=4,
            maximum_total_pcg_iterations=32,
            maximum_fisher_matvec_calls=256,
        ),
    )

    np.testing.assert_allclose(
        report.site_coordinates,
        np.asarray(prepared.model.site_coordinates)[[0]],
    )
    assert report.fit.screening.physical_marginal_variance.shape == (3,)
    assert len(report.fit.screening.covariance_blocks) == 1
    assert report.fit.screening.covariance_blocks[0].name == (
        "site_0_displacement"
    )
    assert report.fit.metadata["physical_output_scope"] == "TARGET_sites_only"
    assert report.fit.metadata["n_physical_outputs"] == 3
    assert report.fit.metadata["n_parameters"] == 6
    assert report.fit.metadata["nuisance_sites_profiled_in_fisher"] == 1
    assert report.metadata["all_site_count"] == 2
    assert report.metadata["reportable_target_site_count"] == 1
    assert report.metadata["profiled_nuisance_site_count"] == 1
    assert report.metadata["missing_nuisance_scopes"] == [
        "probe",
        "scan_geometry",
        "detector_calibration",
    ]
