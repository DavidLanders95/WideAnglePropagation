"""Integration tests for the compact glancing-ptychography workflow API."""

import matplotlib
import numpy as np
import pytest


matplotlib.use("Agg")
jax = pytest.importorskip("jax")
pytest.importorskip("abtem")
pytest.importorskip("optax")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.ptychography_workflow_1d import (  # noqa: E402
    ReconstructionOptions1D,
    SiliconGlancingConfig1D,
    build_silicon_glancing_experiment_1d,
    plot_experiment_overview_1d,
    plot_lattice_reconstruction_1d,
    plot_reconstruction_comparison_1d,
    reconstruct_experiment_1d,
    reconstruction_metrics_1d,
    save_experiment_results_1d,
    simulate_experiment_1d,
)


@pytest.fixture(scope="module")
def tiny_experiment():
    config = SiliconGlancingConfig1D(
        beam_waist_A=1.5,
        slab_depth_A=8.0,
        vacuum_above_A=8.0,
        vacuum_below_A=10.0,
        window_length_A=30.0,
        scan_start_A=10.0,
        scan_stop_A=20.0,
        n_scans=6,
        defect_center_s_A=15.0,
        defect_width_sites=2,
        validation_stride=3,
        atomic_template_cutoff_A=8.0,
        cutoff_check_A=10.0,
        maximum_displacement_A=0.25,
        displacement_control_spacing_A=10.0,
    )
    return build_silicon_glancing_experiment_1d(config)


def test_compact_workflow_builds_simulates_and_reconstructs(tiny_experiment, tmp_path):
    experiment = tiny_experiment
    assert set(experiment.truth_potentials) == {"vacancy", "vacancy_plus_strain"}
    assert experiment.summary["explicit vacancy sites"] == 2
    assert (
        experiment.summary["lattice parameters"] < experiment.summary["pixel unknowns"]
    )
    assert float(np.min(np.asarray(experiment.truth_potentials["vacancy"]))) >= 0.0
    support = np.asarray(experiment.reconstruction_mask)
    s_A = np.asarray(experiment.axial_coordinates)
    u_A = np.asarray(experiment.transverse_coordinates)
    support_s, support_u = np.where(support)
    nearest_landing = np.min(
        np.abs(s_A[support_s, None] - np.asarray(experiment.scan_coordinates)),
        axis=1,
    )
    landing_radius = (
        experiment.config.landing_radius_waists * experiment.config.beam_waist_A
    )
    assert experiment.config.update_region == "landing"
    assert np.all(nearest_landing <= landing_radius)
    assert np.all(u_A[support_u] >= -landing_radius)

    dataset = simulate_experiment_1d(experiment, "vacancy", batch_size=2)
    assert dataset.intensities.shape == (
        experiment.config.n_scans,
        len(experiment.transverse_coordinates),
    )
    assert dataset.template_cutoff_amplitude_nrmse < 1e-4

    results = reconstruct_experiment_1d(
        experiment,
        dataset,
        methods=("lattice_sites",),
        options=ReconstructionOptions1D(
            lattice_updates=2,
            minibatch_size=2,
            validation_interval_lattice=1,
            evaluation_batch_size=2,
            rematerialize=False,
        ),
    )
    assert tuple(results) == ("lattice sites",)
    metrics = reconstruction_metrics_1d(experiment, dataset, results)
    assert (
        metrics["lattice sites"]["specimen parameters"]
        == experiment.summary["lattice parameters"]
    )

    paths = save_experiment_results_1d(tmp_path, dataset, results)
    assert all(path.exists() for path in paths.values())

    overview = plot_experiment_overview_1d(experiment, dataset)
    comparison = plot_reconstruction_comparison_1d(experiment, dataset, results)
    lattice_figures = plot_lattice_reconstruction_1d(
        experiment, results["lattice sites"]
    )
    assert overview is not None
    assert len(comparison) == 2
    assert len(lattice_figures) == 2


def test_compact_workflow_rejects_unknown_case_and_method(tiny_experiment):
    with pytest.raises(ValueError, match="case"):
        simulate_experiment_1d(tiny_experiment, "unknown", batch_size=2)

    dataset = simulate_experiment_1d(tiny_experiment, "vacancy", batch_size=2)
    with pytest.raises(ValueError, match="unknown reconstruction methods"):
        reconstruct_experiment_1d(tiny_experiment, dataset, methods=("not-a-method",))
