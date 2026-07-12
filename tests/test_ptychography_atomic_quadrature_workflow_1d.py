"""Workflow binding for the explicit production-grid quadrature sweep."""

from types import SimpleNamespace

import numpy as np
import pytest

from wide_angle_propagation.ptychography_atomic_validation_1d import (
    AtomicTemplateQuadratureOptions1D,
)
from wide_angle_propagation.ptychography_atomistic_truth_1d import (
    DirectAtomicNumericalOptions1D,
)
import wide_angle_propagation.ptychography_workflow_1d as workflow
from wide_angle_propagation.ptychography_workflow_1d import (
    AtomicTemplateCertification1D,
    SiliconGlancingExperiment1D,
    sweep_experiment_adaptive_atomic_cubature_convergence_1d,
    sweep_experiment_atomic_quadrature_convergence_1d,
)


_SAMPLING_S_A = 0.1875
_SAMPLING_U_A = 0.3125
_CUTOFF_A = 0.625
_PROJECTION_WIDTH_A = 5.431
_SUPPORT_ID = "a" * 64
_TEMPLATE_ID = "b" * 64


def _minimal_experiment() -> SiliconGlancingExperiment1D:
    """Create an isinstance-correct shell containing only helper inputs."""
    experiment = object.__new__(SiliconGlancingExperiment1D)
    options = AtomicTemplateQuadratureOptions1D(
        projection_width_A=_PROJECTION_WIDTH_A,
        cutoff_A=_CUTOFF_A,
        pixel_quadrature_order=2,
        projection_quadrature_order=24,
        maximum_quadrature_evaluations=123_456,
    )
    template = SimpleNamespace(
        sampling_s_A=_SAMPLING_S_A,
        sampling_u_A=_SAMPLING_U_A,
        options=options,
        template_sha256=_TEMPLATE_ID,
        trust_claim=False,
        trust_reason="fail-closed independent-atom numerical diagnostic",
        limitations=(
            "independent neutral-atom parameterization",
            "not experimental or first-principles validation",
        ),
    )
    support = SimpleNamespace(
        contract_id=_SUPPORT_ID,
        strict_requirements_satisfied=True,
    )
    certification = AtomicTemplateCertification1D(
        cutoff_A=_CUTOFF_A,
        reference_cutoff_A=1.25,
        relative_tail_l2=2e-7,
        tolerance=1e-6,
        candidate_errors={f"{_CUTOFF_A:g}": 2e-7},
    )
    object.__setattr__(experiment, "support_contract", support)
    object.__setattr__(experiment, "independent_kirkland_template", template)
    object.__setattr__(experiment, "template_certification", certification)
    object.__setattr__(experiment, "axial_sampling", _SAMPLING_S_A)
    object.__setattr__(experiment, "transverse_sampling", _SAMPLING_U_A)
    return experiment


def _accept_fake_support(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, bool]]:
    calls: list[tuple[object, bool]] = []

    def validate(contract, *, strict=True):
        calls.append((contract, strict))
        return contract

    monkeypatch.setattr(
        workflow,
        "validate_lattice_site_support_contract_1d",
        validate,
    )
    return calls


def test_explicit_sweep_forwards_exact_production_geometry_and_provenance(
    monkeypatch,
):
    experiment = _minimal_experiment()
    support_calls = _accept_fake_support(monkeypatch)
    forwarded = {}
    sentinel = object()

    def sweep(elements, **kwargs):
        forwarded["elements"] = elements
        forwarded.update(kwargs)
        return sentinel

    monkeypatch.setattr(workflow, "sweep_atomic_quadrature_convergence_1d", sweep)
    orders = ((2, 24), (4, 48), (8, 96))
    offsets = ((0.0, 0.0), (0.04, -0.05))
    result = sweep_experiment_atomic_quadrature_convergence_1d(
        experiment,
        order_pairs=orders,
        fractional_offsets_A=offsets,
        relative_l2_tolerance=3e-5,
        relative_integral_tolerance=4e-6,
    )

    assert result is sentinel
    assert support_calls == [(experiment.support_contract, True)]
    assert forwarded["elements"] == ("Si",)
    assert forwarded["sampling_s_A"] == _SAMPLING_S_A
    assert forwarded["sampling_u_A"] == _SAMPLING_U_A
    assert forwarded["base_options"] is experiment.independent_kirkland_template.options
    assert forwarded["base_options"].projection_width_A == _PROJECTION_WIDTH_A
    assert forwarded["base_options"].cutoff_A == _CUTOFF_A
    assert forwarded["order_pairs"] is orders
    assert forwarded["fractional_offsets_A"] is offsets
    assert forwarded["relative_l2_tolerance"] == 3e-5
    assert forwarded["relative_integral_tolerance"] == 4e-6
    metadata = forwarded["metadata"]
    assert metadata["experiment_support_contract_id"] == _SUPPORT_ID
    assert metadata["experiment_template_sha256"] == _TEMPLATE_ID
    assert metadata["experiment_template_options_sha256"] == (
        experiment.independent_kirkland_template.options.options_sha256
    )
    assert metadata["experiment_template_limitations"] == list(
        experiment.independent_kirkland_template.limitations
    )
    assert metadata["production_sampling_s_A"] == _SAMPLING_S_A
    assert metadata["production_sampling_u_A"] == _SAMPLING_U_A
    assert metadata["production_projection_width_A"] == _PROJECTION_WIDTH_A
    assert metadata["certified_cutoff_A"] == _CUTOFF_A
    assert metadata["validation_scope"].endswith("convergence_only")
    assert metadata["experiment_template_trust_claim"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("sampling", "sampling does not exactly match"),
        ("cutoff", "cutoff does not exactly match"),
        ("certification", "cutoff certification is inconsistent"),
        ("trust", "must fail closed"),
        ("limitations", "retain its limitations"),
    ),
)
def test_explicit_sweep_fails_closed_on_inconsistent_experiment(
    monkeypatch,
    mutation,
    message,
):
    experiment = _minimal_experiment()
    _accept_fake_support(monkeypatch)
    if mutation == "sampling":
        experiment.independent_kirkland_template.sampling_s_A = np.nextafter(
            _SAMPLING_S_A, np.inf
        )
    elif mutation == "cutoff":
        object.__setattr__(
            experiment,
            "template_certification",
            AtomicTemplateCertification1D(
                cutoff_A=0.5,
                reference_cutoff_A=1.25,
                relative_tail_l2=2e-7,
                tolerance=1e-6,
            ),
        )
    elif mutation == "certification":
        object.__setattr__(
            experiment,
            "template_certification",
            AtomicTemplateCertification1D(
                cutoff_A=_CUTOFF_A,
                reference_cutoff_A=1.25,
                relative_tail_l2=2e-5,
                tolerance=1e-6,
            ),
        )
    elif mutation == "trust":
        experiment.independent_kirkland_template.trust_claim = True
    else:
        experiment.independent_kirkland_template.limitations = ()

    with pytest.raises(ValueError, match=message):
        sweep_experiment_atomic_quadrature_convergence_1d(
            experiment,
            order_pairs=((2, 24), (4, 48)),
            fractional_offsets_A=((0.0, 0.0),),
            relative_l2_tolerance=1e-4,
            relative_integral_tolerance=1e-4,
        )


def test_explicit_sweep_leaves_unsafe_order_validation_to_truth_layer(monkeypatch):
    experiment = _minimal_experiment()
    _accept_fake_support(monkeypatch)

    with pytest.raises(ValueError, match="componentwise non-decreasing"):
        sweep_experiment_atomic_quadrature_convergence_1d(
            experiment,
            order_pairs=((4, 48), (2, 96)),
            fractional_offsets_A=((0.0, 0.0),),
            relative_l2_tolerance=1e-4,
            relative_integral_tolerance=1e-4,
        )


def test_adaptive_sweep_forwards_production_geometry_and_default_budgets(
    monkeypatch,
):
    experiment = _minimal_experiment()
    support_calls = _accept_fake_support(monkeypatch)
    forwarded = {}
    numerical = DirectAtomicNumericalOptions1D(
        integration_method="adaptive_factorized_cubature",
        adaptive_max_evaluations=100_000,
    )

    def sweep(elements, **kwargs):
        forwarded["elements"] = elements
        forwarded.update(kwargs)
        return SimpleNamespace(
            trust_claim=False,
            limitations=("independent-atom numerical diagnostic",),
            metadata=dict(kwargs["metadata"]),
            report_id="c" * 64,
        )

    monkeypatch.setattr(
        workflow,
        "sweep_adaptive_atomic_cubature_convergence_1d",
        sweep,
    )
    levels = ((1e-5, 1e-7), (1e-7, 1e-9))
    offsets = ((0.0, 0.0), (0.04, -0.05))
    result = sweep_experiment_adaptive_atomic_cubature_convergence_1d(
        experiment,
        tolerance_levels=levels,
        fractional_offsets_A=offsets,
        elements=("Si", "Ge"),
        base_numerical_options=numerical,
    )

    assert result.report_id == "c" * 64
    assert support_calls == [(experiment.support_contract, True)]
    assert forwarded["elements"] == ("Si", "Ge")
    assert forwarded["sampling_s_A"] == _SAMPLING_S_A
    assert forwarded["sampling_u_A"] == _SAMPLING_U_A
    assert forwarded["base_options"] is (
        experiment.independent_kirkland_template.options
    )
    assert forwarded["tolerance_levels"] is levels
    assert forwarded["fractional_offsets_A"] is offsets
    assert forwarded["base_numerical_options"] is numerical
    assert forwarded["relative_l2_tolerance"] == 1e-4
    assert forwarded["relative_integral_tolerance"] == 1e-4
    metadata = forwarded["metadata"]
    assert metadata["experiment_support_contract_id"] == _SUPPORT_ID
    assert metadata["experiment_template_sha256"] == _TEMPLATE_ID
    assert metadata["experiment_template_options_sha256"] == (
        experiment.independent_kirkland_template.options.options_sha256
    )
    assert metadata["production_sampling_s_A"] == _SAMPLING_S_A
    assert metadata["production_sampling_u_A"] == _SAMPLING_U_A
    assert metadata["production_projection_width_A"] == _PROJECTION_WIDTH_A
    assert metadata["certified_cutoff_A"] == _CUTOFF_A
    assert metadata["experiment_template_trust_claim"] is False
    assert metadata["validation_scope"] == (
        "numerical_adaptive_atomic_cubature_convergence_only"
    )


@pytest.mark.parametrize("invalid", ("method", "tolerance"))
def test_adaptive_sweep_rejects_invalid_method_or_tolerance(
    monkeypatch,
    invalid,
):
    experiment = _minimal_experiment()
    _accept_fake_support(monkeypatch)
    numerical = DirectAtomicNumericalOptions1D(
        integration_method=(
            "tensor_product"
            if invalid == "method"
            else "adaptive_factorized_cubature"
        ),
        adaptive_max_evaluations=100_000,
    )
    levels = (
        ((1e-5, 1e-7), (1e-7, 1e-9))
        if invalid == "method"
        else ((1e-7, 1e-9), (1e-5, 1e-10))
    )
    message = (
        "must select adaptive_factorized_cubature"
        if invalid == "method"
        else "tighten componentwise"
    )
    with pytest.raises(ValueError, match=message):
        sweep_experiment_adaptive_atomic_cubature_convergence_1d(
            experiment,
            tolerance_levels=levels,
            fractional_offsets_A=((0.0, 0.0),),
            base_numerical_options=numerical,
        )


def test_adaptive_report_id_binds_experiment_provenance(monkeypatch):
    experiment = _minimal_experiment()
    _accept_fake_support(monkeypatch)
    numerical = DirectAtomicNumericalOptions1D(
        integration_method="adaptive_factorized_cubature",
        adaptive_relative_tolerance=1e-6,
        adaptive_absolute_l2_tolerance=1e-8,
        adaptive_max_intervals=512,
        adaptive_max_evaluations=100_000,
    )
    arguments = {
        "tolerance_levels": ((1e-4, 1e-6), (1e-6, 1e-8)),
        "fractional_offsets_A": ((0.04, -0.05),),
        "elements": ("Si",),
        "base_numerical_options": numerical,
    }
    first = sweep_experiment_adaptive_atomic_cubature_convergence_1d(
        experiment, **arguments
    )
    assert first.passed
    assert np.max(first.maximum_relative_l2_by_level[:-1]) <= 1e-4
    assert np.max(first.maximum_relative_integral_error_by_level[:-1]) <= 1e-4
    assert len(first.report_id) == 64
    assert first.trust_claim is False
    assert first.metadata["experiment_support_contract_id"] == _SUPPORT_ID
    assert first.metadata["experiment_template_sha256"] == _TEMPLATE_ID
    assert first.metadata["experiment_template_options_sha256"] == (
        experiment.independent_kirkland_template.options.options_sha256
    )

    experiment.support_contract.contract_id = "d" * 64
    experiment.independent_kirkland_template.template_sha256 = "e" * 64
    second = sweep_experiment_adaptive_atomic_cubature_convergence_1d(
        experiment, **arguments
    )
    assert second.metadata["experiment_support_contract_id"] == "d" * 64
    assert second.metadata["experiment_template_sha256"] == "e" * 64
    assert second.report_id != first.report_id


@pytest.mark.parametrize("mutation", ("trust", "provenance"))
def test_adaptive_wrapper_rejects_promoted_or_unbound_reports(
    monkeypatch,
    mutation,
):
    experiment = _minimal_experiment()
    _accept_fake_support(monkeypatch)

    def sweep(_elements, **kwargs):
        metadata = dict(kwargs["metadata"])
        if mutation == "provenance":
            metadata.pop("experiment_support_contract_id")
        return SimpleNamespace(
            trust_claim=mutation == "trust",
            limitations=("independent-atom numerical diagnostic",),
            metadata=metadata,
        )

    monkeypatch.setattr(
        workflow,
        "sweep_adaptive_atomic_cubature_convergence_1d",
        sweep,
    )
    message = "remain fail closed" if mutation == "trust" else "lost experiment"
    with pytest.raises(RuntimeError, match=message):
        sweep_experiment_adaptive_atomic_cubature_convergence_1d(
            experiment,
            tolerance_levels=((1e-5, 1e-7), (1e-7, 1e-9)),
            fractional_offsets_A=((0.0, 0.0),),
        )
