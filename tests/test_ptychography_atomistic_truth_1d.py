"""Focused tests for the truth-isolated generic atomic edit renderer."""

from dataclasses import replace

import numpy as np
import pytest

from wide_angle_propagation.ptychography_atomic_validation_1d import (
    AtomicTemplateQuadratureOptions1D,
    render_si_atomic_template_1d,
)
from wide_angle_propagation.ptychography_atomistic_truth_1d import (
    DirectAtomicNumericalOptions1D,
    accumulate_weighted_atomic_potential_1d,
    render_direct_atomic_template_1d,
    sweep_adaptive_atomic_cubature_convergence_1d,
    sweep_atomic_quadrature_convergence_1d,
)


pytest.importorskip("abtem")


_SAMPLING_A = 0.2


def _options(
    *, pixel_order: int = 4, projection_order: int = 48
) -> AtomicTemplateQuadratureOptions1D:
    return AtomicTemplateQuadratureOptions1D(
        projection_width_A=5.431,
        cutoff_A=0.4,
        pixel_quadrature_order=pixel_order,
        projection_quadrature_order=projection_order,
    )


def _axis() -> np.ndarray:
    # A linspace-derived step exposes roundoff-only support-ceiling errors.
    return np.linspace(-1.2, 1.2, 13)


def test_generic_si_renderer_exactly_matches_existing_direct_quadrature():
    options = _options()
    arguments = {
        "sampling_s_A": _SAMPLING_A,
        "sampling_u_A": _SAMPLING_A,
        "options": options,
        "half_shape": (2, 2),
        "fractional_offset_A": (0.04, -0.03),
    }

    established = render_si_atomic_template_1d(**arguments)
    generic = render_direct_atomic_template_1d("Si", **arguments)

    assert np.array_equal(generic.values, established.values)
    assert generic.half_shape == established.half_shape
    assert generic.fractional_offset_A == established.fractional_offset_A
    assert generic.integrated_scattering == pytest.approx(
        np.sum(generic.values) * _SAMPLING_A**2,
        rel=2e-15,
    )
    assert np.sum(generic.unit_integrated_values) * _SAMPLING_A**2 == pytest.approx(
        1.0, rel=2e-15
    )


def test_tensor_si_compatibility_shares_established_precision_cache():
    import abtem

    original_precision = abtem.config.get("precision")
    options = AtomicTemplateQuadratureOptions1D(
        projection_width_A=1.123,
        cutoff_A=0.4,
        pixel_quadrature_order=2,
        projection_quadrature_order=10,
    )
    arguments = {
        "sampling_s_A": _SAMPLING_A,
        "sampling_u_A": _SAMPLING_A,
        "options": options,
        "half_shape": (2, 2),
        "fractional_offset_A": (0.031, -0.027),
    }
    try:
        abtem.config.set({"precision": "float32"})
        established = render_si_atomic_template_1d(**arguments)
        abtem.config.set({"precision": "float64"})
        generic = render_direct_atomic_template_1d("Si", **arguments)
    finally:
        abtem.config.set({"precision": original_precision})

    assert np.array_equal(generic.values, established.values)


def test_generic_renderer_supports_a_second_element_and_preserves_identity():
    arguments = {
        "sampling_s_A": _SAMPLING_A,
        "sampling_u_A": _SAMPLING_A,
        "options": _options(),
        "half_shape": (2, 2),
        "fractional_offset_A": (0.04, -0.03),
    }

    silicon = render_direct_atomic_template_1d("Si", **arguments)
    germanium = render_direct_atomic_template_1d("Ge", **arguments)

    assert germanium.element == "Ge"
    assert np.all(np.isfinite(germanium.values))
    assert np.all(germanium.values >= 0.0)
    assert not np.array_equal(germanium.values, silicon.values)
    assert germanium.integrated_scattering != pytest.approx(
        silicon.integrated_scattering
    )
    assert germanium.template_id != silicon.template_id


def test_weighted_accumulation_is_linear_and_permutation_invariant():
    axis = _axis()
    sites = np.asarray([[-0.36, 0.23], [0.43, -0.18]])
    elements = ("Si", "Ge")
    weights = np.asarray([0.65, 1.35])
    common = {
        "s_coordinates_A": axis,
        "u_coordinates_A": axis,
        "options": _options(pixel_order=2, projection_order=24),
    }

    combined = accumulate_weighted_atomic_potential_1d(
        sites, elements, weights, **common
    )
    first = accumulate_weighted_atomic_potential_1d(
        sites[:1], elements[:1], weights[:1], **common
    )
    second = accumulate_weighted_atomic_potential_1d(
        sites[1:], elements[1:], weights[1:], **common
    )
    permuted = accumulate_weighted_atomic_potential_1d(
        sites[::-1], elements[::-1], weights[::-1], **common
    )

    np.testing.assert_allclose(
        combined.values, first.values + second.values, rtol=2e-15, atol=1e-14
    )
    assert np.array_equal(combined.values, permuted.values)
    assert combined.grid_id == permuted.grid_id
    assert np.array_equal(combined.site_coordinates_A, permuted.site_coordinates_A)
    assert combined.elements == permuted.elements
    assert np.array_equal(combined.scattering_weights, permuted.scattering_weights)


def test_off_grid_centre_is_stamped_from_exact_fractional_quadrature():
    axis = _axis()
    ds = float((axis[-1] - axis[0]) / (len(axis) - 1))
    anchor = 6
    offset = (0.04, -0.03)
    site = np.asarray([[axis[anchor] + offset[0], axis[anchor] + offset[1]]])
    anchor_position = axis[0] + anchor * ds
    exact_offset = tuple(site[0] - anchor_position)
    options = _options(pixel_order=2, projection_order=24)

    grid = accumulate_weighted_atomic_potential_1d(
        site,
        ("Si",),
        (1.0,),
        s_coordinates_A=axis,
        u_coordinates_A=axis,
        options=options,
    )
    exact_template = render_direct_atomic_template_1d(
        "Si",
        sampling_s_A=ds,
        sampling_u_A=ds,
        options=options,
        half_shape=(2, 2),
        fractional_offset_A=exact_offset,
    )
    expected = np.zeros((len(axis), len(axis)))
    expected[anchor - 2 : anchor + 3, anchor - 2 : anchor + 3] = (
        exact_template.values
    )

    assert exact_template.half_shape == (2, 2)
    np.testing.assert_allclose(grid.values, expected, rtol=2e-15, atol=1e-14)
    assert grid.template_ids == (exact_template.template_id,)


def test_full_support_preserves_weighted_integrals_and_edges_fail_closed():
    axis = _axis()
    ds = float((axis[-1] - axis[0]) / (len(axis) - 1))
    options = _options(pixel_order=2, projection_order=24)
    sites = np.asarray([[-0.36, 0.23], [0.43, -0.18]])
    elements = ("Si", "Ge")
    weights = np.asarray([0.65, 1.35])

    grid = accumulate_weighted_atomic_potential_1d(
        sites,
        elements,
        weights,
        s_coordinates_A=axis,
        u_coordinates_A=axis,
        options=options,
    )
    expected_integral = 0.0
    for site, element, weight in zip(sites, elements, weights, strict=True):
        continuous_index = (site - axis[0]) / ds
        anchor = np.floor(continuous_index + 0.5).astype(int)
        anchor_position = axis[0] + anchor * ds
        offset = tuple(site - anchor_position)
        template = render_direct_atomic_template_1d(
            element,
            sampling_s_A=ds,
            sampling_u_A=ds,
            options=options,
            half_shape=(2, 2),
            fractional_offset_A=offset,
        )
        expected_integral += float(weight) * template.integrated_scattering

    assert np.sum(grid.values) * ds**2 == pytest.approx(
        expected_integral, rel=2e-15
    )
    assert grid.require_full_kernel_support
    assert grid.metadata["kernel_support_policy"] == "full_support_required"
    assert grid.metadata["clipped_kernel_count"] == 0

    edge_site = np.asarray([[axis[0], axis[len(axis) // 2]]])
    with pytest.raises(ValueError, match="clipped by the finite grid"):
        accumulate_weighted_atomic_potential_1d(
            edge_site,
            ("Si",),
            (1.0,),
            s_coordinates_A=axis,
            u_coordinates_A=axis,
            options=options,
        )

    explicitly_clipped = accumulate_weighted_atomic_potential_1d(
        edge_site,
        ("Si",),
        (1.0,),
        s_coordinates_A=axis,
        u_coordinates_A=axis,
        options=options,
        require_full_kernel_support=False,
    )
    assert not explicitly_clipped.require_full_kernel_support
    assert explicitly_clipped.metadata["kernel_support_policy"] == (
        "finite_grid_clipping_allowed_fail_closed"
    )
    assert explicitly_clipped.metadata["clipped_kernel_count"] == 1
    assert explicitly_clipped.trust_claim is False


@pytest.mark.parametrize("bad_weight", [-1e-12, -1.0, np.nan, np.inf])
def test_weighted_accumulation_rejects_non_physical_weights(bad_weight):
    with pytest.raises(ValueError, match="finite|non-negative"):
        accumulate_weighted_atomic_potential_1d(
            [[0.0, 0.0]],
            ("Si",),
            (bad_weight,),
            s_coordinates_A=_axis(),
            u_coordinates_A=_axis(),
            options=_options(pixel_order=2, projection_order=24),
        )


def test_geometry_validation_uses_strict_types_and_half_open_anchor_offsets():
    common = {
        "element": "Si",
        "sampling_s_A": _SAMPLING_A,
        "sampling_u_A": _SAMPLING_A,
        "options": _options(pixel_order=2, projection_order=24),
    }
    for invalid_shape in ((2.0, 2), (True, 2), (2, 2, 2)):
        with pytest.raises((TypeError, ValueError), match="half_shape"):
            render_direct_atomic_template_1d(
                **common, half_shape=invalid_shape
            )
    with pytest.raises(TypeError, match="sampling_s_A"):
        render_direct_atomic_template_1d(
            "Si",
            sampling_s_A=True,
            sampling_u_A=_SAMPLING_A,
            options=common["options"],
        )
    with pytest.raises(ValueError, match="fractional s offset"):
        render_direct_atomic_template_1d(
            **common, fractional_offset_A=(0.1, 0.0)
        )

    lower_half_pixel = render_direct_atomic_template_1d(
        **common, fractional_offset_A=(-0.1, 0.0)
    )
    assert lower_half_pixel.fractional_offset_A == (-0.1, 0.0)


def test_template_and_grid_evidence_is_immutable_authenticated_and_fail_closed():
    options = _options(pixel_order=2, projection_order=24)
    caller_metadata = {"purpose": "blind truth", "nested": {"labels": ["original"]}}
    template = render_direct_atomic_template_1d(
        "Si",
        sampling_s_A=_SAMPLING_A,
        sampling_u_A=_SAMPLING_A,
        options=options,
        metadata=caller_metadata,
    )
    grid = accumulate_weighted_atomic_potential_1d(
        [[0.04, -0.03]],
        ("Si",),
        (0.75,),
        s_coordinates_A=_axis(),
        u_coordinates_A=_axis(),
        options=options,
        metadata=caller_metadata,
    )
    caller_metadata["nested"]["labels"][0] = "mutated-after-render"

    for value in (
        template.values,
        template.unit_integrated_values,
        grid.values,
        grid.s_coordinates_A,
        grid.u_coordinates_A,
        grid.site_coordinates_A,
        grid.scattering_weights,
    ):
        assert not value.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        grid.values[0, 0] = 0.0
    with pytest.raises(TypeError):
        grid.metadata["purpose"] = "changed"
    assert template.metadata["nested"]["labels"] == ("original",)
    assert grid.metadata["nested"]["labels"] == ("original",)
    with pytest.raises(TypeError):
        template.metadata["nested"]["forged"] = True
    with pytest.raises(TypeError):
        grid.metadata["direct_atomic_numerical_options"]["integration_method"] = (
            "forged"
        )

    assert len(template.template_id) == 64
    assert len(grid.grid_id) == 64
    assert template.trust_claim is False
    assert grid.trust_claim is False
    assert "not experimental" in template.trust_reason
    assert "not experimental" in grid.trust_reason
    assert template.limitations == grid.limitations

    with pytest.raises(ValueError, match="fail closed"):
        replace(template, trust_claim=True)
    with pytest.raises(ValueError, match="template_id"):
        replace(template, template_id="0" * 64)
    with pytest.raises(ValueError, match="grid_id"):
        replace(grid, grid_id="0" * 64)


def test_quadrature_convergence_reports_pass_and_fail_without_claiming_trust():
    common = {
        "elements": ("Si", "Ge"),
        "sampling_s_A": _SAMPLING_A,
        "sampling_u_A": _SAMPLING_A,
        "base_options": _options(pixel_order=2, projection_order=24),
        "order_pairs": ((2, 24), (4, 48), (8, 96)),
        "fractional_offsets_A": ((0.04, -0.03),),
    }
    loose = sweep_atomic_quadrature_convergence_1d(
        **common,
        relative_l2_tolerance=1.0,
        relative_integral_tolerance=1.0,
    )
    strict = sweep_atomic_quadrature_convergence_1d(
        **common,
        relative_l2_tolerance=1e-14,
        relative_integral_tolerance=1e-14,
    )

    assert loose.passed
    assert not strict.passed
    assert loose.candidate_order_pair == (4, 48)
    assert loose.reference_order_pair == (8, 96)
    assert loose.maximum_relative_l2_by_order[-1] == 0.0
    assert loose.maximum_relative_integral_error_by_order[-1] == 0.0
    assert loose.maximum_relative_l2_by_order[-2] > 1e-14
    assert loose.maximum_relative_integral_error_by_order[-2] > 1e-14
    assert loose.report_id != strict.report_id
    assert len(loose.report_id) == 64
    assert loose.trust_claim is False
    assert "not experimental" in loose.trust_reason
    assert not loose.order_pairs.flags.writeable
    assert not loose.fractional_offsets_A.flags.writeable

    with pytest.raises(ValueError, match="fail closed"):
        replace(loose, trust_claim=True)
    with pytest.raises(ValueError, match="report_id"):
        replace(loose, report_id="0" * 64)
    with pytest.raises(ValueError, match="declared convergence criterion"):
        replace(loose, passed=False)


def test_factorized_cubature_is_stable_for_off_grid_si_and_ge_at_production_sampling():
    ds = 0.3394375
    du = 0.14678378
    options = AtomicTemplateQuadratureOptions1D(
        projection_width_A=5.431,
        cutoff_A=5.0,
        pixel_quadrature_order=2,
        projection_quadrature_order=24,
    )
    numerical = DirectAtomicNumericalOptions1D(
        integration_method="adaptive_factorized_cubature",
        adaptive_relative_tolerance=1e-7,
        adaptive_absolute_l2_tolerance=1e-9,
        adaptive_quadrature_rule="gk21",
        adaptive_max_intervals=4096,
        adaptive_max_evaluations=500_000,
        adaptive_error_safety_factor=4.0,
    )
    sweep_metadata = {
        "geometry": "maintained_sideview_production_sampling",
        "nested": {"labels": ["original"]},
    }
    report = sweep_adaptive_atomic_cubature_convergence_1d(
        ("Si", "Ge"),
        sampling_s_A=ds,
        sampling_u_A=du,
        base_options=options,
        base_numerical_options=numerical,
        tolerance_levels=((1e-5, 1e-7), (1e-7, 1e-9), (1e-9, 1e-11)),
        fractional_offsets_A=(
            (0.0, 0.0),
            (0.49 * ds, -0.49 * du),
            (-0.37 * ds, 0.23 * du),
        ),
        relative_l2_tolerance=1e-4,
        relative_integral_tolerance=1e-4,
        metadata=sweep_metadata,
    )
    sweep_metadata["nested"]["labels"][0] = "mutated-after-sweep"

    assert report.passed
    assert np.max(report.maximum_relative_l2_by_level[:-1]) <= 1e-4
    assert np.max(report.maximum_relative_integral_error_by_level[:-1]) <= 1e-4
    assert report.maximum_relative_l2_by_level[-1] == 0.0
    assert report.maximum_relative_integral_error_by_level[-1] == 0.0
    assert report.maximum_reported_template_l2_error_by_level[-1] > 0.0
    assert np.all(report.maximum_function_evaluations_by_level > 0)
    assert len(report.report_id) == 64
    assert report.trust_claim is False
    assert report.metadata["tensor_order_diagnostic_is_separate"] is True
    assert report.metadata["nested"]["labels"] == ("original",)
    with pytest.raises(TypeError):
        report.metadata["adaptive_runtime_provenance"]["scipy_version"] = "forged"
    with pytest.raises(ValueError, match="report_id"):
        replace(
            report,
            tolerance_levels=np.asarray(report.tolerance_levels)
            * np.asarray([0.9, 0.9]),
        )

    loose_failure = sweep_adaptive_atomic_cubature_convergence_1d(
        ("Si", "Ge"),
        sampling_s_A=ds,
        sampling_u_A=du,
        base_options=options,
        base_numerical_options=replace(
            numerical, adaptive_quadrature_rule="gk15"
        ),
        tolerance_levels=((1.0, 0.1), (1e-5, 1e-7), (1e-9, 1e-11)),
        fractional_offsets_A=((0.49 * ds, -0.49 * du),),
        relative_l2_tolerance=1e-4,
        relative_integral_tolerance=1e-4,
    )
    assert loose_failure.maximum_relative_l2_by_level[0] > 1e-4
    assert loose_failure.maximum_relative_l2_by_level[-2] <= 1e-4
    assert not loose_failure.passed


def test_tensor_default_is_explicit_and_rejected_by_near_core_reference():
    options = _options(pixel_order=8, projection_order=96)
    adaptive = DirectAtomicNumericalOptions1D(
        integration_method="adaptive_factorized_cubature",
        adaptive_relative_tolerance=1e-9,
        adaptive_absolute_l2_tolerance=1e-11,
    )
    arguments = {
        "sampling_s_A": _SAMPLING_A,
        "sampling_u_A": _SAMPLING_A,
        "options": options,
        "fractional_offset_A": (0.04, -0.03),
    }
    for element in ("Si", "Ge"):
        tensor = render_direct_atomic_template_1d(element, **arguments)
        reference = render_direct_atomic_template_1d(
            element, **arguments, numerical_options=adaptive
        )
        relative_l2 = np.linalg.norm(tensor.values - reference.values) / np.linalg.norm(
            reference.values
        )
        relative_integral = abs(
            tensor.integrated_scattering - reference.integrated_scattering
        ) / reference.integrated_scattering
        assert tensor.numerical_options.integration_method == "tensor_product"
        assert relative_l2 > 1e-4
        assert relative_integral > 1e-4
        evidence = reference.metadata["adaptive_factorized_cubature"]
        assert evidence["converged"]
        assert evidence["estimated_template_l2_error"] <= evidence[
            "template_l2_tolerance"
        ]
        assert tensor.template_id != reference.template_id
        with pytest.raises(TypeError):
            evidence["status"] = "forged"
        with pytest.raises(ValueError, match="template_id"):
            replace(
                reference,
                numerical_options=replace(
                    adaptive, adaptive_relative_tolerance=5e-10
                ),
            )

    with pytest.raises(ValueError, match="only meaningful.*tensor_product"):
        sweep_atomic_quadrature_convergence_1d(
            ("Si",),
            sampling_s_A=_SAMPLING_A,
            sampling_u_A=_SAMPLING_A,
            base_options=options,
            numerical_options=adaptive,
            order_pairs=((4, 48), (8, 96)),
        )


def test_factorized_cubature_fails_closed_on_nonconvergence():
    numerical = DirectAtomicNumericalOptions1D(
        integration_method="adaptive_factorized_cubature",
        adaptive_relative_tolerance=1e-14,
        adaptive_absolute_l2_tolerance=1e-16,
        adaptive_max_intervals=1,
        adaptive_max_evaluations=21,
        adaptive_error_safety_factor=4.0,
    )
    with pytest.raises(RuntimeError, match="did not converge|exceeded"):
        render_direct_atomic_template_1d(
            "Si",
            sampling_s_A=_SAMPLING_A,
            sampling_u_A=_SAMPLING_A,
            options=_options(pixel_order=2, projection_order=24),
            fractional_offset_A=(0.04, -0.03),
            numerical_options=numerical,
        )


def test_adaptive_error_safety_factor_cannot_deflate_error_evidence():
    with pytest.raises(ValueError, match="at least one"):
        DirectAtomicNumericalOptions1D(adaptive_error_safety_factor=0.5)
