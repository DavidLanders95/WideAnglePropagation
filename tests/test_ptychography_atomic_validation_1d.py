"""Focused tests for the independent silicon atomic-template validator."""

from dataclasses import replace

import numpy as np
import pytest

from wide_angle_propagation.ptychography_atomic_validation_1d import (
    AtomicTemplateQuadratureOptions1D,
    accumulate_si_atomic_potential_1d,
    atomic_template_cache_info_1d,
    clear_atomic_template_cache_1d,
    compare_si_atomic_template_1d,
    render_si_atomic_template_1d,
)


pytest.importorskip("abtem")


def _options(
    *,
    pixel_order: int = 2,
    projection_order: int = 24,
) -> AtomicTemplateQuadratureOptions1D:
    return AtomicTemplateQuadratureOptions1D(
        projection_width_A=5.431,
        cutoff_A=0.4,
        pixel_quadrature_order=pixel_order,
        projection_quadrature_order=projection_order,
    )


def _render(
    *,
    pixel_order: int = 2,
    projection_order: int = 24,
    offset: tuple[float, float] = (0.0, 0.0),
):
    return render_si_atomic_template_1d(
        sampling_s_A=0.2,
        sampling_u_A=0.2,
        options=_options(
            pixel_order=pixel_order,
            projection_order=projection_order,
        ),
        fractional_offset_A=offset,
    )


def test_tensor_quadrature_converges_toward_higher_order_reference():
    low = _render(pixel_order=2, projection_order=24).values
    medium = _render(pixel_order=4, projection_order=96).values
    reference = _render(pixel_order=8, projection_order=192).values

    low_error = np.linalg.norm(low - reference) / np.linalg.norm(reference)
    medium_error = np.linalg.norm(medium - reference) / np.linalg.norm(reference)

    assert medium_error < 0.15 * low_error
    assert medium_error < 0.03


def test_exact_subpixel_render_changes_without_interpolation_and_obeys_symmetry():
    positive = _render(
        pixel_order=4,
        projection_order=48,
        offset=(0.04, -0.03),
    )
    negative = _render(
        pixel_order=4,
        projection_order=48,
        offset=(-0.04, 0.03),
    )
    centred = _render(pixel_order=4, projection_order=48)

    assert not np.array_equal(positive.values, centred.values)
    assert positive.template_sha256 != centred.template_sha256
    np.testing.assert_allclose(
        positive.values,
        negative.values[::-1, ::-1],
        rtol=2e-15,
        atol=1e-14,
    )


def test_fractional_offset_cache_and_digest_are_deterministic_and_immutable():
    clear_atomic_template_cache_1d()
    first = _render(offset=(0.04, -0.03))
    after_first = atomic_template_cache_info_1d()
    second = _render(offset=(0.04, -0.03))
    after_second = atomic_template_cache_info_1d()

    assert after_first.misses == 1
    assert after_second.hits == 1
    assert first.template_sha256 == second.template_sha256
    assert first.options.options_sha256 == second.options.options_sha256
    assert np.array_equal(first.values, second.values)
    assert not first.values.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first.values[0, 0] = 0.0

    clear_atomic_template_cache_1d()
    rebuilt = _render(offset=(0.04, -0.03))
    assert rebuilt.template_sha256 == first.template_sha256


def test_renderer_does_not_call_abtem_builder_projection_integrals_or_image_shift(
    monkeypatch,
):
    import abtem
    import abtem.integrals
    import scipy.ndimage

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("a prohibited renderer was called")

    monkeypatch.setattr(abtem, "Potential", forbidden)
    monkeypatch.setattr(abtem.integrals, "QuadratureProjectionIntegrals", forbidden)
    monkeypatch.setattr(scipy.ndimage, "shift", forbidden)
    clear_atomic_template_cache_1d()

    rendered = _render(offset=(0.03, 0.02))

    assert np.all(np.isfinite(rendered.values))
    assert np.all(rendered.values >= 0.0)


def test_finite_grid_accumulates_only_explicit_sites_with_exact_local_patch():
    axis = np.linspace(-1.0, 1.0, 11)
    offset = (0.04, -0.03)
    site = np.asarray([[axis[5] + offset[0], axis[5] + offset[1]]])
    grid = accumulate_si_atomic_potential_1d(
        np.concatenate((site, site), axis=0),
        s_coordinates_A=axis,
        u_coordinates_A=axis,
        options=_options(pixel_order=4, projection_order=48),
    )
    template = render_si_atomic_template_1d(
        sampling_s_A=0.2,
        sampling_u_A=0.2,
        options=_options(pixel_order=4, projection_order=48),
        fractional_offset_A=offset,
    )

    expected = np.zeros((11, 11))
    expected[3:8, 3:8] = 2.0 * template.values
    np.testing.assert_allclose(grid.values, expected, rtol=2e-15, atol=1e-14)
    assert grid.site_coordinates_A.shape == (2, 2)
    assert grid.fractional_offsets_A.shape == (1, 2)
    assert grid.template_sha256 == (template.template_sha256,)
    assert not grid.trust_claim
    assert not grid.values.flags.writeable


def test_finite_grid_has_no_implicit_periodic_images_or_hidden_atoms():
    axis = np.linspace(-1.0, 1.0, 11)
    empty = accumulate_si_atomic_potential_1d(
        np.empty((0, 2)),
        s_coordinates_A=axis,
        u_coordinates_A=axis,
        options=_options(),
    )
    outside = accumulate_si_atomic_potential_1d(
        np.asarray([[100.0, 100.0]]),
        s_coordinates_A=axis,
        u_coordinates_A=axis,
        options=_options(),
    )

    assert np.count_nonzero(empty.values) == 0
    assert np.count_nonzero(outside.values) == 0
    assert empty.template_sha256 == ()
    assert empty.grid_sha256 != outside.grid_sha256


def test_comparison_reports_raw_shape_peak_and_integral_mismatch():
    candidate = _render(pixel_order=4, projection_order=48)
    comparison = compare_si_atomic_template_1d(
        candidate,
        2.0 * candidate.values,
        reference_provenance={"generator": "controlled two-times scaling"},
    )

    assert comparison.raw_relative_l2 == pytest.approx(0.5)
    assert comparison.scale_adjusted_shape_relative_l2 < 1e-15
    assert comparison.optimal_candidate_scale == pytest.approx(2.0)
    assert comparison.peak_ratio == pytest.approx(0.5)
    assert comparison.integral_ratio == pytest.approx(0.5)
    assert not comparison.trust_claim
    assert "fail-closed" in comparison.trust_reason
    assert len(comparison.reference_template_sha256) == 64
    assert len(comparison.comparison_sha256) == 64


@pytest.mark.parametrize(
    ("keyword", "value", "match"),
    [
        ("projection_width_A", 0.0, "finite and positive"),
        ("cutoff_A", np.inf, "finite and positive"),
        ("pixel_quadrature_order", 3, "even integer"),
        ("projection_quadrature_order", True, "must be an integer"),
        ("maximum_quadrature_evaluations", 0, "must be positive"),
    ],
)
def test_options_reject_unsafe_quadrature_inputs(keyword, value, match):
    values = {
        "projection_width_A": 5.431,
        "cutoff_A": 0.4,
        "pixel_quadrature_order": 2,
        "projection_quadrature_order": 24,
        "maximum_quadrature_evaluations": 50_000_000,
    }
    values[keyword] = value
    with pytest.raises((TypeError, ValueError), match=match):
        AtomicTemplateQuadratureOptions1D(**values)


def test_render_and_grid_reject_ambiguous_geometry():
    with pytest.raises(ValueError, match="fractional s offset"):
        _render(offset=(0.1, 0.0))
    with pytest.raises(ValueError, match="does not reach cutoff_A"):
        render_si_atomic_template_1d(
            sampling_s_A=0.2,
            sampling_u_A=0.2,
            options=_options(),
            half_shape=(1, 2),
        )
    with pytest.raises(ValueError, match="maximum_quadrature_evaluations"):
        render_si_atomic_template_1d(
            sampling_s_A=0.1,
            sampling_u_A=0.1,
            options=replace(_options(), maximum_quadrature_evaluations=1_000),
        )
    with pytest.raises(ValueError, match="uniformly sampled"):
        accumulate_si_atomic_potential_1d(
            [[0.0, 0.0]],
            s_coordinates_A=[0.0, 0.2, 0.41],
            u_coordinates_A=[0.0, 0.2, 0.4],
            options=_options(),
        )
    with pytest.raises(ValueError, match="2 dimensions"):
        accumulate_si_atomic_potential_1d(
            [0.0, 0.0],
            s_coordinates_A=[0.0, 0.2],
            u_coordinates_A=[0.0, 0.2],
            options=_options(),
        )


def test_comparison_and_result_objects_reject_missing_evidence_or_tampering():
    candidate = _render()
    with pytest.raises(ValueError, match="must not be empty"):
        compare_si_atomic_template_1d(
            candidate,
            candidate.values,
            reference_provenance=(),
        )
    with pytest.raises(ValueError, match="same-grid|shape"):
        compare_si_atomic_template_1d(
            candidate,
            candidate.values[:-1],
            reference_provenance={"source": "wrong shape"},
        )
    with pytest.raises(ValueError, match="finite"):
        bad_reference = candidate.values.copy()
        bad_reference[0, 0] = np.nan
        compare_si_atomic_template_1d(
            candidate,
            bad_reference,
            reference_provenance={"source": "non-finite"},
        )
    with pytest.raises(ValueError, match="non-negative"):
        bad_reference = candidate.values.copy()
        bad_reference[0, 0] = -1.0
        compare_si_atomic_template_1d(
            candidate,
            bad_reference,
            reference_provenance={"source": "negative"},
        )
    with pytest.raises(ValueError, match="does not match"):
        replace(candidate, template_sha256="0" * 64)
    with pytest.raises(ValueError, match="fail closed"):
        replace(candidate, trust_claim=True)
