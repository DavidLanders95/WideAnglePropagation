"""Dependency-light tests for propagation timing result summaries."""

import json

import pytest

from scripts.benchmark_propagation_timing import (
    make_result_payload,
    summarize_timings,
)


def test_summarize_timings_uses_median_quartiles_and_as_baseline():
    summary = summarize_timings(
        {
            "F-MS": [0.9, 1.0, 1.1, 1.2],
            "AS-MS": [1.0, 2.0, 3.0, 4.0],
            "WP-MS-64": [5.0, 6.0, 7.0, 8.0],
        }
    )
    by_name = {row["method"]: row for row in summary}

    assert by_name["AS-MS"] == {
        "method": "AS-MS",
        "median_s": pytest.approx(2.5),
        "q1_s": pytest.approx(1.75),
        "q3_s": pytest.approx(3.25),
        "relative_to_as": pytest.approx(1.0),
    }
    assert by_name["WP-MS-64"]["relative_to_as"] == pytest.approx(2.6)


def test_result_payload_is_json_serializable_and_keeps_raw_timings():
    payload = make_result_payload(
        {"gpu": "test device"},
        {"F-MS": [1.0, 1.1], "AS-MS": [1.2, 1.3]},
    )

    assert payload["metadata"]["gpu"] == "test device"
    assert payload["timings_s"]["F-MS"] == [1.0, 1.1]
    json.dumps(payload)


@pytest.mark.parametrize(
    "timings",
    [
        {"F-MS": [1.0]},
        {"AS-MS": []},
        {"AS-MS": [float("nan")]},
        {"AS-MS": [0.0]},
    ],
)
def test_summarize_timings_rejects_invalid_input(timings):
    with pytest.raises(ValueError):
        summarize_timings(timings)
