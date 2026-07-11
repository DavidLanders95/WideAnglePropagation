from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_ptychography_1d.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "benchmark_ptychography_1d_test_module", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_cli_is_dependency_lazy_and_exposes_required_controls():
    module = _load_harness()
    assert "jax" not in module.__dict__
    assert "numpy" not in module.__dict__

    arguments = module.build_parser().parse_args(
        [
            "--quick",
            "--updates",
            "2",
            "--starts",
            "3",
            "--precision",
            "float32",
            "--device",
            "cpu",
            "--output",
            "result.json",
        ]
    )
    assert arguments.quick
    assert arguments.updates == 2
    assert arguments.starts == 3
    assert arguments.precision == "float32"
    assert arguments.device == "cpu"
    assert arguments.training_diagnostic_scans == 32
    assert arguments.output == "result.json"

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    for option in (
        "--quick",
        "--updates",
        "--starts",
        "--precision",
        "--device",
        "--training-diagnostic-scans",
    ):
        assert option in completed.stdout


def test_allocator_and_platform_environment_are_set_before_runtime_import():
    code = f"""
import json
import runpy
module = runpy.run_path({str(SCRIPT)!r}, run_name='benchmark_test_module')
print(json.dumps(module['_configure_environment'](precision='float64', device='cpu')))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    environment = json.loads(completed.stdout)
    assert environment == {
        "JAX_ENABLE_X64": "true",
        "JAX_PLATFORMS": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }


def test_json_report_writer_is_atomic_and_strict(tmp_path):
    module = _load_harness()
    output = tmp_path / "nested" / "benchmark.json"
    module._write_report(
        {
            "schema": module.REPORT_SCHEMA,
            "schema_version": module.REPORT_SCHEMA_VERSION,
            "finite": 2.5,
            "not_finite": [float("nan"), float("inf"), -float("inf")],
        },
        str(output),
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == module.REPORT_SCHEMA
    assert report["schema_version"] == 1
    assert report["finite"] == 2.5
    assert report["not_finite"] == [None, None, None]
    assert not (output.parent / f".{output.name}.tmp").exists()
