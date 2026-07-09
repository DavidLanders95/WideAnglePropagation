"""Dependency-light repository checks for CI and publication cleanup.

The checks intentionally avoid importing the package, because the scientific
runtime stack (JAX, CuPy, abTEM) is often installed separately from the base
Python used for lightweight CI.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_FILES = [
    ROOT / "setup.py",
    ROOT / "wide_angle_propagation" / "__init__.py",
    ROOT / "wide_angle_propagation" / "propagation_methods.py",
    ROOT / "wide_angle_propagation" / "sideview_geometry.py",
    ROOT / "wide_angle_propagation" / "notebook_utils.py",
]
TARGET_NOTEBOOKS = [
    ROOT / "notebooks" / "sideview_glancing_incidence_example.ipynb",
    ROOT / "notebooks" / "sideview_glancing_benchmark.ipynb",
    ROOT / "notebooks" / "figure_generation" / "01_axel_lubk_verification.ipynb",
    ROOT / "notebooks" / "figure_generation" / "02_converge_probe_si.ipynb",
    ROOT / "notebooks" / "figure_generation" / "03_convergent_probe_au.ipynb",
    ROOT / "notebooks" / "figure_generation" / "04_wpm_binning_diagnostics.ipynb",
]


def parse_python_files() -> None:
    for path in PYTHON_FILES:
        ast.parse(path.read_text(), filename=str(path))
        print(f"{path.relative_to(ROOT)}: ast ok")


def assert_exports_exist(path: Path) -> None:
    module = ast.parse(path.read_text(), filename=str(path))
    defined = set()
    exported = None

    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                exported = ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)

    if exported is None:
        raise AssertionError(f"{path.relative_to(ROOT)} has no __all__")

    missing = [name for name in exported if name not in defined]
    if missing:
        raise AssertionError(f"{path.relative_to(ROOT)} exports undefined names: {missing}")

    print(f"{path.relative_to(ROOT)}: {len(exported)} exports ok")


def parse_target_notebooks(*, require_no_outputs: bool = False) -> None:
    for path in TARGET_NOTEBOOKS:
        notebook = json.loads(path.read_text())
        output_cells = 0

        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            output_cells += bool(cell.get("outputs"))
            source = "".join(cell.get("source", []))
            source = "\n".join(
                line for line in source.splitlines()
                if not line.lstrip().startswith("%")
            )
            ast.parse(source, filename=f"{path.relative_to(ROOT)}:cell{index}")

        if require_no_outputs and output_cells:
            raise AssertionError(
                f"{path.relative_to(ROOT)} contains {output_cells} code cells with outputs"
            )
        output_note = (
            "no outputs"
            if output_cells == 0
            else f"{output_cells} code cell(s) with outputs"
        )
        print(f"{path.relative_to(ROOT)}: code-cell syntax ok, {output_note}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run dependency-light static checks for the repository.",
    )
    parser.add_argument(
        "--enforce-clean-notebooks",
        action="store_true",
        help="fail if maintained notebooks contain saved code-cell outputs",
    )
    args = parser.parse_args(argv)

    parse_python_files()
    assert_exports_exist(ROOT / "wide_angle_propagation" / "propagation_methods.py")
    assert_exports_exist(ROOT / "wide_angle_propagation" / "sideview_geometry.py")
    assert_exports_exist(ROOT / "wide_angle_propagation" / "notebook_utils.py")
    parse_target_notebooks(require_no_outputs=args.enforce_clean_notebooks)


if __name__ == "__main__":
    main()
