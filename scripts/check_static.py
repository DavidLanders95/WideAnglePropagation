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
    ROOT / "wide_angle_propagation" / "ptychography_1d.py",
    ROOT / "wide_angle_propagation" / "ptychography_crystal_1d.py",
]
TARGET_NOTEBOOKS = [
    ROOT / "notebooks" / "sideview_glancing_incidence_example.ipynb",
    ROOT / "notebooks" / "sideview_glancing_ptychography_1d.ipynb",
    ROOT / "notebooks" / "sideview_glancing_silicon_viewer_1d.ipynb",
    ROOT / "notebooks" / "figure_generation" / "01_axel_lubk_verification.ipynb",
    ROOT / "notebooks" / "figure_generation" / "02_converge_probe_si.ipynb",
    ROOT / "notebooks" / "figure_generation" / "03_convergent_probe_au.ipynb",
    ROOT / "notebooks" / "figure_generation" / "04_wpm_binning_diagnostics.ipynb",
]
PTYCHOGRAPHY_NOTEBOOK = ROOT / "notebooks" / "sideview_glancing_ptychography_1d.ipynb"


def parse_python_files() -> None:
    for path in PYTHON_FILES:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        print(f"{path.relative_to(ROOT)}: ast ok")


def assert_exports_exist(path: Path) -> None:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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


def literal_exports(path: Path) -> tuple[str, ...]:
    """Return the literal public export tuple without importing the module."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{path.relative_to(ROOT)} has no literal __all__")


def assert_ptychography_contract() -> None:
    """Check the reduced APIs and the maintained full-slab notebook contract."""
    generic_path = ROOT / "wide_angle_propagation" / "ptychography_1d.py"
    crystal_path = ROOT / "wide_angle_propagation" / "ptychography_crystal_1d.py"
    expected_generic = (
        "GlancingSideviewCache1D",
        "normalized_amplitude_loss_1d",
        "simulate_glancing_scan_1d",
        "simulate_glancing_sideview_cache_1d",
    )
    expected_crystal = (
        "CrystalModel1D",
        "CrystalState1D",
        "CrystalReconstruction1D",
        "make_crystal_model_1d",
        "make_si_atom_template_1d",
        "render_crystal_1d",
        "reconstruct_crystal_1d",
    )
    if literal_exports(generic_path) != expected_generic:
        raise AssertionError("ptychography_1d.py does not expose the four-primitive API")
    if literal_exports(crystal_path) != expected_crystal:
        raise AssertionError("ptychography_crystal_1d.py does not expose the unified API")
    retired_paths = (
        ROOT / "wide_angle_propagation" / "ptychography_atoms_1d.py",
        ROOT / "tests" / "test_ptychography_atoms_1d.py",
        ROOT / "notebooks" / "nine_atom_free_atom_ptychography_1d.ipynb",
    )
    remaining = [path.relative_to(ROOT) for path in retired_paths if path.exists()]
    if remaining:
        raise AssertionError(f"retired ptychography files remain: {remaining}")

    notebook = json.loads(PTYCHOGRAPHY_NOTEBOOK.read_text(encoding="utf-8"))
    identifiers = [cell.get("id") for cell in notebook.get("cells", ())]
    if not identifiers or not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise AssertionError("ptychography notebook cell IDs must be present and unique")
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    if any(cell.get("outputs") or cell.get("execution_count") is not None for cell in code_cells):
        raise AssertionError("ptychography notebook must have clean outputs and execution counts")
    code = "\n".join("".join(cell.get("source", ())) for cell in code_cells)
    markdown = "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    ).lower()
    code_lines = sum(len("".join(cell.get("source", ())).splitlines()) for cell in code_cells)
    if code_lines > 225:
        raise AssertionError(f"ptychography notebook has {code_lines} code lines; expected at most 225")
    required_code = (
        "propagation_length_A",
        "slab_depth_A",
        "1000.0, 50.0",
        "30_000.0, 2.0, 15.0",
        "np.linspace(400.0, 600.0, 21)",
        "training_indices",
        "selection_indices",
        "audit_indices",
        "reconstruct_crystal_1d",
        "progress=True",
        "make_crystal_reconstruction_viewer_1d",
    )
    missing = [token for token in required_code if token not in code]
    if missing:
        raise AssertionError(f"ptychography notebook is missing required code: {missing}")
    forbidden_code = ("import optax", "jax.value_and_grad", "fftconvolve", "_keating")
    present = [token for token in forbidden_code if token in code]
    if present:
        raise AssertionError(f"notebook contains package-level reconstruction code: {present}")
    required_prose = ("temporary", "discard", "fixed latent", "noise", "probe", "limitations")
    missing_prose = [token for token in required_prose if token not in markdown]
    if missing_prose:
        raise AssertionError(f"ptychography notebook is missing explanations: {missing_prose}")
    if "from tqdm.auto import tqdm" not in crystal_path.read_text(encoding="utf-8"):
        raise AssertionError("crystal reconstruction does not provide TQDM progress")
    print("ptychography workflow: reduced APIs and notebook contract ok")


def parse_target_notebooks(*, require_no_outputs: bool = False) -> None:
    for path in TARGET_NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
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
    assert_exports_exist(ROOT / "wide_angle_propagation" / "ptychography_1d.py")
    assert_exports_exist(
        ROOT / "wide_angle_propagation" / "ptychography_crystal_1d.py"
    )
    parse_target_notebooks(require_no_outputs=args.enforce_clean_notebooks)
    assert_ptychography_contract()


if __name__ == "__main__":
    main()
