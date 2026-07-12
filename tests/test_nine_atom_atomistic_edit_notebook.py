"""Contracts for the executable nine-atom atomistic-edit teaching notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "nine_atom_atomistic_edit_ptychography_1d.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _call_names(node: ast.AST) -> set[str]:
    result: set[str] = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
            result.add(item.func.id)
    return result


def test_notebook_uses_only_the_public_sparse_atomistic_edit_path():
    notebook = _notebook()
    sources = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    module = ast.parse("\n\n".join(sources), filename=str(NOTEBOOK))
    calls = _call_names(module)

    required = {
        "classify_lattice_site_support_1d",
        "make_atomistic_edit_discovery_support_1d",
        "make_atomistic_edit_kernel_1d",
        "make_atomistic_edit_model_1d",
        "empty_atomistic_edit_state_1d",
        "render_atomistic_edit_potential_1d",
        "prepare_atomistic_edit_reconstruction_1d",
        "atomistic_edit_objective_components_1d",
        "run_prepared_atomistic_edit_reconstruction_1d",
    }
    assert required <= calls

    source = "\n".join(sources)
    forbidden = {
        "LatticeSiteReconstruction1D",
        "reconstruct_lattice_sites_1d",
        "reconstruct_potential_1d",
        "build_silicon_glancing_experiment",
    }
    assert not any(name in source for name in forbidden)

    imported = [node for node in ast.walk(module) if isinstance(node, ast.ImportFrom)]
    assert all(
        not alias.name.startswith("_")
        for node in imported
        for alias in node.names
    )


def test_reconstruction_calls_cannot_receive_private_truth_names():
    notebook = _notebook()
    module = ast.parse(
        "\n\n".join("".join(cell["source"]) for cell in _code_cells(notebook)),
        filename=str(NOTEBOOK),
    )
    guarded_calls = {
        "prepare_atomistic_edit_reconstruction_1d",
        "run_prepared_atomistic_edit_reconstruction_1d",
    }
    found: set[str] = set()
    for node in ast.walk(module):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in guarded_calls
        ):
            continue
        found.add(node.func.id)
        argument_names = {
            item.id
            for argument in (*node.args, *(item.value for item in node.keywords))
            for item in ast.walk(argument)
            if isinstance(item, ast.Name)
        }
        assert not any("truth" in name.lower() for name in argument_names)
    assert found == guarded_calls


def test_saved_notebook_is_executed_and_retains_key_evidence():
    notebook = _notebook()
    code_cells = _code_cells(notebook)
    assert notebook["nbformat"] == 4
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(isinstance(cell_id, str) and cell_id for cell_id in cell_ids)
    assert len(cell_ids) == len(set(cell_ids))
    assert code_cells
    assert all(isinstance(cell.get("execution_count"), int) for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )

    text_outputs = "\n".join(
        "".join(output.get("text", []))
        if isinstance(output.get("text", []), list)
        else str(output.get("text", ""))
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    assert "Empty-edit identity is exact: True" in text_outputs
    assert "Private Level-1 state is admissible: True" in text_outputs
    assert "Reconstruction boundary is truth-free: True" in text_outputs
    assert "RESULT converged=True" in text_outputs
    assert "active parameters=12" in text_outputs
    assert "TARGET-only displayed pixels:" in text_outputs

    assert any(
        "image/png" in output.get("data", {})
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") in {"display_data", "execute_result"}
    )


def test_every_code_cell_parses_independently():
    for index, cell in enumerate(_code_cells(_notebook())):
        ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK}:cell{index}")
