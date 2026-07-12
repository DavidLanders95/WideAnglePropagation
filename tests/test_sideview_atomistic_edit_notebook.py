"""Static contract for the maintained sparse-edit side-view notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "sideview_glancing_ptychography_1d.ipynb"
)


def _source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"])


def test_sideview_notebook_has_unique_ids_and_valid_python_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    identifiers = [cell.get("id") for cell in notebook["cells"]]
    assert all(identifiers)
    assert len(identifiers) == len(set(identifiers))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ()))
        source = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("%")
        )
        ast.parse(source, filename=f"{NOTEBOOK.name}:cell{index}")


def test_sideview_notebook_uses_only_the_sparse_edit_facade_for_inversion():
    source = _source()
    required = {
        "SiliconAtomisticEditConfig1D",
        "reconstruct_silicon_atomistic_edits_1d",
        "plot_silicon_atomistic_edit_run_1d",
        "save_silicon_atomistic_edit_run_1d",
        "load_silicon_atomistic_edit_run_1d",
        "summarize_silicon_atomistic_edit_run_1d",
    }
    assert all(name in source for name in required)
    forbidden = {
        "reconstruct_pixel_potential_1d",
        "reconstruct_lattice_site_potential_1d",
        "run_prepared_lattice_site_reconstruction_1d",
        "prepare_atomistic_edit_experiment_1d",
        "run_prepared_atomistic_edit_reconstruction_1d",
        "summarize_atomistic_edit_reconstruction_1d",
    }
    assert all(name not in source for name in forbidden)
    assert "experiment.summary" not in source


def test_sideview_notebook_contains_complex_truth_target_view_and_evolution():
    source = _source()
    assert 'DATASET_CASE = "strained_surface_defects"' in source
    assert "ae_progress_events.append" in source
    assert "truth_potential=np.asarray(dataset.potential)" in source
    assert "TARGET-only evolution GIF" in source
    assert "event.state" in source
    assert '"per-Adam-update history claimed": False' in source
