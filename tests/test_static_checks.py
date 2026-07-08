"""Tests for dependency-light repository validation helpers."""

import json

import pytest

from scripts import check_static


def test_python_sources_parse_and_public_exports_exist():
    check_static.parse_python_files()
    check_static.assert_exports_exist(
        check_static.ROOT / "wide_angle_propagation" / "propagation_methods.py"
    )
    check_static.assert_exports_exist(
        check_static.ROOT / "wide_angle_propagation" / "notebook_utils.py"
    )


def test_notebook_output_policy_is_optional(tmp_path, monkeypatch):
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [{"name": "stdout", "output_type": "stream", "text": ["ok\n"]}],
                "source": ["value = 1\n", "print(value)\n"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = tmp_path / "with_outputs.ipynb"
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

    monkeypatch.setattr(check_static, "ROOT", tmp_path)
    monkeypatch.setattr(check_static, "TARGET_NOTEBOOKS", [notebook_path])

    check_static.parse_target_notebooks(require_no_outputs=False)
    with pytest.raises(AssertionError, match="contains 1 code cells with outputs"):
        check_static.parse_target_notebooks(require_no_outputs=True)
