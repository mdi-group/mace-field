"""Small smoke tests for public CLI entry points touched by extensions."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mace.cli.create_lammps_model import select_head


REPO_ROOT = Path(__file__).resolve().parents[2]


CLI_HELP_CASES = {
    "active_learning_md": ("--model_type", "--electric-field"),
    "create_lammps_model": ("--electric-field",),
    "eval_configs": ("--compute_polarization", "--compute_becs"),
    "fine_tuning_select": ("--model_type", "--head", "--electric-field"),
    "run_train": ("--model", "--compute_polarization"),
    "plot_train": ("--path",),
    "preprocess_data": ("--config",),
    "select_head": ("--head_name",),
    "convert_device": ("--target_device",),
}


@pytest.mark.parametrize("cli, expected_options", CLI_HELP_CASES.items())
def test_affected_cli_help_surfaces(cli, expected_options):
    """Every affected script parses ``--help`` from the source checkout."""
    command = [
        sys.executable,
        str(REPO_ROOT / "mace" / "cli" / f"{cli}.py"),
        "--help",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    help_text = result.stdout + result.stderr
    for option in expected_options:
        assert option in help_text


def test_create_lammps_select_head_preserves_single_head():
    """The generic LAMMPS head selector remains model-agnostic."""

    class SingleHeadModel:
        heads = ["Default"]

    assert select_head(SingleHeadModel()) == "Default"
