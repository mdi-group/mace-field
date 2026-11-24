"""Tests for response-label requirements in foundation dataset cleaning."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase import Atoms

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "foundation" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from clean_validate_datasets import clean_dataset  # noqa: E402
from common import read_frames  # noqa: E402


def test_required_response_labels_remove_response_free_frames(tmp_path: Path) -> None:
    bec_frame = Atoms(
        "H",
        positions=[[0.0, 0.0, 0.0]],
        cell=np.eye(3),
        pbc=True,
    )
    bec_frame.arrays["REF_becs"] = np.zeros((1, 9))
    frames = [
        Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True),
        Atoms(
            "H",
            positions=[[0.0, 0.0, 0.0]],
            cell=np.eye(3),
            pbc=True,
            info={"REF_polarizability": np.eye(3)},
        ),
        bec_frame,
    ]
    source = tmp_path / "MP-Dielectric.xyz"
    output = tmp_path / "cleaned.xyz"

    import ase.io

    ase.io.write(source, frames, format="extxyz")
    summary = clean_dataset(
        source,
        output,
        asr_tolerance=0.25,
        max_energy_per_atom=1.0e4,
        max_force=1.0e3,
        max_stress=1.0e3,
        max_bec=100.0,
        max_polarization=1.0e3,
        max_polarizability=1.0e6,
        max_response_atoms=128,
        required_labels=("becs", "polarizability"),
    )

    assert summary["frames_input"] == 3
    assert summary["frames_output"] == 2
    assert summary["dropped_reasons"] == {"missing_required_response_labels": 1}
    assert len(read_frames(output)) == 2
