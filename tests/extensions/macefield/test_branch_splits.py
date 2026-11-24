"""Tests for branch-aware ferroelectric validation splits."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase import Atoms

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "foundation" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from make_branch_splits import (  # noqa: E402
    _finite_field_group,
    _finite_field_sample,
    _mp_ferroelectric_group,
    _mp_ferroelectric_sample,
    split_branch_frames,
)


def test_finite_field_split_keeps_one_random_interior_sample_per_branch() -> None:
    frames = []
    field_values = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025]
    for material in ("A", "B"):
        for branch in ("polar", "nonpolar"):
            for value in field_values:
                frames.append(
                    Atoms(
                        "H",
                        positions=[[0.0, 0.0, 0.0]],
                        cell=np.eye(3),
                        pbc=True,
                        info={
                            "material_formula": material,
                            "structure_type": branch,
                            "REF_electric_field": np.array([0.0, 0.0, value]),
                        },
                    )
                )

    train, valid, branches = split_branch_frames(
        frames,
        group_key=_finite_field_group,
        sample_key=_finite_field_sample,
        seed=7,
    )

    assert branches == 4
    assert len(train) == 20
    assert len(valid) == 4
    assert all(
        0.0 < float(frame.info["REF_electric_field"][2]) < 0.0025 for frame in valid
    )
    assert (
        len(
            {
                (frame.info["material_formula"], frame.info["structure_type"])
                for frame in valid
            }
        )
        == 4
    )
    assert {
        (frame.info["material_formula"], frame.info["structure_type"])
        for frame in train
    } == {
        ("A", "polar"),
        ("A", "nonpolar"),
        ("B", "polar"),
        ("B", "nonpolar"),
    }


def test_mp_ferroelectric_split_keeps_one_random_interior_workflow_per_material() -> (
    None
):
    frames = []
    for material in ("id-a", "id-b"):
        for workflow in range(10):
            frames.append(
                Atoms(
                    "H",
                    positions=[[float(workflow), 0.0, 0.0]],
                    cell=np.eye(3),
                    pbc=True,
                    info={
                        "source_contribution_id": material,
                        "source_workflow_index": workflow,
                    },
                )
            )

    # With seed 7, workflow 8 is selected for the first branch.  Include a
    # duplicate there to verify that only one copy is validated and the other
    # copy is excluded rather than leaked into training.
    frames.append(frames[8].copy())

    train, valid, branches = split_branch_frames(
        frames,
        group_key=_mp_ferroelectric_group,
        sample_key=_mp_ferroelectric_sample,
        seed=7,
    )

    assert branches == 2
    assert len(train) == 18
    assert len(valid) == 2
    assert all(0 < int(frame.info["source_workflow_index"]) < 9 for frame in valid)
    assert len({frame.info["source_contribution_id"] for frame in valid}) == 2
    validation_keys = {
        (
            frame.info["source_contribution_id"],
            int(frame.info["source_workflow_index"]),
        )
        for frame in valid
    }
    assert {
        (
            frame.info["source_contribution_id"],
            int(frame.info["source_workflow_index"]),
        )
        for frame in train
    }.isdisjoint(validation_keys)
    assert {frame.info["source_contribution_id"] for frame in train} == {"id-a", "id-b"}
