"""Tests for the MACE-MH-1 pseudolabel replay split."""

from __future__ import annotations

import sys
from pathlib import Path

import ase.io
from ase import Atoms

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "foundation" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from make_replay_set import collect  # noqa: E402
from common import read_frames  # noqa: E402


def test_replay_validation_is_excluded_from_training(tmp_path: Path) -> None:
    source = tmp_path / "source.xyz"
    output = tmp_path / "replay.xyz"
    frames = [Atoms("H", positions=[[float(index), 0.0, 0.0]]) for index in range(10)]
    ase.io.write(source, frames, format="extxyz")

    manifest = collect(
        [source],
        output,
        samples=10,
        seed=123,
        validation_fraction=0.2,
        manifest_path=tmp_path / "manifest.json",
    )

    train = read_frames(output)
    valid = read_frames(output.with_name("replay_valid.xyz"))
    train_ids = {int(atoms.info["replay_source_index"]) for atoms in train}
    valid_ids = {int(atoms.info["replay_source_index"]) for atoms in valid}

    assert len(train) == 8
    assert len(valid) == 2
    assert train_ids.isdisjoint(valid_ids)
    assert manifest["frames"] == 8
    assert manifest["train_frames"] == 8
    assert manifest["validation_frames"] == 2
