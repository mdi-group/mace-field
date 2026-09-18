"""Create deterministic validation files for structured response branches.

The finite-field and MP-ferroelectric sources contain several correlated
frames for each material. A frame-random split can put adjacent points from
the same branch into both sets and can leave validation at an unrepresentative
field or polarization value. This script keeps one deterministic random
interior sample from every branch for validation and writes the remaining
samples to a separate training file.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Callable, Hashable

import numpy as np

from common import DATA_DIR, MANIFEST_DIR, read_frames, sha256, write_frames, write_json


FrameKey = Callable[[object], Hashable]


def _random_interior_value(
    values: list[Hashable], rng: np.random.Generator
) -> Hashable:
    """Return one reproducible random value excluding both endpoints."""

    ordered = sorted(set(values))
    if len(ordered) < 3:
        raise ValueError("Each branch needs at least three ordered samples")
    return ordered[int(rng.integers(1, len(ordered) - 1))]


def split_branch_frames(
    frames: list[object],
    *,
    group_key: FrameKey,
    sample_key: FrameKey,
    seed: int = 123,
) -> tuple[list[object], list[object], int]:
    """Split frames by branch, selecting one interior sample for validation."""

    groups: dict[Hashable, list[tuple[int, Hashable]]] = defaultdict(list)
    for index, frame in enumerate(frames):
        groups[group_key(frame)].append((index, sample_key(frame)))

    rng = np.random.default_rng(seed)
    validation_indices: set[int] = set()
    excluded_indices: set[int] = set()
    for members in groups.values():
        selected = _random_interior_value([value for _, value in members], rng)
        selected_indices = [index for index, value in members if value == selected]
        # Keep exactly one physical frame for each logical validation point.
        # Any duplicate records at that point are excluded from training too,
        # so an identical frame cannot occur in both partitions.
        validation_indices.add(selected_indices[0])
        excluded_indices.update(selected_indices)

    validation = [
        frame for index, frame in enumerate(frames) if index in validation_indices
    ]
    training = [
        frame for index, frame in enumerate(frames) if index not in excluded_indices
    ]
    return training, validation, len(groups)


def _finite_field_group(frame: object) -> tuple[str, str]:
    return (str(frame.info["material_formula"]), str(frame.info["structure_type"]))


def _finite_field_sample(frame: object) -> tuple[float, tuple[float, ...]]:
    field = np.asarray(frame.info["REF_electric_field"], dtype=float).reshape(3)
    return (float(np.linalg.norm(field)), tuple(float(value) for value in field))


def _mp_ferroelectric_group(frame: object) -> str:
    return str(frame.info["source_contribution_id"])


def _mp_ferroelectric_sample(frame: object) -> int:
    return int(frame.info["source_workflow_index"])


SPECS = {
    "finite-field-ferroelectric": {
        "input": "finite-field-ferroelectric.extxyz",
        "group_key": _finite_field_group,
        "sample_key": _finite_field_sample,
        "description": "one random interior electric-field sample per material and structure branch; duplicate selected records excluded",
    },
    "MP-ferroelectric": {
        "input": "MP-ferroelectric.xyz",
        "group_key": _mp_ferroelectric_group,
        "sample_key": _mp_ferroelectric_sample,
        "description": "one random interior workflow sample per MPContribs material branch; duplicate selected records excluded",
    },
}


def make_split(
    name: str,
    *,
    input_path: Path,
    output_dir: Path,
    manifest_path: Path | None = None,
    seed: int = 123,
) -> dict:
    """Build one branch-aware train/validation pair and its manifest."""

    spec = SPECS[name]
    frames = read_frames(input_path)
    training, validation, group_count = split_branch_frames(
        frames,
        group_key=spec["group_key"],
        sample_key=spec["sample_key"],
        seed=seed,
    )
    if not training or not validation:
        raise ValueError(f"Branch split for {name} produced an empty partition")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{name}_train.extxyz"
    valid_path = output_dir / f"{name}_valid.extxyz"
    write_frames(train_path, training)
    write_frames(valid_path, validation)

    manifest = {
        "dataset": name,
        "input": str(input_path),
        "train_output": str(train_path),
        "valid_output": str(valid_path),
        "source_frames": len(frames),
        "train_frames": len(training),
        "valid_frames": len(validation),
        "excluded_duplicate_frames": len(frames) - len(training) - len(validation),
        "branches": group_count,
        "seed": seed,
        "selection": spec["description"],
        "train_sha256": sha256(train_path),
        "valid_sha256": sha256(valid_path),
    }
    write_json(
        manifest_path or MANIFEST_DIR / f"{name}_branch_split.json",
        manifest,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", *SPECS],
        default="all",
        help="dataset split to build (default: all)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_DIR / "cleaned",
        help="directory containing cleaned source files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "branch_splits",
        help="directory for disjoint train/validation files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="seed for selecting one interior sample per branch",
    )
    args = parser.parse_args()

    names = list(SPECS) if args.dataset == "all" else [args.dataset]
    for name in names:
        manifest = make_split(
            name,
            input_path=args.input_dir / SPECS[name]["input"],
            output_dir=args.output_dir,
            seed=args.seed,
        )
        print(
            f"{name}: {manifest['train_frames']} train, "
            f"{manifest['valid_frames']} validation across "
            f"{manifest['branches']} branches"
        )


if __name__ == "__main__":
    main()
