"""Build a deterministic, structure-only replay set for MACE-MH-1."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import DATA_DIR, MANIFEST_DIR, ensure_workspace, manifest_for, read_frames, write_json, write_frames


DEFAULT_INPUTS = (
    "MP-Dielectric.xyz",
    "MP-ferroelectric.xyz",
    "finite-field-ferroelectric.extxyz",
    "JarvisDB.xyz",
    "C2DB.xyz",
    "Togo.xyz",
)


def default_inputs() -> list[Path]:
    """Prefer the audited copies, while retaining a raw-data fallback."""

    cleaned_dir = DATA_DIR / "cleaned"
    return [
        (cleaned_dir / name if (cleaned_dir / name).exists() else DATA_DIR / name)
        for name in DEFAULT_INPUTS
    ]


def _strip_labels(atoms):
    atoms = atoms.copy()
    for key in list(atoms.info):
        if key.startswith(("REF_", "config_", "MACE_", "MP_", "C2DB_", "JARVIS_", "Togo_")):
            del atoms.info[key]
    for key in list(atoms.arrays):
        if key.startswith(("REF_", "MACE_")):
            del atoms.arrays[key]
    return atoms


def collect(inputs: list[Path], output: Path, *, samples: int | None, seed: int, validation_fraction: float) -> dict:
    ensure_workspace()
    frames = []
    source_counts = {}
    for path in inputs:
        if not path.exists():
            continue
        source_frames = [_strip_labels(atoms) for atoms in read_frames(path)]
        source_counts[str(path)] = len(source_frames)
        frames.extend(source_frames)
    if not frames:
        raise SystemExit("No input datasets exist; run the collectors before making replay data")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frames))
    if samples is not None:
        order = order[: min(samples, len(order))]
    selected = [frames[index] for index in order]
    for index, atoms in enumerate(selected):
        atoms.info["replay_source_index"] = int(index)
    written = write_frames(output, selected)
    valid_count = 0
    if validation_fraction > 0 and len(selected) > 1:
        valid_count = max(1, int(round(len(selected) * validation_fraction)))
        valid_count = min(valid_count, len(selected) - 1)
        valid_path = output.with_name(output.stem + "_valid" + output.suffix)
        write_frames(valid_path, selected[:valid_count])
    manifest = manifest_for(
        dataset="MACE-MH-1 replay",
        output=output,
        frames=written,
        labels={"energy": 0, "forces": 0, "stress": 0, "polarization": 0, "becs": 0, "polarizability": 0},
        sources=[{"path": str(path), "frames": count} for path, count in source_counts.items()],
        notes=[
            "All source target labels are removed; run_train generates E/F/(optional stress) pseudolabels from the plain MACE-MH-1 foundation model.",
            f"Deterministic permutation seed: {seed}.",
            f"Validation frames written separately: {valid_count}.",
        ],
    )
    write_json(MANIFEST_DIR / "mh1_replay.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, dest="inputs")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "mh1_replay.xyz")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    args = parser.parse_args()
    inputs = args.inputs or default_inputs()
    manifest = collect(inputs, args.output, samples=args.samples, seed=args.seed, validation_fraction=args.validation_fraction)
    print(f"Wrote {manifest['frames']} replay frames to {manifest['output']}")


if __name__ == "__main__":
    main()
